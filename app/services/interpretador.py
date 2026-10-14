"""
Interpretacao da fala do jogador, SEM chamar modelo.

Por que existe: no bug do turno 11, "eu matei mesmo fdp" voltou ao detetive
como se fosse uma negacao, e ele repetiu a pergunta sobre o horario. O modelo
pequeno nao recebia nenhuma leitura da fala — so a lista crua do que o suspeito
ja disse. Aqui a fala e classificada por regras de FRASE (verbo + escopo de
negacao), nao por palavra solta: "nao" sozinho nao e negacao de autoria, e
"nao nego que matei" e confissao. O resultado entra no prompt como fato
("ANALISE DA FALA") e atualiza o estado da investigacao.

Custo: zero chamadas extras de IA; roda em microssegundos.

Limite assumido: e heuristica. Ironia ("claro, eu matei, ne?") e lida como
confissao POSSIVEL — que e exatamente o rotulo usado: o detetive trata como
declaracao a verificar, nao como prova.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

CONFISSAO = "confissao"
NEGACAO = "negacao"
EVASIVA = "evasiva"
INFORMACAO = "informacao"
AMBIGUA = "ambigua"

_NEGADORES = {"nao", "nunca", "jamais", "nem", "ninguem"}
# "nao nego", "nao vou negar", "nao adianta negar": a negacao recai sobre NEGAR,
# o que inverte o sentido.
_NEGAR = {"nego", "negar", "negarei", "negaria"}
_HIPOTESE = {"se", "suponha", "imagina", "imagine", "hipoteticamente", "fosse", "tivesse"}

# Verbos de autoria em 1a pessoa do preterito. O sufixo -ei ja marca "eu"; a
# lista e fechada de proposito para nao confundir com outros verbos.
_VERBOS_AUTORIA = {
    # Fora de proposito verbos de dupla leitura ("bati o ponto", "apaguei a luz",
    # "empurrei a porta"): so entram os que, no contexto de um homicidio, nao
    # tem outro uso corrente.
    "matei", "assassinei", "esfaqueei", "esfaquiei", "golpeei",
    "agredi", "estrangulei", "sufoquei", "espanquei",
}
# Formas compostas: "eu o matei", "eu que matei" ja caem em _VERBOS_AUTORIA;
# estas sao as sem verbo de morte.
_FRASES_CONFISSAO = [
    r"\bfui eu( mesmo)?( que| quem)?\b",
    r"\bfoi eu( mesmo)?( que| quem)?\b",
    r"\ba culpa e minha\b",
    r"\bsou o (culpado|assassino|responsavel)\b",
    r"\bsou (culpado|culpada|assassino|assassina|responsavel)\b",
    r"\beu (fiz|causei) (isso|aquilo|tudo)\b",
    r"\bcometi (o|um|esse|este) (crime|assassinato|homicidio)\b",
    r"\bconfesso\b",
]
_FRASES_NEGACAO = [
    r"\bnao fui eu\b", r"\bnao foi eu\b",
    r"\bnao tive (nada|nenhuma) (a ver|participacao|culpa)\b",
    r"\bnao tenho (nada a ver|culpa)\b",
    r"\bnao fiz (isso|aquilo|nada)\b",
    r"\bsou inocente\b", r"\binocente\b",
    r"\bnao sou (o )?(culpado|culpada|assassino|assassina)\b",
    r"\bnao tive nada\b",
]
_FRASES_EVASIVA = [
    r"\bnao sei\b", r"\bnao lembro\b", r"\bnao me lembro\b", r"\bnao recordo\b",
    r"\bprefiro nao (falar|responder|dizer)\b", r"\bsem comentarios?\b",
    r"\bquero (meu|um) advogado\b", r"\bnao vou (responder|falar|dizer)\b",
    r"\bnao quero (falar|responder)\b", r"\bnao te interessa\b",
    r"\bnao interessa\b", r"\bnao tenho nada a dizer\b",
]


def normalizar(texto: str) -> str:
    """minusculas, sem acento, sem pontuacao, espacos colapsados."""
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c)).lower()
    texto = re.sub(r"(.)\1{2,}", r"\1", texto)           # "mateiii" -> "matei"
    texto = re.sub(r"[^a-z0-9: ]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


@dataclass
class Tema:
    id: str
    descricao: str
    gatilhos: List[str]
    pergunta_modelo: str = ""
    prioridade: int = 100

    def casa(self, normalizado: str) -> bool:
        return self.pontos(normalizado) > 0

    def pontos(self, normalizado: str) -> int:
        """Quantos gatilhos distintos aparecem na fala."""
        return sum(1 for g in self.gatilhos if self._casa_um(g, normalizado))

    @staticmethod
    def _casa_um(gatilho: str, normalizado: str) -> bool:
        g = normalizar(gatilho)
        if not g:
            return False
        # fronteira de palavra no inicio: "sai" nao pode casar dentro de "saiba".
        return re.search(rf"(?<![a-z0-9]){re.escape(g)}", normalizado) is not None


@dataclass
class Interpretacao:
    tipo: str
    confianca: str = "media"                    # alta | media | baixa
    confissao: bool = False
    negacao: bool = False
    temas: List[str] = field(default_factory=list)
    palavras: int = 0
    motivo: str = ""

    def como_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _tokens(normalizado: str) -> List[str]:
    return normalizado.split()


def _negado_antes(tokens: List[str], indice: int, janela: int = 3) -> bool:
    """Ha um negador nos `janela` tokens antes do verbo, sem 'negar' no meio?"""
    inicio = max(0, indice - janela)
    for t in tokens[inicio:indice]:
        if t in _NEGADORES:
            return True
    return False


def _dupla_negacao_de_negar(tokens: List[str]) -> bool:
    """'nao nego que matei' / 'nao vou negar' -> a negacao e de NEGAR."""
    for i, t in enumerate(tokens):
        if t in _NEGAR and _negado_antes(tokens, i, janela=3):
            return True
    return False


def _e_hipotese_ou_pergunta(bruto: str, tokens: List[str]) -> bool:
    if any(t in _HIPOTESE for t in tokens[:4]):
        return True
    # "voce acha que eu matei?" — pergunta, nao declaracao.
    if "?" in (bruto or "") and re.search(r"\b(voce|vc|tu|o senhor)\b.*\b(acha|pensa|cre)\b", normalizar(bruto)):
        return True
    return False


def classificar(texto: str, temas: Iterable[Tema] = (),
                pergunta_pendente: bool = False) -> Interpretacao:
    """Le a fala e devolve a Interpretacao (tipo + temas tocados)."""
    n = normalizar(texto)
    tokens = _tokens(n)
    palavras = len(tokens)
    tocados = [t.id for t in temas if t.casa(n)]

    if palavras == 0:
        return Interpretacao(AMBIGUA, "alta", palavras=0, motivo="fala vazia")

    if _e_hipotese_ou_pergunta(texto, tokens):
        return Interpretacao(AMBIGUA, "media", temas=tocados, palavras=palavras,
                             motivo="hipotese ou pergunta ao detetive")

    nega_o_negar = _dupla_negacao_de_negar(tokens)

    # 1) Verbo de autoria em 1a pessoa.
    for i, t in enumerate(tokens):
        if t in _VERBOS_AUTORIA:
            if _negado_antes(tokens, i) and not nega_o_negar:
                return Interpretacao(NEGACAO, "alta", negacao=True, temas=tocados,
                                     palavras=palavras, motivo=f"negou '{t}'")
            return Interpretacao(CONFISSAO, "alta", confissao=True, temas=tocados,
                                 palavras=palavras, motivo=f"disse '{t}'")

    # 2) Frases de confissao / negacao sem o verbo de morte.
    for padrao in _FRASES_NEGACAO:
        if re.search(padrao, n):
            return Interpretacao(NEGACAO, "alta", negacao=True, temas=tocados,
                                 palavras=palavras, motivo="negou autoria")
    if nega_o_negar:
        return Interpretacao(CONFISSAO, "media", confissao=True, temas=tocados,
                             palavras=palavras, motivo="'nao nego' = admite")
    for padrao in _FRASES_CONFISSAO:
        if re.search(padrao, n):
            return Interpretacao(CONFISSAO, "media", confissao=True, temas=tocados,
                                 palavras=palavras, motivo="frase de confissao")

    # 3) Evasiva: so quando a fala e curta ou nao traz conteudo alem da recusa.
    for padrao in _FRASES_EVASIVA:
        if re.search(padrao, n):
            conteudo = len(tokens) > 9 and tocados
            if not conteudo:
                return Interpretacao(EVASIVA, "alta", temas=tocados,
                                     palavras=palavras, motivo="recusa/desconhece")

    # 4) Fala curta e sem tema: nao da para tratar como resposta.
    # "em casa" responde a "onde o senhor estava?": com uma pergunta no ar, duas
    # palavras (uma delas com conteudo) ja contam como resposta.
    resposta_curta_valida = (pergunta_pendente and palavras == 2
                             and any(len(t) >= 4 for t in tokens))
    if palavras <= 2 and not tocados and not resposta_curta_valida:
        return Interpretacao(AMBIGUA, "baixa", temas=tocados, palavras=palavras,
                             motivo="fala curta sem conteudo")

    return Interpretacao(INFORMACAO, "media", temas=tocados, palavras=palavras,
                         motivo="declaracao com conteudo")


_PEDIDO = re.compile(r"\b(me diga|diga|me conte|conte|me fale|fale|explique|descreva|esclareca)\b")


def e_pergunta(texto: str) -> bool:
    """Pergunta de verdade ('?') ou pedido na forma imperativa ('me diga onde...')."""
    return "?" in (texto or "") or _PEDIDO.search(normalizar(texto)) is not None


def novidade(anteriores: Iterable[str], texto: str) -> bool:
    """
    A fala traz algo que o jogador ainda nao tinha dito sobre o tema?

    Compara palavras de conteudo e numeros/horarios: "Ja disse, no beco" nao
    traz nada, "fiquei no beco so ate 20:20" traz um horario novo. Sem isto,
    qualquer mencao ao tema destravava a re-pergunta.
    """
    def conteudo(t: str):
        return {w for w in normalizar(t).split() if len(w) >= 4 or any(c.isdigit() for c in w)}

    conhecido = set()
    for a in anteriores:
        conhecido |= conteudo(a)
    novo = conteudo(texto) - conhecido
    numerico = {w for w in novo if any(c.isdigit() for c in w)}
    return bool(numerico) or len(novo) >= 3


def temas_da_pergunta(temas: Iterable[Tema], texto: str) -> List[str]:
    """
    Sobre o que uma fala do DETETIVE pergunta: os temas de maior pontuacao.

    Diferente de `Interpretacao.temas` (qualquer tema tocado pela fala do
    jogador), aqui so entra o foco. "Quem tem acesso a porta lateral, estava
    aberta quando saiu para fumar?" toca dois temas, mas pergunta sobre o da
    porta — contar o outro acusaria repeticao onde nao ha.
    """
    n = normalizar(texto)
    pontuados = [(t.pontos(n), t.id) for t in temas]
    melhor = max((p for p, _ in pontuados), default=0)
    return [i for p, i in pontuados if p == melhor and melhor > 0]


def carregar_temas(caso: Optional[Dict[str, Any]]) -> List[Tema]:
    """Temas do caso (campo `temas_de_interrogatorio` do JSON)."""
    if not caso:
        return []
    temas = []
    for i, bruto in enumerate(caso.get("temas_de_interrogatorio", [])):
        if not bruto.get("id"):
            continue
        temas.append(Tema(
            id=bruto["id"],
            descricao=bruto.get("descricao", bruto["id"]),
            gatilhos=list(bruto.get("gatilhos", [])),
            pergunta_modelo=bruto.get("pergunta_modelo", ""),
            prioridade=int(bruto.get("prioridade", (i + 1) * 10)),
        ))
    return sorted(temas, key=lambda t: t.prioridade)
