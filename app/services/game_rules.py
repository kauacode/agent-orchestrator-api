"""
Regras DETERMINISTICAS da partida: suspeita, turnos, encerramento e timer.

Nada aqui chama modelo de IA. O LLM so propoe um nivel de suspeita e uma fala;
quem decide se a partida acabou, quanto custou demorar e se um turno conta e
este modulo — assim as regras criticas nao dependem da interpretacao livre do
modelo e sao testaveis sem provedor nenhum.

ESCALA: `nivel_suspeita` e um inteiro de 0 a 100 (ver ResponseContract e
PromptOrchestrator._limitar_faixas). "100%" == 100, "80%" == 80.

DEFINICAO DE TURNO: um turno = uma fala do jogador processada (com a resposta do
detetive) OU uma janela de resposta que expirou (timeout). Varias atualizacoes de
suspeita dentro do mesmo turno (nivel do modelo, penalidade de tempo, bonus de
confissao) sao combinadas ANTES de o turno ser fechado e contam uma vez so.

PRECEDENCIA quando mais de uma condicao de fim ocorre no mesmo turno:
  1. suspeita == 100            -> DERROTA  "suspeita_maxima"      (imediata)
  2. 3 turnos seguidos >= 80    -> DERROTA  "suspeita_sustentada"
  3. turno >= VITORIA_TURNO e suspeita <= VITORIA_SUSPEITA_MAX -> VITORIA "liberado"
  4. turno >= MAX_TURNOS        -> DERROTA se suspeita >= LIMITE_DERROTA_SUSPEITA,
                                   senao VITORIA ("limite_de_turnos")
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Regras:
    """Todos os numeros de balanceamento num lugar so."""

    # ── Encerramento por suspeita ────────────────────────────────────────────
    SUSPEITA_MAXIMA: int = 100          # chegou aqui, acabou na hora
    SUSPEITA_ALTA: int = 80             # >= conta como turno "elegivel"
    TURNOS_ALTA_PARA_FIM: int = 3       # consecutivos

    # ── Vitoria e limite de turnos ───────────────────────────────────────────
    # VITORIA_* eram constantes do GameFlowController (Unity); agora moram aqui
    # para existir uma unica definicao.
    VITORIA_TURNO: int = 8
    VITORIA_SUSPEITA_MAX: int = 30
    # DECISAO DO RESPONSAVEL DO TCC: no maximo 10 turnos (antes: 12, que era
    # uma proposta nossa). O turno 10 e absoluto: fechar_turno encerra a
    # partida nele, entao um turno 11 nunca e processado.
    MAX_TURNOS: int = 10
    LIMITE_DERROTA_SUSPEITA: int = 60

    # ── Variacao maxima de suspeita por turno (anti-salto arbitrario) ────────
    # Vem das ancoras do caso: "+25 a +40" e "-10 a -25".
    MAX_AUMENTO_POR_TURNO: int = 40
    MAX_QUEDA_POR_TURNO: int = 25

    # ── Tempo de resposta ────────────────────────────────────────────────────
    PRAZO_RESPOSTA_S: float = 60.0
    # Folga para latencia/relogio do cliente: o timeout so e aceito a partir de
    # (prazo - TOLERANCIA_TIMEOUT_S) e uma resposta so e tardia depois de
    # (prazo + TOLERANCIA_RESPOSTA_S) — ambas medidas pelo relogio do SERVIDOR.
    TOLERANCIA_TIMEOUT_S: float = 1.5
    TOLERANCIA_RESPOSTA_S: float = 2.0
    # Faixas do tempo de resposta (resposta VALIDA, dentro do prazo):
    #   baixo: < 30 s | medio: 30 s <= t < 45 s | alto: 45 s <= t <= 60 s
    # Sem lacunas nem sobreposicao: cada instante cai em exatamente uma faixa.
    TEMPO_MEDIO_A_PARTIR_S: float = 30.0
    TEMPO_ALTO_A_PARTIR_S: float = 45.0
    # Efeito do tempo na suspeita: sinal SECUNDARIO e pequeno, aplicado uma vez
    # por resposta, separado da proposta (semantica) do modelo. Baixo = 0.
    PENALIDADE_TEMPO_MEDIO: int = 1
    PENALIDADE_TEMPO_ALTO: int = 2
    # Janela fechada sem resposta valida.
    PENALIDADE_TIMEOUT: int = 15

    # ── Confissao ────────────────────────────────────────────────────────────
    # Confessar e se entregar: a suspeita vai a um PISO alto (nao mais um
    # acrescimo pequeno, que deixava "eu matei" valer menos que uma mentira).
    # A 1a confissao leva a pelo menos 85 (comeca a sequencia de 80); confirmar
    # a confissao depois leva a 100 e encerra. A confissao em si nao encerra
    # na primeira vez: o detetive ainda confronta os detalhes.
    PISO_PRIMEIRA_CONFISSAO: int = 85
    PISO_CONFISSAO_REPETIDA: int = 100


REGRAS = Regras()

RESULTADO_VITORIA = "vitoria"
RESULTADO_DERROTA = "derrota"


def limitar(valor: Any, minimo: int = 0, maximo: int = 100) -> int:
    try:
        return max(minimo, min(maximo, int(valor)))
    except (TypeError, ValueError):
        return minimo


NIVEL_BAIXO = "baixo"
NIVEL_MEDIO = "medio"
NIVEL_ALTO = "alto"


def classificar_tempo(segundos: Optional[float], regras: Regras = REGRAS) -> Optional[str]:
    """
    Faixa do tempo de resposta (None se nao houve medicao).

    Limites inclusivos por baixo: 30,0 s ja e "medio" e 45,0 s ja e "alto".
    "timeout" nao e uma faixa: e a ausencia de resposta valida, tratada a parte.
    """
    if segundos is None:
        return None
    if segundos >= regras.TEMPO_ALTO_A_PARTIR_S:
        return NIVEL_ALTO
    if segundos >= regras.TEMPO_MEDIO_A_PARTIR_S:
        return NIVEL_MEDIO
    return NIVEL_BAIXO


def penalidade_por_tempo(segundos: Optional[float], regras: Regras = REGRAS) -> int:
    """Pontos de suspeita pela demora (0 / +1 / +2), pela faixa — sem escala continua."""
    nivel = classificar_tempo(segundos, regras)
    if nivel == NIVEL_ALTO:
        return regras.PENALIDADE_TEMPO_ALTO
    if nivel == NIVEL_MEDIO:
        return regras.PENALIDADE_TEMPO_MEDIO
    return 0


def suspeita_do_turno(anterior: int, proposta_pelo_modelo: Any,
                      penalidade_tempo: int = 0, piso: int = 0,
                      regras: Regras = REGRAS) -> int:
    """
    Suspeita final do turno: a proposta do modelo, presa a variacao maxima por
    turno, mais os ajustes deterministicos do sistema (tempo e confissao). Tudo
    junto vira UM valor — e ele que entra no contador de turnos.
    """
    base = limitar(proposta_pelo_modelo)
    base = max(anterior - regras.MAX_QUEDA_POR_TURNO,
               min(anterior + regras.MAX_AUMENTO_POR_TURNO, base))
    return limitar(max(piso, base + penalidade_tempo))


@dataclass
class Janela:
    """Oportunidade de resposta do jogador (a 'interacao' com prazo)."""
    id: str
    aberta_em: float                    # epoch do servidor
    prazo_em: float                     # aberta_em + prazo
    estado: str = "aberta"              # aberta | respondida | expirada

    def segundos_restantes(self, agora: float) -> float:
        return max(0.0, self.prazo_em - agora)


@dataclass
class EstadoPartida:
    """
    Estado estruturado e persistido de uma partida. Tudo que as regras e o
    detetive precisam saber vive aqui, e nao no historico textual.
    """
    session_id: str
    case_id: Optional[str] = None
    turno: int = 0
    suspeita: int = 0
    turnos_suspeita_alta: int = 0
    encerrada: bool = False
    resultado: Optional[str] = None
    motivo_fim: Optional[str] = None
    janela: Optional[Dict[str, Any]] = None
    # interacao_id -> resposta ja entregue (idempotencia). Limitado.
    processadas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ultima_resposta: Optional[Dict[str, Any]] = None

    # Investigacao
    temas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    declaracoes: List[Dict[str, Any]] = field(default_factory=list)
    confessou: bool = False
    negou_autoria: bool = False
    tema_pergunta_atual: Optional[str] = None
    ultima_frase_impaciencia: Optional[int] = None
    ultima_pergunta: str = ""
    evidencias_reveladas: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, dados: Dict[str, Any]) -> "EstadoPartida":
        validos = {k: v for k, v in dados.items() if k in cls.__dataclass_fields__}
        return cls(**validos)


def avaliar_fim(estado: EstadoPartida,
                regras: Regras = REGRAS) -> Optional[Tuple[str, str]]:
    """
    (resultado, motivo) se a partida deve acabar AGORA, ou None.

    Pura: nao altera o estado. A precedencia esta no docstring do modulo.
    """
    if estado.suspeita >= regras.SUSPEITA_MAXIMA:
        return RESULTADO_DERROTA, "suspeita_maxima"

    if estado.turnos_suspeita_alta >= regras.TURNOS_ALTA_PARA_FIM:
        return RESULTADO_DERROTA, "suspeita_sustentada"

    if (estado.turno >= regras.VITORIA_TURNO
            and estado.suspeita <= regras.VITORIA_SUSPEITA_MAX):
        return RESULTADO_VITORIA, "liberado"

    if estado.turno >= regras.MAX_TURNOS:
        if estado.suspeita >= regras.LIMITE_DERROTA_SUSPEITA:
            return RESULTADO_DERROTA, "limite_de_turnos"
        return RESULTADO_VITORIA, "limite_de_turnos"

    return None


def fechar_turno(estado: EstadoPartida, suspeita_final: int,
                 regras: Regras = REGRAS) -> Optional[Tuple[str, str]]:
    """
    Conclui UM turno: incrementa o contador, grava a suspeita, atualiza a
    sequencia de turnos >= 80 e avalia o fim. Chamar exatamente uma vez por
    turno — a idempotencia fica a cargo de quem chama (ver PartidaService).

    A regra de 100% e verificada junto, mas como `avaliar_fim` olha o estado ja
    atualizado, ela vale em qualquer turno (inclusive o primeiro).
    """
    if estado.encerrada:
        return estado.resultado, estado.motivo_fim  # type: ignore[return-value]

    estado.turno += 1
    estado.suspeita = limitar(suspeita_final)

    if estado.suspeita >= regras.SUSPEITA_ALTA:
        estado.turnos_suspeita_alta += 1
    else:
        estado.turnos_suspeita_alta = 0

    fim = avaliar_fim(estado, regras)
    if fim is not None:
        estado.encerrada = True
        estado.resultado, estado.motivo_fim = fim
    return fim
