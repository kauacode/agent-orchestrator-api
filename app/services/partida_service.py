"""
Estado e regras da partida, sem LLM e sem rede.

Junta o que antes estava espalhado: o contador de turnos (um dict em memoria
que se perdia a cada reinicio do backend), a regra de fim (so "3 turnos >= 90")
e nenhuma nocao de tempo. O estado agora e um arquivo JSON por sessao
(`data/<sessao>_estado.json`) escrito de forma atomica, entao sobrevive a
reinicio e a sessao pode ser retomada.

Concorrencia: quem chama (PromptOrchestrator) serializa por sessao com um
asyncio.Lock — resposta e timeout da MESMA sessao nunca rodam ao mesmo tempo, e
quem pega o lock primeiro decide. Este modulo em si e sincrono e sem await.
"""
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.services import game_rules as gr
from app.services.interpretador import (
    AMBIGUA, CONFISSAO, EVASIVA, INFORMACAO, NEGACAO, Interpretacao, Tema,
    e_pergunta, novidade, temas_da_pergunta,
)

logger = logging.getLogger(__name__)

MAX_PROCESSADAS = 6
MAX_DECLARACOES = 30
MAX_REPERGUNTAS = 2     # quantas vezes um tema evasivo/em aberto pode ser repetido

# Quando o tempo acaba, a fala do detetive vem de frases_detetive.FRASES_IMPACIENCIA.


class PartidaStore:
    """Um JSON por sessao, gravado com tmp + replace (nunca fica pela metade)."""

    def __init__(self, storage_dir: str = "data"):
        self.dir = Path(storage_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def caminho(self, session_id: str) -> Path:
        return self.dir / f"{session_id}_estado.json"

    def carregar(self, session_id: str,
                 case_id: Optional[str] = None) -> gr.EstadoPartida:
        caminho = self.caminho(session_id)
        if caminho.exists():
            try:
                dados = json.loads(caminho.read_text(encoding="utf-8"))
                return gr.EstadoPartida.from_dict(dados)
            except (OSError, ValueError, TypeError) as exc:
                # Estado corrompido: melhor recomecar a contagem do que travar
                # a partida — mas nunca em silencio.
                logger.error(f"Estado de {session_id} ilegivel ({exc}); recomecando.")
        return gr.EstadoPartida(session_id=session_id, case_id=case_id)

    def salvar(self, estado: gr.EstadoPartida) -> None:
        caminho = self.caminho(estado.session_id)
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix=".estado_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(estado.to_dict(), f, ensure_ascii=False)
            os.replace(tmp, caminho)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def apagar(self, session_id: str) -> bool:
        caminho = self.caminho(session_id)
        try:
            if caminho.exists():
                caminho.unlink()
                return True
        except OSError:
            pass
        return False


class JanelaInvalida(Exception):
    """Pedido de timeout fora de hora (cedo demais, janela inexistente...)."""

    def __init__(self, mensagem: str, restante: float = 0.0):
        super().__init__(mensagem)
        self.restante = restante


class PartidaService:
    def __init__(self, store: PartidaStore,
                 relogio: Callable[[], float] = time.time,
                 regras: gr.Regras = gr.REGRAS):
        self.store = store
        self.agora = relogio
        self.regras = regras

    # ─── Janela de resposta (timer de 60 s) ──────────────────────────────────

    def abrir_janela(self, estado: gr.EstadoPartida) -> Dict[str, Any]:
        """
        Abre a janela de 60 s. O backend a abre sozinho ao entregar cada fala do
        detetive (a pergunta ja esta disponivel); a rota publica so a pede para
        a PRIMEIRA pergunta da partida. Idempotente: se ja ha uma janela aberta,
        devolve a mesma com o tempo que REALMENTE resta — o prazo e do servidor.
        """
        if estado.encerrada:
            raise JanelaInvalida("A partida ja terminou.")

        agora = self.agora()
        atual = estado.janela
        if atual and atual["estado"] == "aberta" and agora < atual["prazo_em"] + self.regras.TOLERANCIA_RESPOSTA_S:
            return self._descrever_janela(atual, agora)

        janela = gr.Janela(
            id=uuid.uuid4().hex[:12],
            aberta_em=agora,
            prazo_em=agora + self.regras.PRAZO_RESPOSTA_S,
        )
        estado.janela = janela.__dict__.copy()
        return self._descrever_janela(estado.janela, agora)

    def _descrever_janela(self, janela: Dict[str, Any], agora: float) -> Dict[str, Any]:
        return {
            "interacao_id": janela["id"],
            "prazo_s": self.regras.PRAZO_RESPOSTA_S,
            "restante_s": round(max(0.0, janela["prazo_em"] - agora), 2),
            "estado": janela["estado"],
        }

    def janela_vencida(self, estado: gr.EstadoPartida, interacao_id: Optional[str],
                       chegada: float) -> bool:
        """A resposta que CHEGOU em `chegada` ja passou do prazo (mais a folga)?"""
        j = estado.janela
        if not interacao_id or not j or j["id"] != interacao_id or j["estado"] != "aberta":
            return False
        return chegada > j["prazo_em"] + self.regras.TOLERANCIA_RESPOSTA_S

    def tempo_de_resposta(self, estado: gr.EstadoPartida, interacao_id: Optional[str],
                          chegada: float) -> Optional[float]:
        """
        Segundos entre abrir a janela e a CHEGADA da requisicao. Medido na
        chegada, antes da fila do lock e da IA: o tempo de processamento do
        backend nao e culpa do jogador. Sem arredondar: a classificacao em
        faixas (44,996 s e "medio", nao "alto") depende do valor exato.
        """
        j = estado.janela
        if not interacao_id or not j or j["id"] != interacao_id:
            return None
        return min(max(0.0, chegada - j["aberta_em"]), self.regras.PRAZO_RESPOSTA_S)

    def validar_pedido_de_timeout(self, estado: gr.EstadoPartida,
                                  interacao_id: str) -> None:
        j = estado.janela
        if not j or j["id"] != interacao_id:
            raise JanelaInvalida("Interacao desconhecida.")
        restante = j["prazo_em"] - self.agora()
        if restante > self.regras.TOLERANCIA_TIMEOUT_S:
            raise JanelaInvalida("O prazo ainda nao terminou.", restante)

    # ─── Fechamento de turno ─────────────────────────────────────────────────

    def montar_resposta(self, estado: gr.EstadoPartida, base: Dict[str, Any],
                        *, penalidade_tempo: int = 0, tempo_resposta_s: Optional[float] = None,
                        timeout: bool = False, interpretacao: Optional[Interpretacao] = None,
                        piso_suspeita: int = 0, penalidade_timeout: int = 0,
                        frase_impaciencia: str = "") -> Dict[str, Any]:
        """
        Fecha UM turno: combina suspeita do modelo + ajustes do sistema, avanca
        o contador, avalia o fim e devolve o contrato final.
        """
        status = base.setdefault("status_investigacao", {})
        proposta = status.get("nivel_suspeita", estado.suspeita)
        final = gr.suspeita_do_turno(
            estado.suspeita, proposta,
            penalidade_tempo=penalidade_tempo + penalidade_timeout,
            piso=piso_suspeita, regras=self.regras,
        )
        fim = gr.fechar_turno(estado, final, self.regras)

        status["nivel_suspeita"] = estado.suspeita
        status["congelar_input"] = estado.suspeita > self.regras.SUSPEITA_ALTA
        status["fim_de_jogo"] = estado.encerrada
        status["resultado"] = estado.resultado or ""
        status["motivo_fim"] = estado.motivo_fim or ""
        status["turnos_suspeita_alta"] = estado.turnos_suspeita_alta
        status["turnos_restantes"] = max(0, self.regras.MAX_TURNOS - estado.turno)

        base["id_turno"] = estado.turno
        base["timeout"] = timeout
        base["penalidade_tempo"] = penalidade_tempo        # efeito da faixa (0/+1/+2)
        base["penalidade_timeout"] = penalidade_timeout    # so quando o prazo estourou
        base["tempo_resposta_s"] = None if tempo_resposta_s is None else round(tempo_resposta_s, 2)
        base["nivel_tempo_resposta"] = "timeout" if timeout else (gr.classificar_tempo(tempo_resposta_s, self.regras) or "")
        base["frase_impaciencia"] = frase_impaciencia
        base["interpretacao"] = interpretacao.tipo if interpretacao else ""
        base["ignorado"] = False
        base.setdefault("interacao_id", "")

        # A proxima pergunta ja esta disponivel: o prazo da proxima resposta
        # comeca AGORA (apos o processamento da IA), no relogio do servidor.
        base["proxima_interacao_id"] = ""
        base["proxima_restante_s"] = 0.0
        if not estado.encerrada:
            janela = self.abrir_janela(estado)
            base["proxima_interacao_id"] = janela["interacao_id"]
            base["proxima_restante_s"] = janela["restante_s"]
        return base

    def resposta_de_timeout(self, estado: gr.EstadoPartida, texto: str,
                            frase: str = "") -> Dict[str, Any]:
        """Aplica o timeout UMA vez e devolve o contrato (sem chamar a IA)."""
        j = estado.janela
        interacao_id = j["id"] if j else ""
        if j:
            j["estado"] = "expirada"
        base = {
            "id_turno": estado.turno,
            "texto_detetive": texto,
            "status_investigacao": {"nivel_suspeita": estado.suspeita, "congelar_input": False,
                                    "detectou_mentira": False, "fim_de_jogo": False},
            "feedback_visual": {"cor_iluminacao": "#FFAA00", "bpm_musica": 110,
                                "animacao_trigger": "Thinking"},
        }
        resposta = self.montar_resposta(
            estado, base, penalidade_timeout=self.regras.PENALIDADE_TIMEOUT,
            tempo_resposta_s=self.regras.PRAZO_RESPOSTA_S, timeout=True,
            frase_impaciencia=frase,
        )
        resposta["interacao_id"] = interacao_id
        self.guardar_entrega(estado, interacao_id, resposta)
        return resposta

    def guardar_entrega(self, estado: gr.EstadoPartida, interacao_id: str,
                        resposta: Dict[str, Any]) -> None:
        """Guarda a resposta entregue: um reenvio da mesma interacao devolve esta."""
        estado.ultima_resposta = resposta
        if interacao_id:
            estado.processadas[interacao_id] = resposta
            while len(estado.processadas) > MAX_PROCESSADAS:
                estado.processadas.pop(next(iter(estado.processadas)))

    # ─── Investigacao ────────────────────────────────────────────────────────

    def registrar_fala_do_jogador(self, estado: gr.EstadoPartida,
                                  interp: Interpretacao, texto: str) -> Dict[str, Any]:
        """
        Atualiza o estado da investigacao com a fala. Devolve fatos uteis ao
        prompt: se respondeu a pergunta pendente e se mudou de versao.
        """
        pendente = estado.tema_pergunta_atual
        respondeu_pendente = False
        mudou_versao = False

        com_novidade = []
        for tema_id in interp.temas:
            t = estado.temas.setdefault(tema_id, _tema_novo())
            if interp.tipo == EVASIVA:
                t["evasivas"] += 1
            elif interp.tipo in (INFORMACAO, CONFISSAO, NEGACAO):
                # So conta como assunto "reaberto" se a fala traz algo que ele
                # ainda nao tinha dito (ver interpretador.novidade).
                if not t["textos"] or novidade(t["textos"], texto):
                    com_novidade.append(tema_id)
                _responder(t, texto)
            if tema_id == pendente and interp.tipo not in (EVASIVA, AMBIGUA):
                respondeu_pendente = True

        # Confissao/negacao respondem a qualquer pergunta sobre autoria.
        if interp.tipo in (CONFISSAO, NEGACAO) and pendente:
            respondeu_pendente = True

        # Resposta com conteudo que nao toca nenhum tema ("Eu estava em casa
        # com minha mae") responde a pergunta pendente: nao ha palavra-chave
        # para tudo que um jogador pode dizer.
        if (pendente and not respondeu_pendente and interp.tipo == INFORMACAO
                and not interp.temas and interp.palavras >= 2):
            respondeu_pendente = True
            _responder(estado.temas.setdefault(pendente, _tema_novo()), texto)

        if interp.tipo == CONFISSAO:
            mudou_versao = estado.negou_autoria
            estado.confessou = True
        elif interp.tipo == NEGACAO:
            mudou_versao = estado.confessou
            estado.negou_autoria = True

        estado.declaracoes.append({
            "turno": estado.turno + 1, "tipo": interp.tipo,
            "texto": texto.strip()[:240], "temas": interp.temas,
        })
        del estado.declaracoes[:-MAX_DECLARACOES]

        return {"respondeu_pendente": respondeu_pendente, "mudou_versao": mudou_versao,
                "temas_com_novidade": com_novidade}

    def registrar_pergunta_do_detetive(self, estado: gr.EstadoPartida, texto: str,
                                       temas: List[Tema]) -> None:
        """Marca o(s) tema(s) que a fala do detetive pergunta (so se for pergunta)."""
        estado.tema_pergunta_atual = None
        if not e_pergunta(texto):
            return
        estado.ultima_pergunta = ultima_pergunta(texto)
        for tema_id in temas_da_pergunta(temas, texto):
            estado.temas.setdefault(tema_id, _tema_novo())["perguntado"] += 1
            if estado.tema_pergunta_atual is None:
                estado.tema_pergunta_atual = tema_id

    def temas_repetidos(self, estado: gr.EstadoPartida, texto: str,
                        temas: List[Tema], fatos: Optional[Dict[str, Any]]) -> List[str]:
        """
        Temas que a nova fala do detetive PERGUNTA de novo sem justificativa:
        ja esclarecidos (e o jogador nao voltou a eles agora) ou ja repetidos
        MAX_REPERGUNTAS vezes. Por tema, nao por texto: a mesma pergunta com
        outras palavras tambem e pega.
        """
        if not e_pergunta(texto):
            return []
        # Re-perguntar e legitimo quando a fala atual trouxe algo NOVO sobre o
        # tema ou mudou de versao (contradicao); mencionar de novo nao basta.
        reabertos = set((fatos or {}).get("temas_com_novidade", []))
        if (fatos or {}).get("mudou_versao"):
            return []
        repetidos = []
        for tema_id in temas_da_pergunta(temas, texto):
            info = estado.temas.get(tema_id)
            if not info:
                continue
            if info["respondido"] and tema_id not in reabertos:
                repetidos.append(tema_id)
            elif not info["respondido"] and info["perguntado"] > MAX_REPERGUNTAS:
                repetidos.append(tema_id)
        return repetidos

    def proximo_tema(self, estado: gr.EstadoPartida, temas: List[Tema]) -> Optional[Tema]:
        """Primeiro tema (por prioridade) ainda em aberto e nao esgotado."""
        for tema in temas:
            info = estado.temas.get(tema.id)
            if info is None:
                return tema
            if not info["respondido"] and info["perguntado"] <= MAX_REPERGUNTAS:
                return tema
        return None


def ultima_pergunta(texto: str) -> str:
    """Ultima frase interrogativa da fala (ou a propria fala, se nao houver '?')."""
    import re
    achados = re.findall(r"[^.!?]*\?", texto or "")
    frase = (achados[-1] if achados else texto or "").strip()
    return frase[:220]


def _tema_novo() -> Dict[str, Any]:
    return {"perguntado": 0, "respondido": False, "evasivas": 0, "resposta": "", "textos": []}


def _responder(tema: Dict[str, Any], texto: str) -> None:
    tema["respondido"] = True
    tema["resposta"] = texto.strip()[:160]
    tema.setdefault("textos", []).append(texto.strip()[:300])
    del tema["textos"][:-5]
