from pydantic import BaseModel
from typing import Any, Optional

class InterrogateRequest(BaseModel):
    session_id: str
    player_text: str
    # Opcional de proposito: o Unity ainda nao envia. Sem ele o backend usa o
    # caso padrao, entao o cliente atual continua funcionando sem alteracao.
    # Quando a CaseSelectionScene passar o CaseInfo.caseId, e so preencher.
    case_id: Optional[str] = None
    # IA escolhida nas Configuracoes do jogo: "gemini" ou o id de um modelo do
    # catalogo (app/content/modelos.json). Ausente = padrao do .env.
    provider: Optional[str] = None
    # Chave do Gemini do PROPRIO jogador. Trafega so ate o backend local
    # (127.0.0.1) e nunca e gravada nem logada pelo servidor.
    gemini_api_key: Optional[str] = None
    # Janela de resposta aberta por POST /session/{id}/interacao (timer de
    # 60 s). Opcional: sem ele a fala e processada sem regra de tempo, entao
    # clientes antigos continuam funcionando. Tambem e a chave de idempotencia:
    # reenviar a mesma interacao devolve a mesma resposta, sem efeito duplo.
    interacao_id: Optional[str] = None

class TesteChaveRequest(BaseModel):
    gemini_api_key: str

class StatusInvestigacao(BaseModel):
    nivel_suspeita: int
    congelar_input: bool
    detectou_mentira: bool
    fim_de_jogo: bool

class FeedbackVisual(BaseModel):
    cor_iluminacao: str
    bpm_musica: int
    animacao_trigger: str

class ResponseContract(BaseModel):
    id_turno: int
    texto_detetive: str
    status_investigacao: StatusInvestigacao
    feedback_visual: FeedbackVisual



# ─── Contrato DEVOLVIDO ao jogo ──────────────────────────────────────────────
# ResponseContract acima e o que o MODELO gera (e vira a gramatica do llama.cpp
# — por isso nao ganha campos novos). Os campos abaixo sao preenchidos pelo
# SISTEMA depois da resposta do modelo; o Unity ignora o que nao conhece.

class StatusInvestigacaoJogo(StatusInvestigacao):
    resultado: str = ""              # "vitoria" | "derrota" | "" (em andamento)
    motivo_fim: str = ""             # suspeita_maxima | suspeita_sustentada | liberado | limite_de_turnos
    turnos_suspeita_alta: int = 0    # turnos seguidos com suspeita >= 80
    turnos_restantes: int = 0


class RespostaJogo(ResponseContract):
    status_investigacao: StatusInvestigacaoJogo
    interacao_id: str = ""
    timeout: bool = False            # True: janela expirou, texto_detetive e do sistema
    penalidade_tempo: int = 0        # efeito da faixa de tempo (0 / +1 / +2)
    penalidade_timeout: int = 0      # so quando o prazo estourou sem resposta
    nivel_tempo_resposta: str = ""   # baixo | medio | alto | timeout
    frase_impaciencia: str = ""      # preenchida so no timeout
    proxima_interacao_id: str = ""   # janela da PROXIMA resposta, ja aberta
    proxima_restante_s: float = 0.0
    tempo_resposta_s: Optional[float] = None
    interpretacao: str = ""          # confissao | negacao | evasiva | informacao | ambigua
    ignorado: bool = False           # partida ja encerrada: nada foi alterado


class JanelaResposta(BaseModel):
    interacao_id: str
    prazo_s: float
    restante_s: float
    estado: str
