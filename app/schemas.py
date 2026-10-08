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
