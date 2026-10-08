import asyncio
import logging
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import InterrogateRequest, ResponseContract, TesteChaveRequest
from app.services.case_repository import list_cases
from app.services.context_memory import ContextMemoryManager
from app.services.llm_providers import (
    LLMProvider,
    LLMUnavailableError,
    get_llm_provider,
)
from app.services.model_catalog import obter_modelo
from app.services.model_downloader import GerenciadorDownloads
from app.services.prompt_orchestrator import PromptOrchestrator
from app.services.provider_registry import ProviderRegistry

class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "time": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(data, ensure_ascii=False)

pasta_logs = Path("logs")
pasta_logs.mkdir(parents=True, exist_ok=True)

arquivo_log = pasta_logs / f'api_{datetime.now().strftime("%Y%m%d")}.log'

logger = logging.getLogger(__name__)

# Handlers no logger do pacote "app", nao no deste modulo: pendurados so no
# app.main, os logs de app.services.* (resposta bruta do modelo, tempo e tok/s
# do llama.cpp) eram descartados — nao chegavam nem ao console nem ao arquivo.
logger_app = logging.getLogger("app")
logger_app.setLevel(logging.INFO)

console = logging.StreamHandler()
arquivo = logging.FileHandler(str(arquivo_log), encoding='utf-8')

formatter = JsonFormatter()
console.setFormatter(formatter)
arquivo.setFormatter(formatter)

logger_app.addHandler(console)
logger_app.addHandler(arquivo)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cada partida e unica. No jogo distribuido (executavel) nenhuma sessao
    # sobrevive a uma nova abertura: apaga todas, o que tambem cobre quem fechou
    # o jogo no meio da partida. Em desenvolvimento os historicos ficam 30 dias
    # — sao eles que permitem analisar depois como o modelo se saiu.
    dias = 0 if getattr(sys, "frozen", False) else 30
    try:
        apagadas = memoria.cleanup_old_sessions(max_age_days=dias)
        if apagadas:
            logger.info(f"{apagadas} sessoes antigas removidas")
    except OSError as exc:
        logger.warning(f"Limpeza de sessoes antigas falhou: {exc}")
    yield
    # O provedor local mantem um httpx.AsyncClient aberto entre requisicoes;
    # sem fechar no shutdown o uvicorn reclama de conexao vazando no reload.
    await get_llm_provider().aclose()
    await provedores.aclose()
    await downloads.aclose()

app = FastAPI(
    title="Guilty API",
    description="API do jogo",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

memoria = ContextMemoryManager(storage_dir="data")
orchestrator = PromptOrchestrator(memory=memoria)
provedores = ProviderRegistry()
downloads = GerenciadorDownloads()

# controle de spam
rate_limit: Dict[str, List[float]] = {}
MAX_REQ = 30
JANELA_TEMPO = 60

@app.get("/health")
def health():
    # "servico" deixa o launcher do Unity distinguir o nosso backend de outro
    # programa qualquer que esteja usando a mesma porta.
    return {"status": "ok", "servico": "guilty-backend"}


def _sem_provedor_fixo() -> Optional[LLMProvider]:
    # Na rota de verdade o provedor vem da requisicao (ProviderRegistry). O
    # parametro existe para os testes injetarem um MockProvider direto.
    return None


@app.post("/interrogate", response_model=ResponseContract)
async def interrogate(req: InterrogateRequest,
                      provider: Optional[LLMProvider] = Depends(_sem_provedor_fixo)):
    sessao = req.session_id
    agora = time.time()

    historico = rate_limit.setdefault(sessao, [])
    historico = [t for t in historico if agora - t <= JANELA_TEMPO]
    rate_limit[sessao] = historico

    if len(historico) >= MAX_REQ:
        logger.warning(f"Sessao {sessao} bloqueada (rate limit)")
        raise HTTPException(status_code=429, detail="Muitas requisicoes")

    historico.append(agora)
    rate_limit[sessao] = historico

    if provider is None:
        # Resolve ANTES de gravar a fala: com a IA indisponivel (modelo nao
        # baixado, sem chave), a fala do jogador nao pode ficar orfa no
        # historico e reaparecer duplicada quando ele tentar de novo.
        try:
            provider = await provedores.resolver(req.provider, req.gemini_api_key)
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(f"Mensagem recebida na sessao {sessao}")
    memoria.append_turn(sessao, req.player_text, role="suspeito")

    logger.info(f"Provedor: {provider.nome}")

    try:
        res = await orchestrator.analyze(sessao, req.player_text,
                                         provider=provider, case_id=req.case_id)
    except LLMUnavailableError as exc:
        # 503 e nao 500: o problema nao e a nossa API, e o modelo que nao esta
        # no ar. O ApiClient do Unity ja trata nao-2xx e mostra a mensagem.
        logger.error(f"Provedor {provider.nome} indisponivel: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    memoria.append_turn(sessao, res.get("texto_detetive", ""), role="detetive")

    return res

@app.get("/cases")
def listar_casos():
    """
    Casos disponiveis, so com o que pode ser mostrado antes do interrogatorio.
    A secao `verdade` nunca sai daqui — vazaria o culpado.
    """
    return {"cases": list_cases()}

@app.get("/session/{session_id}")
def get_session(session_id: str):
    return memoria.get_session_info(session_id)

@app.delete("/session/{session_id}")
def finalizar_sessao(session_id: str):
    """
    O jogo chama quando a partida acaba (reiniciar, voltar ao menu, nova
    partida). Apaga o historico e o estado em memoria da sessao.
    """
    removeu = memoria.finalizar_sessao(session_id)
    orchestrator.turn_counter.pop(session_id, None)
    rate_limit.pop(session_id, None)
    if removeu:
        logger.info(f"Sessao {session_id} finalizada")
    return {"session_id": session_id, "finalizada": True}

@app.get("/session/{session_id}/history")
def get_history(session_id: str, limit: int = 10):
    content = memoria.load_session_md(session_id)
    linhas = content.splitlines()[-limit:] if content else []
    
    return {
        "session_id": session_id,
        "history": linhas
    }

# ─── Escolha de IA e modelos locais (tela Configuracoes > Detetive do Unity) ───

@app.get("/providers")
def listar_provedores():
    """IAs disponiveis e o estado de cada uma (baixada? precisa de chave?)."""
    itens = provedores.catalogo()
    for item in itens:
        modelo = obter_modelo(item["id"])
        item["download"] = downloads.status(modelo).como_dict() if modelo else None
    return {"providers": itens}


# Referencias fortes: sem elas o asyncio pode coletar a tarefa no meio.
_aquecimentos: set = set()


@app.post("/providers/{provider_id}/aquecer")
async def aquecer_provedor(provider_id: str):
    """
    O jogo chama ao entrar na partida. Responde na hora; o trabalho (carregar
    o modelo e processar a parte fixa do prompt) segue em segundo plano e a 1a
    pergunta encontra tudo pronto.
    """
    if provider_id == "gemini":
        return {"ok": True, "erro": None}   # nuvem: nada a adiantar
    try:
        provider = await provedores.resolver(provider_id)
    except LLMUnavailableError as exc:
        return {"ok": False, "erro": str(exc)}

    tarefa = asyncio.create_task(orchestrator.aquecer(provider))
    _aquecimentos.add(tarefa)

    def _fim(t: asyncio.Task):
        _aquecimentos.discard(t)
        if not t.cancelled() and t.exception() is not None:
            logger.warning(f"Aquecimento de {provider_id} falhou: {t.exception()}")

    tarefa.add_done_callback(_fim)
    return {"ok": True, "erro": None}


@app.post("/providers/gemini/testar")
async def testar_chave_gemini(req: TesteChaveRequest):
    try:
        await provedores.gemini(req.gemini_api_key).verificar_chave()
    except LLMUnavailableError as exc:
        return {"ok": False, "erro": str(exc)}
    return {"ok": True, "erro": None}


def _modelo_ou_404(modelo_id: str):
    modelo = obter_modelo(modelo_id)
    if modelo is None:
        raise HTTPException(status_code=404, detail=f"Modelo desconhecido: '{modelo_id}'.")
    return modelo


# async def de proposito nas tres rotas de download: rota `def` comum roda numa
# thread do threadpool, onde nao ha event loop — o create_task do download
# estourava com 500 (so apareceu no executavel; o teste de unidade ja roda
# dentro de um loop).
@app.post("/models/{modelo_id}/download")
async def iniciar_download(modelo_id: str):
    return downloads.iniciar(_modelo_ou_404(modelo_id)).como_dict()


@app.get("/models/{modelo_id}/download")
async def status_download(modelo_id: str):
    return downloads.status(_modelo_ou_404(modelo_id)).como_dict()


@app.delete("/models/{modelo_id}/download")
async def cancelar_download(modelo_id: str):
    return downloads.cancelar(_modelo_ou_404(modelo_id)).como_dict()


@app.delete("/models/{modelo_id}")
async def apagar_modelo(modelo_id: str):
    """Libera os ~2 GB do disco. Descarrega da RAM antes de apagar o arquivo."""
    modelo = _modelo_ou_404(modelo_id)
    await provedores.descarregar(modelo_id)
    try:
        return (await downloads.apagar(modelo)).como_dict()
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Não foi possível apagar o modelo: {exc}") from exc
