"""
Escolha de IA por requisicao, catalogo de modelos e download — tudo sem rede
e sem .gguf de verdade (catalogo trocado por um modelo de poucos bytes).
"""
import asyncio
import hashlib
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi import HTTPException

from app import main
from app.schemas import InterrogateRequest
from app.services import model_catalog
from app.services.llama_cpp_provider import LlamaCppProvider
from app.services.llm_providers import GeminiProvider, LLMUnavailableError
from app.services.model_catalog import ModeloLocal
from app.services.model_downloader import GerenciadorDownloads
from app.services.provider_registry import ProviderRegistry

CONTEUDO = b"gguf de mentira, so para o teste" * 64
MODELO_FALSO = dict(
    id="falso-1b", nome="Falso 1B (local)", descricao="teste",
    repo="teste/falso", arquivo="falso.gguf",
    tamanho_bytes=len(CONTEUDO), sha256=hashlib.sha256(CONTEUDO).hexdigest(),
    ram_minima_gb=1,
)


@contextmanager
def _catalogo_falso(**sobrescrever):
    """Pasta de modelos temporaria e catalogo com um modelo de poucos bytes."""
    pasta = tempfile.mkdtemp(prefix="guilty_modelos_")
    modelo = ModeloLocal(**{**MODELO_FALSO, **sobrescrever})
    original = model_catalog._carregar
    antigo_env = os.environ.get("GUILTY_MODELS_DIR")

    model_catalog._carregar = lambda: {modelo.id: modelo}
    os.environ["GUILTY_MODELS_DIR"] = pasta
    try:
        yield modelo
    finally:
        model_catalog._carregar = original
        if antigo_env is None:
            os.environ.pop("GUILTY_MODELS_DIR", None)
        else:
            os.environ["GUILTY_MODELS_DIR"] = antigo_env


@contextmanager
def _sem_chave_no_env():
    antiga = os.environ.pop("GEMINI_API_KEY", None)
    try:
        yield
    finally:
        if antiga is not None:
            os.environ["GEMINI_API_KEY"] = antiga


def _hugging_face_falso(conteudo: bytes, pedidos: list, aceita_range=True):
    def responder(request: httpx.Request) -> httpx.Response:
        pedidos.append(request)
        intervalo = request.headers.get("Range")
        if intervalo and aceita_range:
            inicio = int(intervalo.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=conteudo[inicio:])
        return httpx.Response(200, content=conteudo)
    return httpx.MockTransport(responder)


def _esperar_download(gerenciador, modelo):
    async def rodar():
        gerenciador.iniciar(modelo)
        await gerenciador._tarefas[modelo.id]
        return gerenciador.status(modelo)
    return asyncio.run(rodar())


# ─── Resolucao do provedor ──────────────────────────────────────────────────

def test_gemini_sem_chave_pede_para_configurar():
    with _sem_chave_no_env():
        try:
            asyncio.run(ProviderRegistry().resolver("gemini", None))
        except LLMUnavailableError as exc:
            assert "Configurações" in str(exc)
        else:
            raise AssertionError("sem chave deveria ser indisponivel")


def test_gemini_usa_a_chave_do_jogador_e_reaproveita_o_cliente():
    registro = ProviderRegistry()
    p1 = asyncio.run(registro.resolver("gemini", "chave-do-jogador"))
    p2 = asyncio.run(registro.resolver("gemini", "  chave-do-jogador  "))
    assert isinstance(p1, GeminiProvider)
    assert p1 is p2
    assert p1.api_key == "chave-do-jogador"


def test_ia_desconhecida_e_indisponivel():
    try:
        asyncio.run(ProviderRegistry().resolver("ollama-qualquer", None))
    except LLMUnavailableError as exc:
        assert "desconhecida" in str(exc)
    else:
        raise AssertionError("id fora do catalogo deveria ser recusado")


def test_modelo_nao_baixado_vira_503_sem_sujar_o_historico():
    with _catalogo_falso() as modelo:
        sessao = "escolha_ia_sem_modelo"
        for caminho in (main.memoria._session_md_path(sessao), main.memoria._session_version_path(sessao)):
            caminho.unlink(missing_ok=True)

        req = InterrogateRequest(session_id=sessao, player_text="Eu estava em casa.",
                                 provider=modelo.id)
        try:
            asyncio.run(main.interrogate(req, provider=None))
        except HTTPException as exc:
            assert exc.status_code == 503
            assert "não foi baixado" in exc.detail
        else:
            raise AssertionError("modelo ausente deveria virar 503")

        # A fala nao pode ficar orfa no historico (reapareceria duplicada).
        assert main.memoria.load_session_md(sessao) == ""


def test_um_modelo_local_por_vez():
    with _catalogo_falso() as modelo:
        modelo.caminho.parent.mkdir(parents=True, exist_ok=True)
        modelo.caminho.write_bytes(CONTEUDO)
        registro = ProviderRegistry()

        provedor = asyncio.run(registro.resolver(modelo.id))
        assert isinstance(provedor, LlamaCppProvider)
        assert provedor.model_path == modelo.caminho
        assert asyncio.run(registro.resolver(modelo.id)) is provedor

        asyncio.run(registro.descarregar(modelo.id))
        assert registro._local is None


def test_aquecer_responde_na_hora_e_avisa_modelo_ausente():
    with _catalogo_falso() as modelo:
        assert asyncio.run(main.aquecer_provedor("gemini"))["ok"] is True

        sem_modelo = asyncio.run(main.aquecer_provedor(modelo.id))
        assert sem_modelo["ok"] is False
        assert "não foi baixado" in sem_modelo["erro"]


def test_catalogo_lista_gemini_e_modelos_locais():
    with _catalogo_falso() as modelo:
        itens = {p["id"]: p for p in main.listar_provedores()["providers"]}

    assert itens["gemini"]["requer_chave"] is True
    assert itens[modelo.id]["tipo"] == "local"
    assert itens[modelo.id]["baixado"] is False
    assert itens[modelo.id]["download"]["estado"] == "parado"


def test_catalogo_real_tem_o_qwen_com_hash():
    qwen = model_catalog.obter_modelo("qwen2.5-3b")
    assert qwen is not None
    assert len(qwen.sha256) == 64
    assert qwen.url.startswith("https://huggingface.co/Qwen/")


# ─── Download ───────────────────────────────────────────────────────────────

def test_download_completo_confere_hash_e_libera_o_modelo():
    with _catalogo_falso() as modelo:
        pedidos = []
        estado = _esperar_download(GerenciadorDownloads(_hugging_face_falso(CONTEUDO, pedidos)), modelo)

        assert estado.estado == "concluido", estado.erro
        assert modelo.baixado
        assert modelo.caminho.read_bytes() == CONTEUDO
        assert not modelo.caminho.with_name(modelo.arquivo + ".part").exists()


def test_download_retoma_de_onde_parou():
    with _catalogo_falso() as modelo:
        metade = len(CONTEUDO) // 2
        parcial = modelo.caminho.with_name(modelo.arquivo + ".part")
        parcial.parent.mkdir(parents=True, exist_ok=True)
        parcial.write_bytes(CONTEUDO[:metade])

        pedidos = []
        estado = _esperar_download(GerenciadorDownloads(_hugging_face_falso(CONTEUDO, pedidos)), modelo)

        assert estado.estado == "concluido", estado.erro
        assert pedidos[0].headers["Range"] == f"bytes={metade}-"
        assert modelo.caminho.read_bytes() == CONTEUDO


def test_download_recomeca_se_o_servidor_ignorar_o_range():
    with _catalogo_falso() as modelo:
        parcial = modelo.caminho.with_name(modelo.arquivo + ".part")
        parcial.parent.mkdir(parents=True, exist_ok=True)
        parcial.write_bytes(CONTEUDO[:100])

        pedidos = []
        gerenciador = GerenciadorDownloads(_hugging_face_falso(CONTEUDO, pedidos, aceita_range=False))
        estado = _esperar_download(gerenciador, modelo)

        assert estado.estado == "concluido", estado.erro
        assert modelo.caminho.read_bytes() == CONTEUDO


def test_download_corrompido_e_descartado():
    with _catalogo_falso() as modelo:
        corrompido = b"X" * len(CONTEUDO)
        estado = _esperar_download(GerenciadorDownloads(_hugging_face_falso(corrompido, [])), modelo)

        assert estado.estado == "erro"
        assert "não confere" in estado.erro
        assert not modelo.baixado
        assert not modelo.caminho.with_name(modelo.arquivo + ".part").exists()


def test_download_pelas_rotas_http():
    """
    Pela app ASGI de verdade, nao pelo gerenciador direto: com as rotas
    declaradas como `def`, o FastAPI as rodava fora do event loop e o
    create_task do download virava HTTP 500 — so visto no executavel.
    """
    with _catalogo_falso() as modelo:
        transporte_original = main.downloads._transport
        main.downloads._transport = _hugging_face_falso(CONTEUDO, [])
        try:
            async def rodar():
                cliente = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                            base_url="http://teste")
                async with cliente:
                    r = await cliente.post(f"/models/{modelo.id}/download")
                    assert r.status_code == 200, r.text
                    for _ in range(100):
                        estado = (await cliente.get(f"/models/{modelo.id}/download")).json()
                        if estado["estado"] not in ("baixando", "verificando"):
                            return estado
                        await asyncio.sleep(0.02)
            estado = asyncio.run(rodar())
        finally:
            main.downloads._transport = transporte_original
            main.downloads._estados.pop(modelo.id, None)
            main.downloads._tarefas.pop(modelo.id, None)

        assert estado["estado"] == "concluido", estado
        assert modelo.baixado


def test_apagar_remove_o_arquivo():
    with _catalogo_falso() as modelo:
        modelo.caminho.parent.mkdir(parents=True, exist_ok=True)
        modelo.caminho.write_bytes(CONTEUDO)

        estado = asyncio.run(GerenciadorDownloads().apagar(modelo))

        assert not modelo.caminho.exists()
        assert estado.estado == "parado"


# ─── Erros de conta do Gemini ───────────────────────────────────────────────

class _ErroApi(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def test_chave_invalida_do_gemini_tem_mensagem_propria():
    msg = GeminiProvider._erro_de_conta(_ErroApi(400, "API key not valid. Please pass a valid API key."))
    assert msg and "chave" in msg.lower()
    assert GeminiProvider._erro_de_conta(_ErroApi(429, "Resource exhausted")) is not None
    # Outros erros continuam recuperaveis (retry do orquestrador).
    assert GeminiProvider._erro_de_conta(_ErroApi(500, "internal")) is None
