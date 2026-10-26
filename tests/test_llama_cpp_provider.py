"""
Provedor llama.cpp sem llama.cpp: a inferencia e trocada por um modulo falso
injetado em sys.modules, entao os testes rodam sem a lib instalada e sem baixar
nenhum .gguf.
"""
import asyncio
import json
import os
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException

from app.main import interrogate, memoria
from app.schemas import InterrogateRequest, ResponseContract
from app.services import llm_providers
from app.services.case_repository import get_case
from app.services.context_memory import ContextMemoryManager
from app.services.llama_cpp_provider import SCHEMA_CONTRATO, LlamaCppProvider
from app.services.llm_providers import LLMGenerationError, LLMUnavailableError
from app.services.prompt_orchestrator import PromptOrchestrator

PASTA_TMP = Path(tempfile.mkdtemp(prefix="guilty_llamacpp_"))
GGUF_FALSO = PASTA_TMP / "falso.gguf"
GGUF_FALSO.write_bytes(b"nao e um gguf de verdade")

RESPOSTA_VALIDA = {
    "id_turno": 1,
    "texto_detetive": "Senhor Reis, o crachá diz 20:48. Onde o senhor estava às 20:15?",
    "status_investigacao": {
        "nivel_suspeita": 45,
        "congelar_input": False,
        "detectou_mentira": False,
        "fim_de_jogo": False,
    },
    "feedback_visual": {
        "cor_iluminacao": "#AA3333",
        "bpm_musica": 110,
        "animacao_trigger": "Lean_Forward",
    },
}


class _LlamaFalso:
    """Imita a parte da API do llama_cpp.Llama que o provedor usa."""

    instancias = 0
    chamadas = []
    conteudo = json.dumps(RESPOSTA_VALIDA)
    finish_reason = "stop"
    erro = None

    def __init__(self, **kwargs):
        _LlamaFalso.instancias += 1
        self.kwargs = kwargs

    def create_chat_completion(self, **kwargs):
        _LlamaFalso.chamadas.append(kwargs)
        if _LlamaFalso.erro is not None:
            raise _LlamaFalso.erro
        return {
            "choices": [{
                "message": {"role": "assistant", "content": _LlamaFalso.conteudo},
                "finish_reason": _LlamaFalso.finish_reason,
            }],
            "usage": {"prompt_tokens": 900, "completion_tokens": 120},
        }


@contextmanager
def _llama_cpp_falso():
    _LlamaFalso.instancias = 0
    _LlamaFalso.chamadas = []
    _LlamaFalso.conteudo = json.dumps(RESPOSTA_VALIDA)
    _LlamaFalso.finish_reason = "stop"
    _LlamaFalso.erro = None

    modulo = types.ModuleType("llama_cpp")
    modulo.Llama = _LlamaFalso
    anterior = sys.modules.get("llama_cpp")
    sys.modules["llama_cpp"] = modulo
    try:
        yield _LlamaFalso
    finally:
        if anterior is None:
            sys.modules.pop("llama_cpp", None)
        else:
            sys.modules["llama_cpp"] = anterior


@contextmanager
def _env(**valores):
    antigos = {k: os.environ.get(k) for k in valores}
    os.environ.update({k: v for k, v in valores.items()})
    llm_providers.get_llm_provider.cache_clear()
    try:
        yield
    finally:
        for k, v in antigos.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        llm_providers.get_llm_provider.cache_clear()


def _sessao_limpa(session_id: str) -> str:
    ContextMemoryManager(storage_dir="data").finalizar_sessao(session_id)
    return session_id


# ─── Inicializacao e indisponibilidade ──────────────────────────────────────

def test_factory_reconhece_llamacpp_sem_carregar_o_modelo():
    """Subir a API com o .gguf ainda nao baixado nao pode quebrar nada."""
    with _env(LLM_TYPE="llamacpp", LLAMA_CPP_MODEL_PATH=str(PASTA_TMP / "ainda_nao_baixado.gguf")):
        provider = llm_providers.get_llm_provider()

        assert isinstance(provider, LlamaCppProvider)
        assert provider._llm is None
        assert provider.suporta_contexto_confidencial is False


def test_gguf_ausente_vira_503_explicativo_na_rota():
    provider = LlamaCppProvider(model_path=str(PASTA_TMP / "nao_existe.gguf"))
    req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_sem_gguf"),
                             player_text="Eu estava no beco fumando.")

    try:
        asyncio.run(interrogate(req, provider=provider))
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "nao_existe.gguf" in exc.detail
        assert "LLAMA_CPP_MODEL_PATH" in exc.detail
    else:
        raise AssertionError("gguf ausente deveria virar HTTP 503")


def test_lib_nao_instalada_vira_indisponivel():
    anterior = sys.modules.get("llama_cpp")
    # None em sys.modules faz o `import llama_cpp` levantar ImportError.
    sys.modules["llama_cpp"] = None
    try:
        provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
        try:
            asyncio.run(provider.generate("qualquer coisa"))
        except LLMUnavailableError as exc:
            assert "requirements-llamacpp.txt" in str(exc)
        else:
            raise AssertionError("sem a lib deveria levantar LLMUnavailableError")
    finally:
        if anterior is None:
            sys.modules.pop("llama_cpp", None)
        else:
            sys.modules["llama_cpp"] = anterior


def test_prompt_maior_que_o_contexto_vira_indisponivel():
    """Configuracao errada nao se resolve com retry: tem que chegar ao jogador."""
    with _llama_cpp_falso() as falso:
        falso.erro = ValueError("Requested tokens (5000) exceed context window of 4096")
        provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
        try:
            asyncio.run(provider.generate("prompt"))
        except LLMUnavailableError as exc:
            assert "LLAMA_CPP_N_CTX" in str(exc)
        else:
            raise AssertionError("estouro de contexto deveria ser LLMUnavailableError")


# ─── Inferencia (mockada) ───────────────────────────────────────────────────

def test_turno_completo_respeita_o_contrato_do_unity():
    with _llama_cpp_falso() as falso:
        provider = LlamaCppProvider(model_path=str(GGUF_FALSO), n_ctx=4096,
                                    n_gpu_layers=0, n_threads=4)
        req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_turno"),
                                 player_text="Sai para fumar no beco as 20:05.")

        resultado = asyncio.run(interrogate(req, provider=provider))

        ResponseContract.model_validate(resultado)
        assert resultado["texto_detetive"] == RESPOSTA_VALIDA["texto_detetive"]

        # Config do .env/construtor chegou ao llama.cpp.
        llm = provider._llm
        assert llm.kwargs["n_ctx"] == 4096
        assert llm.kwargs["n_gpu_layers"] == 0
        assert llm.kwargs["n_threads"] == 4

        chamada = falso.chamadas[-1]
        # Saida presa ao schema do contrato, sem $ref que o conversor possa nao entender.
        formato = chamada["response_format"]
        assert formato["type"] == "json_object"
        assert "$ref" not in json.dumps(formato["schema"])

        # A verdade do caso nunca chega a um modelo pequeno.
        prompt = chamada["messages"][-1]["content"]
        culpado = get_case()["verdade"]["culpado"].split(",")[0]
        assert culpado not in prompt
        assert "VERDADE DO CASO" not in prompt


def test_modelo_e_carregado_uma_vez_so():
    with _llama_cpp_falso() as falso:
        provider = LlamaCppProvider(model_path=str(GGUF_FALSO))

        async def dois_turnos():
            await provider.generate("primeiro")
            await provider.generate("segundo")

        asyncio.run(dois_turnos())
        assert falso.instancias == 1


def test_resposta_cortada_e_falha_recuperavel():
    """JSON cortado no max_tokens nao pode virar fala do detetive na tela."""
    with _llama_cpp_falso() as falso:
        falso.conteudo = '{"id_turno": 1, "texto_detetive": "Senhor Re'
        falso.finish_reason = "length"
        provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
        try:
            asyncio.run(provider.generate("prompt"))
        except LLMGenerationError as exc:
            assert "LLAMA_CPP_MAX_TOKENS" in str(exc)
        else:
            raise AssertionError("resposta cortada deveria ser LLMGenerationError")


def test_resposta_cortada_cai_no_fallback_do_orquestrador():
    with _llama_cpp_falso() as falso:
        falso.conteudo = '{"id_turno": 1'
        falso.finish_reason = "length"
        provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
        req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_cortado"),
                                 player_text="Nao lembro.")

        resultado = asyncio.run(interrogate(req, provider=provider))

        ResponseContract.model_validate(resultado)
        assert '{"id_turno"' not in resultado["texto_detetive"]
        # 3 tentativas antes de desistir (while tentativas <= 2).
        assert len(falso.chamadas) == 3


# ─── Contrato e prompt ──────────────────────────────────────────────────────

def test_gramatica_pede_so_o_que_o_modelo_decide():
    """
    id_turno, congelar_input e fim_de_jogo sao do SISTEMA: fora da gramatica,
    o modelo deixa de gerar tokens que o servidor sobrescreveria. O contrato
    COMPLETO que o Unity recebe e conferido no teste seguinte.
    """
    assert set(SCHEMA_CONTRATO["required"]) == {"texto_detetive", "status_investigacao", "feedback_visual"}
    props = SCHEMA_CONTRATO["properties"]
    assert set(props["status_investigacao"]["required"]) == {"nivel_suspeita", "detectou_mentira"}
    # bpm e animacao continuam: serao usados pela trilha/animacao do detetive
    assert set(props["feedback_visual"]["required"]) == {"cor_iluminacao", "bpm_musica", "animacao_trigger"}


def test_resposta_ao_unity_continua_completa_sem_os_campos_do_sistema():
    """O modelo nao gera mais os campos do sistema; o Unity continua recebendo todos."""
    with _llama_cpp_falso() as falso:
        enxuta = {
            "texto_detetive": "Prossiga com o seu relato, por favor.",
            "status_investigacao": {"nivel_suspeita": 30, "detectou_mentira": False},
            "feedback_visual": {"cor_iluminacao": "#AA3333", "bpm_musica": 100, "animacao_trigger": "Neutral"},
        }
        falso.conteudo = json.dumps(enxuta)
        provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
        req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_enxuta"), player_text="Eu estava em casa.")
        resultado = asyncio.run(interrogate(req, provider=provider))

        ResponseContract.model_validate(resultado)
        assert resultado["id_turno"] == 1
        assert {"congelar_input", "fim_de_jogo"} <= set(resultado["status_investigacao"])
        assert len(falso.chamadas) == 1   # sem retry por "contrato invalido"


def test_prompt_compacto_tem_um_exemplo_e_nao_cita_verdade():
    orq = PromptOrchestrator(memory=memoria)
    caso = get_case()
    sessao = _sessao_limpa("llamacpp_prompt")

    completo = orq.build_prompt(sessao, "Eu estava no beco.", caso)
    compacto = orq.build_prompt(sessao, "Eu estava no beco.", caso,
                                incluir_verdade=False, compacto=True)

    assert completo.count("Exemplo") == 4
    assert compacto.count("Exemplo") == 1
    assert len(compacto) < len(completo)
    # Sem a secao de verdade, as instrucoes nao podem mandar comparar com ela.
    assert "VERDADE DO CASO" not in compacto


def test_historico_vem_depois_dos_blocos_fixos():
    """Prefixo estavel entre turnos = KV cache reaproveitado no runtime local."""
    orq = PromptOrchestrator(memory=memoria)
    prompt = orq.build_prompt(_sessao_limpa("llamacpp_ordem"), "Oi.", get_case(),
                              incluir_verdade=False, compacto=True)

    assert prompt.index("INSTRUÇÕES:") < prompt.index("=== HISTÓRICO DA SESSÃO ===")
    assert prompt.index("=== HISTÓRICO DA SESSÃO ===") < prompt.index("FALA DO JOGADOR:")


def test_fala_repetida_do_detetive_gera_nova_tentativa():
    """Visto no Qwen 2.5 3B: devolver a fala anterior inteira, palavra por palavra."""
    ja_dita = "Senhor Reis, onde o senhor estava entre 20:00 e 20:35?"
    nova = "O crachá registra sua saída às 20:48. O que fez até lá?"

    class _Repetidor(llm_providers.LLMProvider):
        nome = "repetidor"
        suporta_contexto_confidencial = False

        def __init__(self):
            self.falas = [ja_dita, nova]

        async def generate(self, prompt):
            return {**RESPOSTA_VALIDA, "texto_detetive": self.falas.pop(0)}

    sessao = _sessao_limpa("llamacpp_repeticao")
    memoria.append_turn(sessao, "Eu estava em casa.", role="suspeito")
    memoria.append_turn(sessao, ja_dita, role="detetive")

    req = InterrogateRequest(session_id=sessao, player_text="Ja disse, em casa.")
    resultado = asyncio.run(interrogate(req, provider=_Repetidor()))

    assert resultado["texto_detetive"] == nova


def test_aquecimento_cacheia_o_mesmo_prefixo_da_primeira_pergunta():
    """
    O aquecimento so ajuda se o prompt dele for identico ao de um 1o turno real
    ate a fala do jogador — senao o KV cache nao e reaproveitado.
    """
    with _llama_cpp_falso() as falso:
        provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
        orq = PromptOrchestrator(memory=memoria)

        asyncio.run(orq.aquecer(provider))
        aquecimento = falso.chamadas[-1]
        assert aquecimento["max_tokens"] == 1
        assert falso.instancias == 1   # o modelo ja ficou carregado

        req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_aquecido"),
                                 player_text="Eu estava em casa.")
        asyncio.run(interrogate(req, provider=provider))
        pergunta = falso.chamadas[-1]

        assert falso.instancias == 1
        assert aquecimento["messages"][0] == pergunta["messages"][0]
        prefixo = aquecimento["messages"][1]["content"].split('FALA DO JOGADOR: "')[0]
        assert pergunta["messages"][1]["content"].startswith(prefixo)
        assert len(prefixo) > 2000   # o caso inteiro esta no trecho cacheado


def test_numeros_fora_da_faixa_sao_limitados():
    """A gramatica nao impõe 0-100; um 150 estourava a barra de suspeita da HUD."""
    with _llama_cpp_falso() as falso:
        fora = json.loads(json.dumps(RESPOSTA_VALIDA))
        fora["status_investigacao"]["nivel_suspeita"] = 150
        fora["feedback_visual"]["bpm_musica"] = 300
        falso.conteudo = json.dumps(fora)

        provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
        req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_faixa"), player_text="Oi.")
        resultado = asyncio.run(interrogate(req, provider=provider))

        # 150 -> preso em 100 pela faixa; o 1o turno ainda sobe no maximo
        # MAX_AUMENTO_POR_TURNO (40) a partir de 0.
        assert resultado["status_investigacao"]["nivel_suspeita"] == 40
        assert resultado["feedback_visual"]["bpm_musica"] == 140


def test_n_gpu_layers_auto_vira_menos_um():
    with _env(LLAMA_CPP_N_GPU_LAYERS="auto"):
        assert LlamaCppProvider(model_path=str(GGUF_FALSO)).n_gpu_layers == -1


def test_saudacao_repetida_sai_a_partir_do_segundo_turno():
    """Visto nas medicoes: "Boa noite, senhor Reis." no comeco de quase toda fala."""
    tira = PromptOrchestrator._sem_saudacao
    assert tira("Boa noite, senhor Reis. Onde o senhor esteve?") == "Onde o senhor esteve?"
    assert tira("Olá, Daniel! Por que mentiu sobre o crachá?") == "Por que mentiu sobre o crachá?"
    # so o cumprimento: melhor manter do que mandar fala vazia
    assert tira("Boa noite, senhor Reis.") == "Boa noite, senhor Reis."
    # "boa noite" no meio da frase nao e saudacao
    assert tira("O senhor disse boa noite ao vigia?") == "O senhor disse boa noite ao vigia?"


def test_fala_copiada_do_exemplo_do_prompt_e_refeita():
    """Visto na validacao: o Qwen devolveu o placeholder do modelo de JSON como fala."""
    with _llama_cpp_falso() as falso:
        respostas = [
            {"texto_detetive": "<escreva aqui a sua fala, em segunda pessoa>",
             "status_investigacao": {"nivel_suspeita": 30, "detectou_mentira": False},
             "feedback_visual": {"cor_iluminacao": "#AA3333", "bpm_musica": 100, "animacao_trigger": "N"}},
            {"texto_detetive": "Prossiga com o seu relato, por favor.",
             "status_investigacao": {"nivel_suspeita": 30, "detectou_mentira": False},
             "feedback_visual": {"cor_iluminacao": "#AA3333", "bpm_musica": 100, "animacao_trigger": "N"}},
        ]
        criar_original = falso.create_chat_completion

        def roteiro(self, **kwargs):
            falso.conteudo = json.dumps(respostas[min(len(falso.chamadas), 1)])
            return criar_original(self, **kwargs)

        falso.create_chat_completion = roteiro
        try:
            provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
            req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_placeholder"), player_text="Eu estava em casa.")
            resultado = asyncio.run(interrogate(req, provider=provider))
        finally:
            falso.create_chat_completion = criar_original

        assert resultado["texto_detetive"] == "Prossiga com o seu relato, por favor."
        assert len(falso.chamadas) == 2


def test_analise_interna_nao_vaza_na_fala():
    """Visto em jogo: a fala do Qwen terminava em 'Detectou_mentira: true, suspeita sobe +25.'"""
    limpa = PromptOrchestrator._sem_metadados
    original = ("Entendo que você estava no prédio. No entanto, seu crachá registra saída às 20:48. "
                "Isso colide com a informação que você deu. Detectou_mentira: true, suspeita sobe +25.")
    assert limpa(original) == ("Entendo que você estava no prédio. No entanto, seu crachá registra saída às 20:48. "
                               "Isso colide com a informação que você deu.")
    assert limpa("Nível de suspeita: 80.") == ""          # nada sobra: a fala sera refeita
    assert limpa("O senhor mentiu sobre o horário?") == "O senhor mentiu sobre o horário?"


def _roteiro(falso, respostas):
    """Faz o llama falso devolver uma resposta diferente a cada chamada."""
    criar_original = falso.create_chat_completion

    def roteiro(self, **kwargs):
        texto = respostas[min(len(falso.chamadas), len(respostas) - 1)]
        falso.conteudo = json.dumps(
            {"texto_detetive": texto,
             "status_investigacao": {"nivel_suspeita": 30, "detectou_mentira": False},
             "feedback_visual": {"cor_iluminacao": "#AA3333", "bpm_musica": 100, "animacao_trigger": "N"}})
        return criar_original(self, **kwargs)

    falso.create_chat_completion = roteiro
    return criar_original


def test_fala_com_ideogramas_e_refeita():
    """Visto no caso 03: 'há uma questão que preciso澄清: você mencionou...'"""
    with _llama_cpp_falso() as falso:
        original = _roteiro(falso, ["Há uma questão que preciso澄清: onde o senhor estava?",
                                    "Prossiga com o seu relato, por favor."])
        try:
            provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
            req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_ideograma"), player_text="Eu estava em casa.")
            resultado = asyncio.run(interrogate(req, provider=provider))
        finally:
            falso.create_chat_completion = original

        assert resultado["texto_detetive"] == "Prossiga com o seu relato, por favor."
        assert len(falso.chamadas) == 2


def test_ideogramas_insistentes_saem_da_fala():
    """Se o modelo insistir, a fala vai sem os ideogramas em vez do fallback generico."""
    with _llama_cpp_falso() as falso:
        original = _roteiro(falso, ["Há uma questão que preciso澄清: onde o senhor estava?"])
        try:
            provider = LlamaCppProvider(model_path=str(GGUF_FALSO))
            req = InterrogateRequest(session_id=_sessao_limpa("llamacpp_ideograma2"), player_text="Eu estava em casa.")
            resultado = asyncio.run(interrogate(req, provider=provider))
        finally:
            falso.create_chat_completion = original

        assert resultado["texto_detetive"] == "Há uma questão que preciso: onde o senhor estava?"
