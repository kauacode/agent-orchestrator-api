"""
Provedor local por inferencia direta (llama-cpp-python), sem daemon.

Diferente do OllamaProvider, o modelo roda dentro do proprio processo da API:
nao ha servico externo para subir, e o .gguf e carregado direto da pasta
models/. O foco e CPU/RAM — GPU e opcional (LLAMA_CPP_N_GPU_LAYERS).

Trocar de modelo e trocar LLAMA_CPP_MODEL_PATH. Nada aqui depende de um modelo
especifico: o template de chat vem embutido no proprio .gguf, e o formato de
saida e garantido por gramatica, nao pela obediencia do modelo.
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas import ResponseContract
from app.services.llm_providers import (
    LLMGenerationError,
    LLMProvider,
    LLMUnavailableError,
)
from app.services.response_parser import parse_llm_json

logger = logging.getLogger(__name__)

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

LLAMA_CPP_MODELO_PADRAO = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
LLAMA_CPP_N_CTX_PADRAO = 4096
LLAMA_CPP_N_GPU_LAYERS_PADRAO = 0
# 512 cobre o contrato com folga (~150-250 tokens em portugues). Teto baixo
# demais corta o JSON no meio — ver o tratamento de finish_reason == "length".
LLAMA_CPP_MAX_TOKENS_PADRAO = 512
# Mesmo valor do Modelfile do Ollama, para os dois provedores locais serem
# comparaveis com o mesmo modelo.
LLAMA_CPP_TEMPERATURA_PADRAO = 0.7

# Mesmo papel do SYSTEM do Modelfile do Ollama. Sem system, alguns templates
# (Qwen) injetam o deles ("You are Qwen...") — inofensivo, mas desperdica token.
SYSTEM_PROMPT = (
    "Você é um detetive de homicídios experiente, cínico e paciente, conduzindo "
    "um interrogatório. Responda SEMPRE com um único objeto JSON válido."
)


def _env_int(nome: str, padrao: int) -> int:
    valor = os.getenv(nome)
    if valor is None or not valor.strip():
        return padrao
    try:
        return int(valor)
    except ValueError:
        logger.warning(f"{nome}='{valor}' nao e inteiro; usando {padrao}.")
        return padrao


def _env_float(nome: str, padrao: float) -> float:
    valor = os.getenv(nome)
    if valor is None or not valor.strip():
        return padrao
    try:
        return float(valor)
    except ValueError:
        logger.warning(f"{nome}='{valor}' nao e numero; usando {padrao}.")
        return padrao


def _n_gpu_layers_do_env() -> int:
    # "auto" e aceito por legibilidade no .env; para o llama.cpp, -1 = todas as
    # camadas na GPU. Num build sem CUDA o valor e ignorado e roda em CPU.
    valor = os.getenv("LLAMA_CPP_N_GPU_LAYERS", "").strip().lower()
    if valor == "auto":
        return -1
    return _env_int("LLAMA_CPP_N_GPU_LAYERS", LLAMA_CPP_N_GPU_LAYERS_PADRAO)


def _threads_padrao() -> int:
    # os.cpu_count() conta nucleos LOGICOS. Com SMT, metade disso aproxima os
    # fisicos — que e o que rende no llama.cpp: threads alem dos nucleos
    # fisicos disputam a mesma unidade de calculo e a geracao fica MAIS lenta.
    # E a mesma heuristica do proprio llama-cpp-python; calculada aqui so para
    # aparecer no log. psutil daria o numero exato, mas seria dependencia nova.
    return max(1, (os.cpu_count() or 2) // 2)


def _resolver_caminho(caminho: str) -> Path:
    # Relativo a raiz do projeto, nao ao cwd: subir o uvicorn de outra pasta
    # nao pode fazer o modelo "sumir".
    p = Path(caminho)
    return p if p.is_absolute() else RAIZ_PROJETO / p


def _schema_sem_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expande os $ref do schema do Pydantic.

    O model_json_schema() poe StatusInvestigacao e FeedbackVisual em $defs e
    referencia por $ref. O conversor schema->gramatica do llama.cpp ja resolveu
    $ref local em algumas versoes e quebrou em outras; schema plano funciona em
    todas, e e isso que deixa trocar a versao da lib sem medo.
    """
    defs = schema.get("$defs", {})

    def resolver(no: Any) -> Any:
        if isinstance(no, dict):
            if "$ref" in no:
                return resolver(defs[no["$ref"].rsplit("/", 1)[-1]])
            return {k: resolver(v) for k, v in no.items() if k != "$defs"}
        if isinstance(no, list):
            return [resolver(item) for item in no]
        return no

    return resolver(schema)


def _schema_da_gramatica() -> Dict[str, Any]:
    """
    Schema do contrato, mais restricoes que so a gramatica impoe.

    Medido no Qwen 2.5 3B: sem o pattern, ele copiou o placeholder do prompt e
    mandou cor_iluminacao = "#HEXCODE" — valido para o Pydantic, mas o
    TryParseHtmlString do Unity rejeita. Fica aqui e nao no ResponseContract
    para nao endurecer a validacao do Gemini, que nunca precisou disso.

    Faixas de inteiro (minimum/maximum) nao entram: o conversor da lib ignora.
    """
    schema = _schema_sem_refs(ResponseContract.model_json_schema())
    visual = schema["properties"]["feedback_visual"]["properties"]
    visual["cor_iluminacao"]["pattern"] = "^#[0-9A-Fa-f]{6}$"
    return schema


# Calculado uma vez: o contrato nao muda em tempo de execucao.
SCHEMA_CONTRATO = _schema_da_gramatica()


class LlamaCppProvider(LLMProvider):
    """
    Modelo GGUF carregado em processo, via llama-cpp-python.

    Carregamento PREGUICOSO: nada e lido no construtor. A factory instancia o
    provedor na primeira requisicao, e um .gguf ausente ou a lib nao instalada
    nao podem derrubar a API — viram LLMUnavailableError (HTTP 503 explicativo)
    so quando alguem tenta interrogar.
    """

    nome = "llamacpp"

    # Mesmo motivo do OllamaProvider: modelos de 3-4B nao sustentam a proibicao
    # de revelar a verdade do caso. Isto tambem liga o prompt compacto no
    # PromptOrchestrator (menos tokens de entrada = menos espera em CPU).
    suporta_contexto_confidencial = False

    def __init__(self, model_path: Optional[str] = None,
                 n_ctx: Optional[int] = None,
                 n_gpu_layers: Optional[int] = None,
                 n_threads: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None):
        self.model_path = _resolver_caminho(
            model_path or os.getenv("LLAMA_CPP_MODEL_PATH", LLAMA_CPP_MODELO_PADRAO)
        )
        self.n_ctx = n_ctx or _env_int("LLAMA_CPP_N_CTX", LLAMA_CPP_N_CTX_PADRAO)
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else _n_gpu_layers_do_env()
        self.n_threads = n_threads or _env_int("LLAMA_CPP_N_THREADS", _threads_padrao())
        self.max_tokens = max_tokens or _env_int("LLAMA_CPP_MAX_TOKENS", LLAMA_CPP_MAX_TOKENS_PADRAO)
        self.temperature = (temperature if temperature is not None
                            else _env_float("LLAMA_CPP_TEMPERATURE", LLAMA_CPP_TEMPERATURA_PADRAO))

        self._llm = None
        # Uma instancia de Llama NAO e thread-safe (um unico KV cache). Duas
        # requisicoes simultaneas precisam ir para a fila, nao para o mesmo
        # contexto ao mesmo tempo.
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        """Nome curto para log, no mesmo papel do `model` dos outros provedores."""
        return self.model_path.name

    def _carregar(self):
        """Bloqueante (le ~2 GB do disco). Sempre chamado fora do event loop."""
        if not self.model_path.is_file():
            raise LLMUnavailableError(
                f"Modelo GGUF nao encontrado em '{self.model_path}'. Baixe o arquivo "
                f"para essa pasta ou ajuste LLAMA_CPP_MODEL_PATH no .env."
            )

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise LLMUnavailableError(
                "llama-cpp-python nao esta instalado. Rode: "
                "pip install -r requirements-llamacpp.txt"
            ) from exc

        inicio = time.perf_counter()
        try:
            llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=False,
            )
        except Exception as exc:
            # .gguf corrompido, download pela metade, arquitetura que a versao
            # instalada da lib nao conhece (modelo novo demais)...
            raise LLMUnavailableError(
                f"Falha ao carregar '{self.model_path.name}': {exc}"
            ) from exc

        logger.info(
            f"llama.cpp carregou {self.model_path.name} em {time.perf_counter() - inicio:.1f}s "
            f"(n_ctx={self.n_ctx}, n_threads={self.n_threads}, n_gpu_layers={self.n_gpu_layers})"
        )
        return llm

    @staticmethod
    def _mensagens(prompt: str) -> List[Dict[str, str]]:
        # Unico lugar que monta as mensagens: o aquecimento so adianta trabalho
        # se gerar EXATAMENTE os mesmos tokens de prefixo que a pergunta real.
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def _preencher_cache(self, prompt: str) -> None:
        """Bloqueante. Processa o prompt e gera 1 token so para encher o KV cache."""
        self._llm.create_chat_completion(messages=self._mensagens(prompt),
                                         max_tokens=1, temperature=0.0)

    async def aquecer(self, prompt: str) -> None:
        """
        Carrega o modelo e processa a parte fixa do prompt (caso, regras,
        instrucoes) enquanto o jogador ainda esta olhando a cena.

        Sem isto, a 1a pergunta pagava a carga do .gguf (3-13s, mais com disco
        frio) e o processamento de ~1200 tokens de prompt (~12s em CPU). Com o
        prefixo no KV cache, a 1a pergunta custa o mesmo que as seguintes.
        Usa o mesmo lock da inferencia: se o jogador perguntar no meio do
        aquecimento, a pergunta espera e ja encontra o cache pronto.
        """
        async with self._lock:
            if self._llm is None:
                self._llm = await asyncio.to_thread(self._carregar)
            inicio = time.perf_counter()
            await asyncio.to_thread(self._preencher_cache, prompt)
        logger.info(f"llama.cpp ({self.model}): aquecido em {time.perf_counter() - inicio:.1f}s")

    def _inferir(self, prompt: str) -> Dict[str, Any]:
        """Bloqueante (CPU). Sempre chamado fora do event loop."""
        try:
            return self._llm.create_chat_completion(
                messages=self._mensagens(prompt),
                # Vira gramatica GBNF: o modelo fica fisicamente impedido de
                # gerar qualquer coisa fora do schema — inclusive <think> de
                # modelos com raciocinio (Qwen3, DeepSeek-R1), cerca markdown
                # ou prosa antes do JSON. E o que torna o modelo trocavel.
                response_format={"type": "json_object", "schema": SCHEMA_CONTRATO},
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except ValueError as exc:
            # A lib levanta ValueError quando o prompt nao cabe no n_ctx. Isso
            # e configuracao, nao azar: tentar de novo o mesmo prompt falha
            # igual, entao vai como indisponivel (503) com a correcao no texto.
            if "context window" in str(exc).lower():
                raise LLMUnavailableError(
                    f"O prompt nao cabe em LLAMA_CPP_N_CTX={self.n_ctx}. Aumente o valor no .env."
                ) from exc
            raise LLMGenerationError(str(exc)) from exc

    async def generate(self, prompt: str) -> Dict[str, Any]:
        async with self._lock:
            if self._llm is None:
                self._llm = await asyncio.to_thread(self._carregar)

            inicio = time.perf_counter()
            # to_thread: a inferencia segura o processo por segundos. Rodando
            # direto na rota async, /health e todo o resto congelariam junto.
            resposta = await asyncio.to_thread(self._inferir, prompt)
            duracao = time.perf_counter() - inicio

        escolha = (resposta.get("choices") or [{}])[0]
        texto = (escolha.get("message") or {}).get("content", "")

        uso = resposta.get("usage") or {}
        gerados = uso.get("completion_tokens", 0)
        # Metrica para comparar modelos candidatos na mesma maquina.
        logger.info(
            f"llama.cpp ({self.model}): {duracao:.1f}s, "
            f"{uso.get('prompt_tokens', '?')} tokens de prompt, {gerados} gerados"
            + (f" ({gerados / duracao:.1f} tok/s no total)" if gerados and duracao else "")
        )

        if escolha.get("finish_reason") == "length":
            # A gramatica garante a FORMA, nao o fim: estourando max_tokens o
            # JSON sai cortado. Sem este check, o parser embrulharia o JSON
            # quebrado como fala do detetive e o jogador o veria na tela.
            raise LLMGenerationError(
                f"Resposta cortada em {self.max_tokens} tokens; aumente LLAMA_CPP_MAX_TOKENS."
            )

        if not texto:
            raise LLMGenerationError("llama.cpp devolveu resposta vazia.")

        logger.info(f"Resposta bruta (llamacpp): {texto}")
        return parse_llm_json(texto)

    async def aclose(self) -> None:
        # Libera a RAM do modelo no shutdown/reload do uvicorn. close() so
        # existe nas versoes mais novas da lib; nas antigas o GC resolve.
        if self._llm is not None:
            fechar = getattr(self._llm, "close", None)
            if callable(fechar):
                fechar()
            self._llm = None
