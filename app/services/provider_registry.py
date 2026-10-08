"""
Escolha da IA por requisicao (o jogador escolhe nas Configuracoes do jogo).

O get_llm_provider() continua existindo e e o PADRAO quando a requisicao nao
diz nada — e o que o servidor de desenvolvimento usa, configurado no .env.
Este registro so entra quando o Unity manda `provider`.
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional

from app.services.llama_cpp_provider import LlamaCppProvider
from app.services.llm_providers import (
    GeminiProvider,
    LLMProvider,
    LLMUnavailableError,
    get_llm_provider,
)
from app.services.model_catalog import listar_modelos, obter_modelo

logger = logging.getLogger(__name__)

PROVEDOR_GEMINI = "gemini"


class ProviderRegistry:
    def __init__(self):
        # Um GeminiProvider por chave: o jogador pode trocar a chave sem
        # reiniciar o backend, e o SDK nao e reinstanciado a cada pergunta.
        self._gemini: Dict[str, GeminiProvider] = {}
        # UM modelo local carregado por vez. Cada um ocupa ~2-3 GB de RAM; numa
        # maquina de 8 GB, dois juntos com o jogo aberto nao cabem.
        self._local: Optional[LlamaCppProvider] = None
        self._local_id: Optional[str] = None
        self._lock = asyncio.Lock()

    def gemini(self, chave: Optional[str]) -> GeminiProvider:
        # A chave do jogador vence; a do .env so existe em desenvolvimento.
        chave = (chave or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        if not chave:
            raise LLMUnavailableError(
                "Para usar o Gemini, cole sua chave em Configurações > Detetive (IA)."
            )
        if chave not in self._gemini:
            self._gemini[chave] = GeminiProvider(api_key=chave)
        return self._gemini[chave]

    async def resolver(self, provider_id: Optional[str],
                       gemini_api_key: Optional[str] = None) -> LLMProvider:
        if not provider_id:
            return get_llm_provider()

        if provider_id == PROVEDOR_GEMINI:
            return self.gemini(gemini_api_key)

        modelo = obter_modelo(provider_id)
        if modelo is None:
            raise LLMUnavailableError(f"IA desconhecida: '{provider_id}'.")
        if not modelo.baixado:
            raise LLMUnavailableError(
                f"O modelo {modelo.nome} ainda não foi baixado. "
                f"Baixe em Configurações > Detetive (IA)."
            )

        async with self._lock:
            if self._local_id != modelo.id:
                await self._descarregar_local()
                self._local = LlamaCppProvider(model_path=str(modelo.caminho))
                self._local_id = modelo.id
                logger.info(f"Modelo local ativo: {modelo.id}")
            return self._local

    async def descarregar(self, modelo_id: str) -> None:
        """Solta o .gguf antes de apagar: no Windows, arquivo mapeado não sai do disco."""
        async with self._lock:
            if self._local_id == modelo_id:
                await self._descarregar_local()

    async def _descarregar_local(self) -> None:
        if self._local is not None:
            await self._local.aclose()
        self._local, self._local_id = None, None

    def chave_gemini_no_servidor(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    def catalogo(self) -> List[dict]:
        provedores = [{
            "id": PROVEDOR_GEMINI,
            "nome": "Gemini 2.5 Flash (nuvem)",
            "tipo": "nuvem",
            "descricao": "Mais rápido e esperto. Precisa de internet e de uma chave "
                         "gratuita do Google AI Studio.",
            "requer_chave": True,
            "chave_no_servidor": self.chave_gemini_no_servidor(),
            "baixado": True,
            "tamanho_bytes": 0,
            "ram_minima_gb": 0,
        }]
        for modelo in listar_modelos():
            provedores.append({
                "id": modelo.id,
                "nome": modelo.nome,
                "tipo": "local",
                "descricao": modelo.descricao,
                "requer_chave": False,
                "chave_no_servidor": False,
                "baixado": modelo.baixado,
                "tamanho_bytes": modelo.tamanho_bytes,
                "ram_minima_gb": modelo.ram_minima_gb,
            })
        return provedores

    async def aclose(self) -> None:
        await self._descarregar_local()
