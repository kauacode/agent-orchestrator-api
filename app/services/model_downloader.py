"""
Download dos modelos do catalogo, com progresso consultavel pelo Unity.

Roda dentro do backend (e nao no Unity) para a regra ficar num lugar so: o
mesmo processo que usa o .gguf e quem decide se ele esta integro.

- Baixa para `<arquivo>.part` e so renomeia depois de conferir tamanho e
  SHA-256. O provedor nunca ve um arquivo pela metade com o nome final.
- Retoma de onde parou (HTTP Range): cair a internet nos 1,8 GB de 2,1 GB nao
  custa recomecar do zero.
- httpx e nao huggingface_hub: o httpx ja e dependencia, e o hub acrescentaria
  dezenas de MB ao backend empacotado so para fazer um GET.
"""
import asyncio
import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import httpx

from app.services.model_catalog import ModeloLocal

logger = logging.getLogger(__name__)

PEDACO = 1 << 20  # 1 MiB


@dataclass
class EstadoDownload:
    # parado | baixando | verificando | concluido | erro | cancelado
    estado: str = "parado"
    baixados: int = 0
    total: int = 0
    erro: Optional[str] = None

    def como_dict(self) -> dict:
        return asdict(self)


def _arquivo_parcial(modelo: ModeloLocal) -> Path:
    return modelo.caminho.with_name(modelo.caminho.name + ".part")


def _hash_do_que_ja_existe(caminho: Path) -> "hashlib._Hash":
    hasher = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(16 * PEDACO), b""):
            hasher.update(bloco)
    return hasher


class GerenciadorDownloads:
    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None):
        # transport: so para os testes simularem o Hugging Face sem rede.
        self._transport = transport
        self._estados: Dict[str, EstadoDownload] = {}
        self._tarefas: Dict[str, asyncio.Task] = {}

    def status(self, modelo: ModeloLocal) -> EstadoDownload:
        estado = self._estados.get(modelo.id)
        if estado is not None:
            return estado

        # Nada em andamento nesta execucao do backend: descreve o disco.
        if modelo.baixado:
            return EstadoDownload("concluido", modelo.tamanho_bytes, modelo.tamanho_bytes)
        parcial = _arquivo_parcial(modelo)
        ja = parcial.stat().st_size if parcial.exists() else 0
        return EstadoDownload("parado", ja, modelo.tamanho_bytes)

    def iniciar(self, modelo: ModeloLocal) -> EstadoDownload:
        tarefa = self._tarefas.get(modelo.id)
        if tarefa is not None and not tarefa.done():
            return self._estados[modelo.id]

        if modelo.baixado:
            estado = EstadoDownload("concluido", modelo.tamanho_bytes, modelo.tamanho_bytes)
            self._estados[modelo.id] = estado
            return estado

        estado = EstadoDownload("baixando", 0, modelo.tamanho_bytes)
        self._estados[modelo.id] = estado
        self._tarefas[modelo.id] = asyncio.create_task(self._baixar(modelo, estado))
        return estado

    def cancelar(self, modelo: ModeloLocal) -> EstadoDownload:
        tarefa = self._tarefas.get(modelo.id)
        if tarefa is not None and not tarefa.done():
            tarefa.cancel()
        return self.status(modelo)

    async def apagar(self, modelo: ModeloLocal) -> EstadoDownload:
        tarefa = self._tarefas.pop(modelo.id, None)
        if tarefa is not None and not tarefa.done():
            tarefa.cancel()
            try:
                await tarefa
            except (asyncio.CancelledError, Exception):
                pass

        for caminho in (modelo.caminho, _arquivo_parcial(modelo)):
            caminho.unlink(missing_ok=True)

        self._estados.pop(modelo.id, None)
        return self.status(modelo)

    async def aclose(self) -> None:
        for tarefa in self._tarefas.values():
            if not tarefa.done():
                tarefa.cancel()

    async def _baixar(self, modelo: ModeloLocal, estado: EstadoDownload) -> None:
        parcial = _arquivo_parcial(modelo)
        try:
            parcial.parent.mkdir(parents=True, exist_ok=True)

            ja = parcial.stat().st_size if parcial.exists() else 0
            if ja > modelo.tamanho_bytes:
                parcial.unlink()
                ja = 0

            # Retomando: o hash precisa cobrir o que ja esta no disco.
            hasher = (await asyncio.to_thread(_hash_do_que_ja_existe, parcial)
                      if ja else hashlib.sha256())
            estado.baixados = ja

            cabecalhos = {"Range": f"bytes={ja}-"} if ja else {}
            timeout = httpx.Timeout(30.0, read=120.0)
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout,
                                         transport=self._transport) as client:
                async with client.stream("GET", modelo.url, headers=cabecalhos) as resposta:
                    if ja and resposta.status_code == 200:
                        # Servidor ignorou o Range e mandou o arquivo inteiro.
                        ja, hasher, estado.baixados = 0, hashlib.sha256(), 0
                    elif resposta.status_code not in (200, 206):
                        raise RuntimeError(f"Hugging Face respondeu HTTP {resposta.status_code}.")

                    with open(parcial, "ab" if ja else "wb") as f:
                        async for pedaco in resposta.aiter_bytes(PEDACO):
                            f.write(pedaco)
                            hasher.update(pedaco)
                            estado.baixados += len(pedaco)

            estado.estado = "verificando"
            tamanho = parcial.stat().st_size
            if tamanho != modelo.tamanho_bytes or hasher.hexdigest() != modelo.sha256:
                parcial.unlink(missing_ok=True)
                raise RuntimeError("O arquivo baixado não confere (tamanho ou SHA-256). Tente de novo.")

            parcial.replace(modelo.caminho)
            estado.estado = "concluido"
            logger.info(f"Modelo {modelo.id} baixado e conferido em {modelo.caminho}")

        except asyncio.CancelledError:
            # O .part fica: o proximo "baixar" continua dele.
            estado.estado = "cancelado"
            raise
        except httpx.HTTPError as exc:
            estado.estado, estado.erro = "erro", "Falha de conexão durante o download. Clique em baixar para continuar de onde parou."
            logger.error(f"Download de {modelo.id} falhou: {exc}")
        except Exception as exc:
            estado.estado, estado.erro = "erro", str(exc)
            logger.error(f"Download de {modelo.id} falhou: {exc}")
