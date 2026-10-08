"""
Catalogo dos modelos locais que o jogador pode baixar (app/content/modelos.json).

O catalogo e conteudo versionado; os arquivos .gguf nao. Eles moram em
MODELS_DIR, que no jogo empacotado e a pasta de dados do usuario (o Unity passa
GUILTY_MODELS_DIR ao iniciar o backend) — a pasta de instalacao do jogo pode
estar em "Arquivos de Programas", sem permissao de escrita.
"""
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

ARQUIVO_CATALOGO = Path(__file__).resolve().parent.parent / "content" / "modelos.json"
RAIZ_PROJETO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModeloLocal:
    id: str
    nome: str
    descricao: str
    repo: str
    arquivo: str
    tamanho_bytes: int
    sha256: str
    ram_minima_gb: int

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.arquivo}"

    @property
    def caminho(self) -> Path:
        return pasta_modelos() / self.arquivo

    @property
    def baixado(self) -> bool:
        # Tamanho exato, nao so existencia: um download interrompido deixa o
        # .part, mas um arquivo copiado pela metade com o nome final passaria.
        # O hash completo so e conferido no fim do download (custa ~10s em 2 GB).
        try:
            return self.caminho.stat().st_size == self.tamanho_bytes
        except OSError:
            return False


def pasta_modelos() -> Path:
    pasta = os.getenv("GUILTY_MODELS_DIR")
    return Path(pasta) if pasta else RAIZ_PROJETO / "models"


@lru_cache(maxsize=1)
def _carregar() -> Dict[str, ModeloLocal]:
    dados = json.loads(ARQUIVO_CATALOGO.read_text(encoding="utf-8"))
    return {m["id"]: ModeloLocal(**m) for m in dados["modelos"]}


def listar_modelos() -> List[ModeloLocal]:
    return list(_carregar().values())


def obter_modelo(modelo_id: str) -> Optional[ModeloLocal]:
    return _carregar().get(modelo_id)
