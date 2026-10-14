"""Estado de partida deixado por execucoes anteriores nao pode vazar para esta."""
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _sem_estado_de_partidas_antigas():
    for caminho in Path("data").glob("*_estado.json"):
        try:
            caminho.unlink()
        except OSError:
            pass
    yield
