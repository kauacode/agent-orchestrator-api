"""
Cada partida e unica: finalizar apaga tudo, e nada da partida anterior pode
vazar para a seguinte (suspeita alta herdada fazia a partida nova ja nascer
perdida).
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import main
from app.schemas import InterrogateRequest
from app.services.context_memory import ContextMemoryManager
from app.services.llm_providers import MockProvider


def _jogar(sessao, fala="Eu nunca estive la."):
    req = InterrogateRequest(session_id=sessao, player_text=fala)
    return asyncio.run(main.interrogate(req, provider=MockProvider()))


def test_finalizar_apaga_a_partida_inteira():
    sessao = "finalizar_partida"
    main.finalizar_sessao(sessao)
    _jogar(sessao)
    _jogar(sessao)

    md = main.memoria._session_md_path(sessao)
    assert md.exists() and main.memoria._session_version_path(sessao).exists()

    resposta = main.finalizar_sessao(sessao)

    assert resposta["finalizada"] is True
    assert not md.exists()
    assert not main.memoria._session_version_path(sessao).exists()
    assert not md.with_suffix(".lock").exists()
    assert sessao not in main.orchestrator.turn_counter
    assert sessao not in main.rate_limit


def test_partida_nova_nao_herda_nada_da_anterior():
    sessao = "finalizar_heranca"
    main.finalizar_sessao(sessao)
    for _ in range(3):
        _jogar(sessao)   # mock: suspeita 80 a cada mentira
    assert main.memoria.get_recent_suspicions(sessao, n=3) == [80, 80, 80]

    main.finalizar_sessao(sessao)
    resposta = _jogar(sessao, "Boa noite.")

    assert resposta["id_turno"] == 1
    assert main.memoria.get_recent_suspicions(sessao, n=3) == [20]


def test_finalizar_sessao_inexistente_nao_quebra():
    assert main.finalizar_sessao("nunca_existiu")["finalizada"] is True


def test_limpeza_total_na_abertura_do_jogo():
    """max_age_days=0 e o que o executavel usa ao abrir: nenhuma sessao sobra."""
    mem = ContextMemoryManager(storage_dir=tempfile.mkdtemp(prefix="guilty_sessoes_"))
    for sessao in ("a", "b"):
        mem.append_turn(sessao, "fala", role="suspeito")
        mem.append_suspicion(sessao, 50)

    assert mem.cleanup_old_sessions(max_age_days=0) == 2
    assert list(mem.storage_dir.iterdir()) == []
