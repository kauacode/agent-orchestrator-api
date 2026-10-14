"""As rotas HTTP do timer e o contrato novo, pela API de verdade (TestClient)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app import main
from app.services.llm_providers import MockProvider


def _cliente():
    main.app.dependency_overrides[main._sem_provedor_fixo] = lambda: MockProvider()
    return TestClient(main.app)


def test_fluxo_completo_pela_api():
    c = _cliente()
    sessao = "api_timer_fluxo"
    c.delete(f"/session/{sessao}")
    try:
        j = c.post(f"/session/{sessao}/interacao").json()
        assert j["prazo_s"] == 60 and j["restante_s"] > 55

        r = c.post("/interrogate", json={"session_id": sessao, "player_text": "Eu matei mesmo.",
                                         "interacao_id": j["interacao_id"]})
        assert r.status_code == 200
        corpo = r.json()
        # campos novos do contrato, alem dos antigos
        assert corpo["interpretacao"] == "confissao"
        assert corpo["timeout"] is False
        assert corpo["interacao_id"] == j["interacao_id"]
        assert corpo["status_investigacao"]["turnos_restantes"] == main.orchestrator.partida.regras.MAX_TURNOS - 1
        assert {"nivel_suspeita", "congelar_input", "detectou_mentira", "fim_de_jogo"} <= set(corpo["status_investigacao"])

        # reenvio da mesma interacao: mesma resposta, sem novo turno
        r2 = c.post("/interrogate", json={"session_id": sessao, "player_text": "Eu matei mesmo.",
                                          "interacao_id": j["interacao_id"]})
        assert r2.json() == corpo

        # timeout "antecipado" do cliente para uma interacao ja respondida: inofensivo
        e = c.post(f"/session/{sessao}/interacao/{j['interacao_id']}/expirar")
        assert e.status_code == 200 and e.json()["aplicado"] is False
    finally:
        c.delete(f"/session/{sessao}")


def test_timeout_antecipado_devolve_409_e_desconhecida_404():
    c = _cliente()
    sessao = "api_timer_409"
    c.delete(f"/session/{sessao}")
    try:
        j = c.post(f"/session/{sessao}/interacao").json()
        assert c.post(f"/session/{sessao}/interacao/{j['interacao_id']}/expirar").status_code == 409
        assert c.post(f"/session/{sessao}/interacao/inventada/expirar").status_code == 404
    finally:
        c.delete(f"/session/{sessao}")


def test_contrato_com_o_unity_campos_do_ApiModels_existem_no_backend():
    """
    Todo campo que o Unity declara em AnalyzeResponse/StatusInvestigacao precisa
    existir no JSON do backend — senão o JsonUtility o deixa no padrao em silencio.
    """
    import re
    from app.schemas import RespostaJogo, StatusInvestigacaoJogo

    cs = Path(r"C:\Users\Kauã\Documents\TCC\Guilty-Game-Tcc\Assets\Scripts\ApiModels.cs")
    if not cs.exists():
        import pytest
        pytest.skip("projeto Unity nao esta nesta maquina")
    codigo = cs.read_text(encoding="utf-8-sig")

    def campos(classe):
        corpo = re.search(rf"class {classe}\s*\{{(.*?)\n\}}", codigo, re.S).group(1)
        return set(re.findall(r"public\s+(?!const)[\w\[\]<>]+\s+(\w+)\s*(?:;|=)", corpo)) - {classe}

    assert campos("AnalyzeResponse") <= set(RespostaJogo.model_fields)
    assert campos("StatusInvestigacao") <= set(StatusInvestigacaoJogo.model_fields)


def test_resposta_traz_a_proxima_janela_e_o_tempo():
    c = _cliente()
    sessao = "api_timer_proxima"
    c.delete(f"/session/{sessao}")
    try:
        j = c.post(f"/session/{sessao}/interacao").json()
        r = c.post("/interrogate", json={"session_id": sessao, "player_text": "Estava no beco fumando.",
                                         "interacao_id": j["interacao_id"]}).json()
        assert r["proxima_interacao_id"] and r["proxima_interacao_id"] != j["interacao_id"]
        assert 55 < r["proxima_restante_s"] <= 60
        assert r["nivel_tempo_resposta"] == "baixo"
    finally:
        c.delete(f"/session/{sessao}")
