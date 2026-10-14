"""
Segunda rodada: limite de 10 turnos, faixas de tempo, contexto temporal da IA,
timeout com frase de impaciencia e janela aberta junto da pergunta.
Relogio falso: nenhum teste espera 60 s reais.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services import game_rules as gr
from app.services.frases_detetive import FRASES_IMPACIENCIA, escolher_frase
from tests.test_regras_e_timer import Roteiro, SEM_TETO, jogar, novo, run


# Falas bem diferentes entre si: o filtro de repeticao (>= 85% de semelhanca)
# gasta um passo do roteiro quando as falas se parecem.
FALAS_DISTINTAS = [
    "Conte-me sobre sua rotina na gráfica.", "O crachá mostra algo curioso.",
    "Quem mais trabalhava naquele turno?", "Há câmeras? Isso o incomoda?",
    "Seu patrão lhe devia dinheiro, certo?", "Prefere café ou água?",
    "Anotei tudo que o senhor disse até agora.", "A perícia volta amanhã cedo.",
    "Ninguém sai daqui sem explicar o depósito.", "Última chance de ser sincero.",
]


# ─── Limite de 10 turnos ─────────────────────────────────────────────────────

def test_o_limite_e_10():
    assert gr.REGRAS.MAX_TURNOS == 10


def _jogar_ate_o_fim(orq, suspeita, max_chamadas=15):
    prov = Roteiro([(suspeita, "Continue.")])
    resp = []
    for i in range(max_chamadas):
        r = jogar(orq, prov, f"resposta numero {i} no beco")
        resp.append(r)
        if r["status_investigacao"]["fim_de_jogo"] and r["ignorado"] is False:
            break
    return resp, prov


def test_turno_10_encerra_com_vitoria_se_a_suspeita_esta_abaixo_do_corte():
    orq, _ = novo(SEM_TETO)
    resp, _ = _jogar_ate_o_fim(orq, 45)
    final = resp[-1]
    assert final["id_turno"] == 10 and len(resp) == 10
    assert final["status_investigacao"]["motivo_fim"] == "limite_de_turnos"
    assert final["status_investigacao"]["resultado"] == "vitoria"


def test_turno_10_encerra_com_derrota_se_a_suspeita_esta_no_corte_ou_acima():
    orq, _ = novo(SEM_TETO)
    resp, _ = _jogar_ate_o_fim(orq, 70)       # 70 < 80: nao cai nas regras de sequencia
    final = resp[-1]
    assert final["id_turno"] == 10
    assert final["status_investigacao"]["resultado"] == "derrota"
    assert final["status_investigacao"]["motivo_fim"] == "limite_de_turnos"


def test_nunca_existe_turno_11():
    orq, _ = novo(SEM_TETO)
    resp, prov = _jogar_ate_o_fim(orq, 45)
    chamadas = len(prov.prompts)
    for _ in range(3):
        r = jogar(orq, prov, "mais uma tentativa")
        assert r["ignorado"] is True and r["id_turno"] == 10
    assert len(prov.prompts) == chamadas
    assert orq.partida.store.carregar("s").turno == 10
    assert resp[-1]["proxima_interacao_id"] == ""      # sem janela depois do fim


def test_precedencia_no_turno_10_suspeita_vence_o_limite():
    # 9 turnos a 45, depois 100 no turno 10: derrota por suspeita maxima, nao por limite.
    orq, _ = novo(SEM_TETO)
    prov = Roteiro([(45, t) for t in FALAS_DISTINTAS[:9]] + [(100, "fim")])
    for i in range(10):
        r = jogar(orq, prov, f"fala {i} sobre o beco")
    assert r["id_turno"] == 10
    assert r["status_investigacao"]["motivo_fim"] == "suspeita_maxima"


def test_precedencia_no_turno_10_sequencia_de_80_vence_o_limite():
    orq, _ = novo(SEM_TETO)
    prov = Roteiro([(45, t) for t in FALAS_DISTINTAS[:7]] + [(85, "b"), (85, "c"), (85, "d")])
    for i in range(10):
        r = jogar(orq, prov, f"fala {i} sobre o beco")
    assert r["status_investigacao"]["motivo_fim"] == "suspeita_sustentada"


def test_timeout_no_turno_10_encerra_pelo_limite_e_nao_abre_turno_11():
    orq, rel = novo(SEM_TETO)
    prov = Roteiro([(45, "a")] * 9)
    for i in range(9):
        jogar(orq, prov, f"fala {i} sobre o beco")
    j = run(orq.abrir_janela("s"))
    rel.avanca(61)
    out = run(orq.expirar_janela("s", j["interacao_id"]))
    r = out["resposta"]
    assert out["aplicado"] is True and r["timeout"] is True
    assert r["id_turno"] == 10
    assert r["status_investigacao"]["fim_de_jogo"] is True
    assert r["status_investigacao"]["motivo_fim"] == "limite_de_turnos"
    assert r["proxima_interacao_id"] == ""
    again = run(orq.expirar_janela("s", j["interacao_id"]))
    assert again["aplicado"] is False
    assert orq.partida.store.carregar("s").turno == 10


# ─── Faixas de tempo ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("segundos,nivel,efeito", [
    (10, "baixo", 0), (29, "baixo", 0), (29.999, "baixo", 0),
    (30, "medio", 1), (42, "medio", 1), (44, "medio", 1), (44.999, "medio", 1),
    (45, "alto", 2), (58, "alto", 2), (60, "alto", 2),
])
def test_faixas_de_tempo_sem_lacuna_nem_sobreposicao(segundos, nivel, efeito):
    assert gr.classificar_tempo(segundos) == nivel
    assert gr.penalidade_por_tempo(segundos) == efeito


@pytest.mark.parametrize("segundos,nivel,efeito", [
    (12, "baixo", 0), (29, "baixo", 0), (30, "medio", 1),
    (44, "medio", 1), (45, "alto", 2), (59.5, "alto", 2),
])
def test_resposta_fim_a_fim_classifica_e_aplica_o_efeito_uma_vez(segundos, nivel, efeito):
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(segundos)
    prov = Roteiro([(30, "Hum.")])
    r = jogar(orq, prov, "Estava no beco fumando.", interacao_id=j["interacao_id"])

    assert r["nivel_tempo_resposta"] == nivel
    assert r["penalidade_tempo"] == efeito
    assert r["penalidade_timeout"] == 0 and r["timeout"] is False
    assert r["frase_impaciencia"] == ""
    assert r["status_investigacao"]["nivel_suspeita"] == 30 + efeito
    # a IA recebeu o contexto temporal, classificado pelo backend
    assert f"nivel_tempo_resposta: {nivel}" in prov.prompts[0]
    assert "status_tempo_resposta: dentro_do_prazo" in prov.prompts[0]


def test_prompt_explica_que_tempo_nao_e_prova_de_culpa():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(50)
    prov = Roteiro([(30, "Hum.")])
    jogar(orq, prov, "Estava no beco fumando.", interacao_id=j["interacao_id"])
    p = prov.prompts[0]
    assert "exclusivamente quanto tempo o jogador levou" in p
    assert "não representa, por si só, culpa, mentira, confissão, nervosismo ou contradição" in p
    assert "sinal contextual secundário e fraco" in p
    assert "NÃO o inclua no nível_suspeita" in p


def test_sem_medicao_nao_ha_bloco_de_tempo_nem_efeito():
    orq, _ = novo()
    prov = Roteiro([(30, "Hum.")])
    r = jogar(orq, prov, "Estava no beco fumando.")       # sem interacao_id
    assert "TEMPO DE RESPOSTA" not in prov.prompts[0]
    assert r["penalidade_tempo"] == 0 and r["nivel_tempo_resposta"] == ""


def test_tempo_independe_do_conteudo_da_resposta():
    """Texto curto ou longo, mesmo tempo => mesma faixa."""
    for texto in ("Sim.", "Eu estava no beco fumando, sozinho, das 20:05 às 20:35, sem ver ninguém."):
        orq, rel = novo()
        j = run(orq.abrir_janela("s"))
        rel.avanca(31)
        r = jogar(orq, Roteiro([(30, "Hum.")]), texto, interacao_id=j["interacao_id"])
        assert r["nivel_tempo_resposta"] == "medio"


def test_ia_nao_acumula_com_o_efeito_de_tempo():
    # A IA propoe 30; o sistema soma UMA vez o efeito da faixa (+2), nunca +5 ou duplicado.
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(58)
    r = jogar(orq, Roteiro([(30, "Hum.")]), "Estava no beco.", interacao_id=j["interacao_id"])
    assert r["status_investigacao"]["nivel_suspeita"] == 32


# ─── Janela abre com a pergunta ──────────────────────────────────────────────

def test_cada_resposta_ja_vem_com_a_proxima_janela_aberta():
    orq, rel = novo()
    prov = Roteiro([(20, "a"), (20, "b")])
    j = run(orq.abrir_janela("s"))
    r1 = jogar(orq, prov, "Estava no beco fumando.", interacao_id=j["interacao_id"])
    assert r1["proxima_interacao_id"] and r1["proxima_interacao_id"] != j["interacao_id"]
    assert r1["proxima_restante_s"] == pytest.approx(60, abs=0.01)

    # o tempo da IA nao conta: a janela abre DEPOIS do processamento
    rel.avanca(20)
    r2 = jogar(orq, prov, "Discuti com ele por horas extras.", interacao_id=r1["proxima_interacao_id"])
    assert r2["tempo_resposta_s"] == pytest.approx(20)
    assert r2["nivel_tempo_resposta"] == "baixo"


def test_abrir_janela_e_idempotente_e_nao_reinicia_o_prazo():
    orq, rel = novo()
    prov = Roteiro([(20, "a")])
    r1 = jogar(orq, prov, "Estava no beco fumando.")
    rel.avanca(25)
    j = run(orq.abrir_janela("s"))
    assert j["interacao_id"] == r1["proxima_interacao_id"]
    assert j["restante_s"] == pytest.approx(35)


# ─── Timeout ─────────────────────────────────────────────────────────────────

def test_timeout_usa_frase_de_impaciencia_e_penalidade_propria():
    orq, rel = novo()
    prov = Roteiro([])
    j = run(orq.abrir_janela("s"))
    rel.avanca(60)
    r = run(orq.expirar_janela("s", j["interacao_id"]))["resposta"]
    assert r["timeout"] is True and r["nivel_tempo_resposta"] == "timeout"
    assert r["frase_impaciencia"] in FRASES_IMPACIENCIA
    assert r["texto_detetive"].startswith(r["frase_impaciencia"])
    assert r["penalidade_timeout"] == gr.REGRAS.PENALIDADE_TIMEOUT == 15
    assert r["penalidade_tempo"] == 0                         # nao soma o efeito de faixa
    assert r["status_investigacao"]["nivel_suspeita"] == 15
    assert r["proxima_interacao_id"]                          # a investigacao segue
    assert prov.prompts == []
    assert "(SUSPEITO)" not in orq.memory.load_session_md("s")


def test_ha_10_frases_e_timeouts_seguidos_nao_repetem():
    assert len(FRASES_IMPACIENCIA) == 10
    assert len(set(FRASES_IMPACIENCIA)) == 10
    ultimo = None
    for _ in range(200):
        i, _ = escolher_frase(ultimo)
        assert i != ultimo
        ultimo = i


def test_timeouts_consecutivos_no_fluxo_real_usam_frases_diferentes():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    frases = []
    for _ in range(4):
        rel.avanca(61)
        r = run(orq.expirar_janela("s", j["interacao_id"]))["resposta"]
        frases.append(r["frase_impaciencia"])
        j = {"interacao_id": r["proxima_interacao_id"]}
    assert all(a != b for a, b in zip(frases, frases[1:]))


def test_resposta_enviada_depois_do_timeout_nao_e_processada_nem_vira_novo_turno():
    orq, rel = novo()
    prov = Roteiro([(10, "nao deveria ser chamada")])
    j = run(orq.abrir_janela("s"))
    rel.avanca(60)
    run(orq.expirar_janela("s", j["interacao_id"]))
    r = jogar(orq, prov, "Cheguei tarde.", interacao_id=j["interacao_id"])
    assert r["timeout"] is True and r["id_turno"] == 1
    assert prov.prompts == []
    e = orq.partida.store.carregar("s")
    assert e.turno == 1 and e.suspeita == 15


def test_resposta_dentro_do_prazo_nunca_recebe_penalidade_nem_frase_de_timeout():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(59.5)
    r = jogar(orq, Roteiro([(30, "Hum.")]), "Estava no beco.", interacao_id=j["interacao_id"])
    assert r["timeout"] is False and r["penalidade_timeout"] == 0
    assert r["frase_impaciencia"] == ""


def test_cliente_que_some_tem_o_timeout_aplicado_ao_reabrir():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(100)                                   # ninguem avisou o vencimento
    nova = run(orq.abrir_janela("s"))
    e = orq.partida.store.carregar("s")
    assert e.turno == 1 and e.suspeita == 15          # penalidade aplicada UMA vez
    assert nova["interacao_id"] != j["interacao_id"]


def test_falha_da_ia_apos_envio_no_prazo_nao_penaliza_e_devolve_o_prazo():
    from tests.test_regras_e_timer import Quebrado
    from app.services.llm_providers import LLMUnavailableError
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(55)
    with pytest.raises(LLMUnavailableError):
        jogar(orq, Quebrado(), "Estava no beco.", interacao_id=j["interacao_id"])
    e = orq.partida.store.carregar("s")
    assert e.turno == 0 and e.suspeita == 0
    assert e.janela["prazo_em"] == pytest.approx(rel() + 60)


def test_ia_lenta_depois_de_envio_no_prazo_nao_vira_timeout():
    orq, rel = novo()

    class Lento(Roteiro):
        async def generate(self, prompt):
            rel.avanca(120)            # a IA leva 2 minutos
            return await super().generate(prompt)

    j = run(orq.abrir_janela("s"))
    rel.avanca(10)

    async def corrida():
        a = asyncio.create_task(orq.analyze("s", "Estava no beco.", provider=Lento([(30, "Hum.")]),
                                            interacao_id=j["interacao_id"]))
        await asyncio.sleep(0)
        # o cliente acha que 60 s passaram e pede o timeout enquanto a IA ainda trabalha
        b = asyncio.create_task(orq.expirar_janela("s", j["interacao_id"]))
        return await asyncio.gather(a, b, return_exceptions=True)

    resposta, expirar = run(corrida())
    assert resposta["timeout"] is False and resposta["nivel_tempo_resposta"] == "baixo"
    # o pedido de expirar ou foi recusado (cedo) ou virou no-op: nunca um 2o efeito
    e = orq.partida.store.carregar("s")
    assert e.turno == 1
    assert e.suspeita == 30
