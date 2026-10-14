"""
Regras deterministicas de suspeita/encerramento e o timer de 60 s.

Tudo sem rede e sem modelo: um provedor roteirizado devolve a suspeita que o
teste quer e um relogio falso controla o tempo. Cada teste usa uma pasta de
dados propria — nada vaza entre testes nem entre execucoes.
"""
import asyncio
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services import game_rules as gr
from app.services.context_memory import ContextMemoryManager
from app.services.llm_providers import LLMProvider, LLMUnavailableError
from app.services.partida_service import JanelaInvalida
from app.services.prompt_orchestrator import PromptOrchestrator


class Relogio:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def avanca(self, s):
        self.t += s


class Roteiro(LLMProvider):
    """Devolve, em ordem, (suspeita, fala). Guarda os prompts recebidos."""
    nome = "roteiro"

    def __init__(self, passos):
        self.passos = list(passos)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        suspeita, fala = self.passos.pop(0) if self.passos else (self.ultimo)
        self.ultimo = (suspeita, fala)
        return {
            "id_turno": 1, "texto_detetive": fala,
            "status_investigacao": {"nivel_suspeita": suspeita, "congelar_input": False,
                                    "detectou_mentira": False, "fim_de_jogo": False},
            "feedback_visual": {"cor_iluminacao": "#FFFFFF", "bpm_musica": 90,
                                "animacao_trigger": "Neutral"},
        }


class Quebrado(LLMProvider):
    nome = "quebrado"

    async def generate(self, prompt):
        raise LLMUnavailableError("fora do ar")


def novo(regras=gr.REGRAS):
    """Orquestrador isolado, com relogio falso e regras injetaveis."""
    mem = ContextMemoryManager(storage_dir=tempfile.mkdtemp(prefix="guilty_t_"))
    orq = PromptOrchestrator(memory=mem)
    relogio = Relogio()
    orq.partida.agora = relogio
    orq.partida.regras = regras
    return orq, relogio


def run(coro):
    return asyncio.run(coro)


def jogar(orq, provider, texto="Estava no beco fumando.", sessao="s", **kw):
    return run(orq.analyze(sessao, texto, provider=provider, **kw))


SEM_TETO = replace(gr.REGRAS, MAX_AUMENTO_POR_TURNO=100, MAX_QUEDA_POR_TURNO=100)


# ─── Regras puras ────────────────────────────────────────────────────────────

def _estado(): return gr.EstadoPartida(session_id="x")


@pytest.mark.parametrize("valor,fim", [(79, False), (80, False), (99, False), (100, True)])
def test_so_100_encerra_de_imediato_em_qualquer_turno(valor, fim):
    e = _estado()
    r = gr.fechar_turno(e, valor)
    assert e.turno == 1                       # vale ate no primeiro turno
    assert (r is not None) is fim
    assert e.encerrada is fim
    if fim:
        assert r == (gr.RESULTADO_DERROTA, "suspeita_maxima")


def test_79_nao_conta_80_conta():
    e = _estado()
    gr.fechar_turno(e, 79)
    assert e.turnos_suspeita_alta == 0
    gr.fechar_turno(e, 80)
    assert e.turnos_suspeita_alta == 1


def test_tabela_do_enunciado_85_90_82_encerra_no_terceiro():
    e = _estado()
    assert [gr.fechar_turno(e, v) for v in (85, 90)] == [None, None]
    assert [e.turnos_suspeita_alta] == [2]    # dois seguidos nao encerram
    assert gr.fechar_turno(e, 82) == (gr.RESULTADO_DERROTA, "suspeita_sustentada")
    assert e.turnos_suspeita_alta == 3 and e.turno == 3


def test_tabela_do_enunciado_reinicia_ao_cair_abaixo_de_80():
    e = _estado()
    contagens = []
    for v in (85, 78, 90, 88, 91):
        gr.fechar_turno(e, v)
        contagens.append(e.turnos_suspeita_alta)
    assert contagens == [1, 0, 1, 2, 3]
    assert e.encerrada and e.motivo_fim == "suspeita_sustentada" and e.turno == 5


def test_precedencia_100_vence_a_sequencia():
    e = _estado()
    e.turnos_suspeita_alta = 2
    assert gr.fechar_turno(e, 100) == (gr.RESULTADO_DERROTA, "suspeita_maxima")


def test_turno_apos_encerramento_nao_altera_nada():
    e = _estado()
    gr.fechar_turno(e, 100)
    antes = e.to_dict().copy()
    gr.fechar_turno(e, 10)
    assert e.to_dict() == antes


def test_vitoria_e_limite_de_turnos():
    e = _estado()
    for _ in range(7):
        assert gr.fechar_turno(e, 20) is None
    assert gr.fechar_turno(e, 20) == (gr.RESULTADO_VITORIA, "liberado")    # turno 8

    e = _estado()
    for _ in range(gr.REGRAS.MAX_TURNOS - 1):
        assert gr.fechar_turno(e, 45) is None
    assert gr.fechar_turno(e, 45) == (gr.RESULTADO_VITORIA, "limite_de_turnos")

    e = _estado()
    for _ in range(gr.REGRAS.MAX_TURNOS - 1):
        gr.fechar_turno(e, 70)
    assert gr.fechar_turno(e, 70) == (gr.RESULTADO_DERROTA, "limite_de_turnos")


def test_penalidade_por_tempo_vem_da_faixa():
    assert gr.penalidade_por_tempo(0) == 0
    assert gr.penalidade_por_tempo(29.99) == 0
    assert gr.penalidade_por_tempo(30) == gr.REGRAS.PENALIDADE_TEMPO_MEDIO
    assert gr.penalidade_por_tempo(44.99) == gr.REGRAS.PENALIDADE_TEMPO_MEDIO
    assert gr.penalidade_por_tempo(45) == gr.REGRAS.PENALIDADE_TEMPO_ALTO
    assert gr.penalidade_por_tempo(60) == gr.REGRAS.PENALIDADE_TEMPO_ALTO
    assert gr.penalidade_por_tempo(9999) == gr.REGRAS.PENALIDADE_TEMPO_ALTO


def test_variacao_por_turno_e_limitada_e_resultado_fica_em_0_100():
    assert gr.suspeita_do_turno(0, 100) == 40
    assert gr.suspeita_do_turno(80, 0) == 55
    assert gr.suspeita_do_turno(95, 100, penalidade_tempo=5) == 100
    assert gr.suspeita_do_turno(10, 10, piso=85) == 85
    assert gr.suspeita_do_turno(0, "lixo") == 0


# ─── Integracao com o orquestrador ───────────────────────────────────────────

def test_100_no_primeiro_turno_encerra_imediatamente_e_trava_a_partida():
    orq, _ = novo(SEM_TETO)
    prov = Roteiro([(100, "Chega.")])
    r = jogar(orq, prov)
    assert r["id_turno"] == 1
    assert r["status_investigacao"]["fim_de_jogo"] is True
    assert r["status_investigacao"]["resultado"] == "derrota"
    assert r["status_investigacao"]["motivo_fim"] == "suspeita_maxima"

    # Nova mensagem: nada muda, o modelo nem e chamado, o historico nao cresce.
    chamadas = len(prov.prompts)
    historico = orq.memory.load_session_md("s")
    r2 = jogar(orq, prov, "Espera, eu explico!")
    assert r2["ignorado"] is True and r2["id_turno"] == 1
    assert len(prov.prompts) == chamadas
    assert orq.memory.load_session_md("s") == historico


def test_99_nao_encerra():
    orq, _ = novo(SEM_TETO)
    r = jogar(orq, Roteiro([(99, "Hum.")]))
    assert r["status_investigacao"]["fim_de_jogo"] is False
    assert r["status_investigacao"]["nivel_suspeita"] == 99


def test_tres_turnos_seguidos_a_80_encerram_no_terceiro():
    orq, _ = novo(SEM_TETO)
    prov = Roteiro([(85, "a"), (90, "b"), (82, "c")])
    r1 = jogar(orq, prov, "Fui fumar.")
    r2 = jogar(orq, prov, "Sozinho, sim.")
    assert not r1["status_investigacao"]["fim_de_jogo"]
    assert not r2["status_investigacao"]["fim_de_jogo"]
    assert r2["status_investigacao"]["turnos_suspeita_alta"] == 2
    r3 = jogar(orq, prov, "Ninguem me viu.")
    assert r3["status_investigacao"]["fim_de_jogo"] is True
    assert r3["status_investigacao"]["motivo_fim"] == "suspeita_sustentada"
    assert r3["id_turno"] == 3


def test_turno_abaixo_de_80_reinicia_a_contagem():
    orq, _ = novo(SEM_TETO)
    prov = Roteiro([(85, "a"), (78, "b"), (90, "c"), (88, "d")])
    resp = [jogar(orq, prov, f"fala numero {i} no beco") for i in range(4)]
    assert [r["status_investigacao"]["turnos_suspeita_alta"] for r in resp] == [1, 0, 1, 2]
    assert not resp[-1]["status_investigacao"]["fim_de_jogo"]


def test_reenvio_da_mesma_interacao_nao_duplica_turno_nem_contagem():
    orq, _ = novo(SEM_TETO)
    prov = Roteiro([(85, "a"), (85, "b")])
    j = run(orq.abrir_janela("s"))
    r1 = jogar(orq, prov, "Fui fumar no beco.", interacao_id=j["interacao_id"])
    r2 = jogar(orq, prov, "Fui fumar no beco.", interacao_id=j["interacao_id"])
    assert r1 == r2
    assert len(prov.prompts) == 1
    assert orq.partida.store.carregar("s").turno == 1
    assert orq.partida.store.carregar("s").turnos_suspeita_alta == 1
    assert orq.memory.load_session_md("s").count("Fui fumar no beco.") == 1


def test_estado_sobrevive_a_reinicio_do_backend():
    orq, _ = novo(SEM_TETO)
    jogar(orq, Roteiro([(85, "a")]))
    outro = PromptOrchestrator(memory=orq.memory)   # "reiniciou": memoria nova
    e = outro.partida.store.carregar("s")
    assert (e.turno, e.suspeita, e.turnos_suspeita_alta) == (1, 85, 1)


def test_falha_do_modelo_nao_altera_a_partida_e_reabre_a_janela():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(50)
    with pytest.raises(LLMUnavailableError):
        jogar(orq, Quebrado(), "Estava no beco.", interacao_id=j["interacao_id"])
    e = orq.partida.store.carregar("s")
    assert e.turno == 0 and e.declaracoes == []
    assert orq.memory.load_session_md("s") == ""           # sem fala orfa
    assert e.janela["estado"] == "aberta"
    assert e.janela["prazo_em"] == pytest.approx(rel() + 60)   # prazo inteiro de novo


# ─── Timer ───────────────────────────────────────────────────────────────────

def test_janela_comeca_ao_acionar_responder_e_e_idempotente():
    orq, rel = novo()
    j1 = run(orq.abrir_janela("s"))
    assert j1["prazo_s"] == 60 and j1["restante_s"] == 60
    rel.avanca(10)
    j2 = run(orq.abrir_janela("s"))        # duplo clique / recarga da tela
    assert j2["interacao_id"] == j1["interacao_id"]
    assert j2["restante_s"] == 50


def test_resposta_rapida_nao_tem_penalidade_nem_timeout():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(12)
    r = jogar(orq, Roteiro([(30, "Hum.")]), interacao_id=j["interacao_id"])
    assert r["tempo_resposta_s"] == 12 and r["penalidade_tempo"] == 0
    assert r["timeout"] is False and r["status_investigacao"]["nivel_suspeita"] == 30
    # E o timeout tardio do cliente e inofensivo:
    rel.avanca(60)
    out = run(orq.expirar_janela("s", j["interacao_id"]))
    assert out["aplicado"] is False
    assert orq.partida.store.carregar("s").turno == 1


def test_resposta_lenta_tem_efeito_pequeno_e_limitado():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(59)
    r = jogar(orq, Roteiro([(30, "Hum.")]), interacao_id=j["interacao_id"])
    assert r["penalidade_tempo"] == gr.REGRAS.PENALIDADE_TEMPO_ALTO == 2
    assert r["status_investigacao"]["nivel_suspeita"] == 30 + 2


def test_tempo_de_processamento_da_ia_nao_conta_como_demora_do_jogador():
    orq, rel = novo()

    class Lento(Roteiro):
        async def generate(self, prompt):
            rel.avanca(45)                 # a IA demora 45 s
            return await super().generate(prompt)

    j = run(orq.abrir_janela("s"))
    rel.avanca(5)                          # o jogador levou 5 s
    r = jogar(orq, Lento([(30, "Hum.")]), interacao_id=j["interacao_id"])
    assert r["tempo_resposta_s"] == 5 and r["penalidade_tempo"] == 0


def test_timeout_aplica_penalidade_uma_vez_sem_chamar_a_ia():
    orq, rel = novo()
    prov = Roteiro([])
    j = run(orq.abrir_janela("s"))
    rel.avanca(60)
    out = run(orq.expirar_janela("s", j["interacao_id"]))
    r = out["resposta"]
    assert out["aplicado"] is True and r["timeout"] is True
    assert r["status_investigacao"]["nivel_suspeita"] == gr.REGRAS.PENALIDADE_TIMEOUT
    assert r["id_turno"] == 1
    assert prov.prompts == []

    again = run(orq.expirar_janela("s", j["interacao_id"]))     # repetido
    assert again["aplicado"] is False
    e = orq.partida.store.carregar("s")
    assert e.turno == 1 and e.suspeita == gr.REGRAS.PENALIDADE_TIMEOUT

    # O texto do timeout NAO entra como fala do suspeito.
    assert "(SUSPEITO)" not in orq.memory.load_session_md("s")


def test_timeout_antecipado_e_recusado():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(20)
    with pytest.raises(JanelaInvalida):
        run(orq.expirar_janela("s", j["interacao_id"]))
    assert orq.partida.store.carregar("s").turno == 0


def test_timeout_de_interacao_desconhecida_e_recusado():
    orq, _ = novo()
    run(orq.abrir_janela("s"))
    with pytest.raises(JanelaInvalida):
        run(orq.expirar_janela("s", "inventada"))


def test_resposta_depois_do_prazo_vira_timeout_e_nao_vai_para_a_ia():
    orq, rel = novo()
    prov = Roteiro([(10, "nao deveria ser chamada")])
    j = run(orq.abrir_janela("s"))
    rel.avanca(70)
    r = jogar(orq, prov, "Cheguei tarde.", interacao_id=j["interacao_id"])
    assert r["timeout"] is True
    assert prov.prompts == []
    assert "Cheguei tarde." not in orq.memory.load_session_md("s")
    assert orq.partida.store.carregar("s").turno == 1


def test_resposta_e_timeout_simultaneos_geram_um_unico_efeito():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(59)

    async def corrida():
        prov = Roteiro([(30, "Hum.")])
        # a resposta chega a 59 s, o timeout do cliente logo depois (60 s)
        a = asyncio.create_task(orq.analyze("s", "Estava no beco.", provider=prov,
                                            interacao_id=j["interacao_id"]))
        await asyncio.sleep(0)
        rel.avanca(1)
        b = asyncio.create_task(orq.expirar_janela("s", j["interacao_id"]))
        return await asyncio.gather(a, b)

    resposta, timeout = run(corrida())
    assert resposta["timeout"] is False
    assert timeout["aplicado"] is False
    e = orq.partida.store.carregar("s")
    assert e.turno == 1 and e.suspeita == 30 + gr.penalidade_por_tempo(59)


def test_timeout_nao_acumula_apos_o_fim_da_partida():
    orq, rel = novo(SEM_TETO)
    j1 = run(orq.abrir_janela("s"))
    jogar(orq, Roteiro([(100, "Fim.")]), interacao_id=j1["interacao_id"])
    with pytest.raises(JanelaInvalida):
        run(orq.abrir_janela("s"))
    rel.avanca(100)
    out = run(orq.expirar_janela("s", j1["interacao_id"]))
    assert out["aplicado"] is False
    assert orq.partida.store.carregar("s").turno == 1


def test_timeouts_repetidos_levam_ao_fim_pelas_mesmas_regras():
    orq, rel = novo()
    suspeitas = []
    for _ in range(gr.REGRAS.MAX_TURNOS):
        j = run(orq.abrir_janela("s"))
        rel.avanca(61)
        r = run(orq.expirar_janela("s", j["interacao_id"]))["resposta"]
        suspeitas.append(r["status_investigacao"]["nivel_suspeita"])
        if r["status_investigacao"]["fim_de_jogo"]:
            break
    # +15 por timeout: 15, 30, 45, 60, 75, 90, 100 (teto) -> 100 no 7o turno.
    assert suspeitas == [15, 30, 45, 60, 75, 90, 100]
    assert r["status_investigacao"]["motivo_fim"] == "suspeita_maxima"
    assert r["id_turno"] == 7
