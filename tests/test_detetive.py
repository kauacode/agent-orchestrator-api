"""
Inteligencia do detetive: interpretacao da fala, confissao, repeticao de
perguntas, contradicao e partida encerrada (cenarios A–I do enunciado).

Sem modelo real: um provedor roteirizado devolve falas controladas e guarda o
prompt recebido, que e o que de fato decide o comportamento do modelo.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.case_repository import get_case
from app.services.interpretador import (
    AMBIGUA, CONFISSAO, EVASIVA, INFORMACAO, NEGACAO, carregar_temas, classificar,
    temas_da_pergunta,
)
from tests.test_regras_e_timer import Roteiro, SEM_TETO, jogar, novo, run

TEMAS = carregar_temas(get_case())


def tipo(texto):
    return classificar(texto, TEMAS).tipo


# ─── Interpretacao (nao por palavra solta) ───────────────────────────────────

@pytest.mark.parametrize("texto", [
    "Eu matei a vítima.",
    "eu matei mesmo fdp",                    # o bug do turno 11
    "Fui eu. Eu matei ele.",
    "matei sim, e daí?",
    "eu o matei",
    "Eu matei o Otávio com o pé de cabra",
    "não nego que matei",                    # dupla negacao = admite
    "A culpa é minha, eu confesso.",
    "EU MATEIIII MESMO",
])
def test_confissoes_sao_reconhecidas(texto):
    assert tipo(texto) == CONFISSAO


@pytest.mark.parametrize("texto", [
    "Eu não matei a vítima.",
    "não fui eu",
    "Eu nunca matei ninguém!",
    "sou inocente",
    "Não tenho nada a ver com isso.",
])
def test_negacoes_sao_reconhecidas(texto):
    assert tipo(texto) == NEGACAO


@pytest.mark.parametrize("texto,esperado", [
    ("não sei", EVASIVA),
    ("prefiro não falar sobre isso", EVASIVA),
    ("quero meu advogado", EVASIVA),
    ("se eu tivesse matado, você acha que eu estaria aqui?", AMBIGUA),
    ("você acha que eu matei?", AMBIGUA),
    ("hm", AMBIGUA),
    ("", AMBIGUA),
    # "nao" sozinho nao e negacao de autoria:
    ("Eu não estava no depósito às 20:15, estava no beco.", INFORMACAO),
    ("Não, a porta lateral estava trancada quando saí.", INFORMACAO),
    # terceira pessoa nao e confissao:
    ("O Marcelo matou o Otávio, eu vi o carro dele", INFORMACAO),
    ("bati o ponto às 19h e fui para as máquinas", INFORMACAO),
])
def test_nao_depende_de_palavra_solta(texto, esperado):
    assert tipo(texto) == esperado


def test_temas_tocados_pela_fala():
    i = classificar("Eu estava no beco fumando, sozinho, das 20:05 às 20:35", TEMAS)
    assert "paradeiro_janela" in i.temas


def test_tema_da_pergunta_pega_o_foco_e_nao_todo_tema_citado():
    # Pergunta sobre a porta lateral cita "fumar", mas o foco e a porta.
    assert temas_da_pergunta(TEMAS, "Quem tem acesso à porta lateral quando o senhor saiu para fumar?") == ["porta_lateral"]
    # Citar a evidencia 20:48 nao e perguntar de novo o paradeiro das 20h.
    assert "paradeiro_janela" not in temas_da_pergunta(
        TEMAS, "O crachá registra saída às 20:48. O que fez até lá?")


# ─── Cenários A/B: confissão ─────────────────────────────────────────────────

@pytest.mark.parametrize("texto", ["Eu matei a vítima.", "eu matei mesmo fdp"])
def test_confissao_chega_ao_modelo_como_confissao_e_nao_como_negacao(texto):
    orq, _ = novo()
    prov = Roteiro([(40, "Entendo. Como foi, com o quê?")])
    r = jogar(orq, prov, texto)

    prompt = prov.prompts[0]
    assert "Tipo: CONFISSAO" in prompt
    assert "Isto NÃO é uma negação" in prompt
    assert "NÃO volte a perguntar horário" in prompt
    assert r["interpretacao"] == "confissao"
    # a confissao e registrada no estado e NAO encerra a partida sozinha
    e = orq.partida.store.carregar("s")
    assert e.confessou is True and e.encerrada is False
    assert r["status_investigacao"]["fim_de_jogo"] is False


def test_confissao_leva_a_suspeita_a_um_piso_alto_e_confirmar_encerra():
    orq, _ = novo()
    prov = Roteiro([(10, "Como foi?"), (10, "Então confirma?")])
    r1 = jogar(orq, prov, "Eu matei a vítima.")
    assert r1["status_investigacao"]["nivel_suspeita"] == 85       # nao cresce so uns pontos
    assert r1["status_investigacao"]["fim_de_jogo"] is False        # a 1a confissao nao encerra
    r2 = jogar(orq, prov, "Sim, eu matei mesmo, podem me prender.")
    assert r2["status_investigacao"]["nivel_suspeita"] == 100
    assert r2["status_investigacao"]["fim_de_jogo"] is True
    assert r2["status_investigacao"]["resultado"] == "derrota"


def test_bug_do_turno_11_modelo_volta_a_perguntar_horario_apos_confissao():
    """O jogador ja disse onde estava; confessa; o modelo repete a pergunta do horario."""
    orq, _ = novo()
    repete = (30, "Senhor Reis, onde o senhor estava entre 20h00 e 20h35?")
    prov = Roteiro([(30, "Entendi, no beco. Alguém confirma?"), repete, repete])
    jogar(orq, prov, "Eu estava no beco fumando, sozinho, das 20:05 às 20:35.")
    r = jogar(orq, prov, "eu matei mesmo fdp")

    assert "onde o senhor estava entre 20h00" not in r["texto_detetive"]
    # a 2a tentativa recebeu a correcao explicita
    assert any("CORREÇÃO OBRIGATÓRIA" in p for p in prov.prompts)
    # e a fala final reage a confissao em vez de repetir o horario
    assert "matou" in r["texto_detetive"] or "como foi" in r["texto_detetive"].lower()


# ─── Cenário C: negação ──────────────────────────────────────────────────────

def test_negacao_e_reconhecida_no_prompt():
    orq, _ = novo()
    prov = Roteiro([(30, "Certo.")])
    r = jogar(orq, prov, "Eu não matei a vítima.")
    assert "Tipo: NEGACAO" in prov.prompts[0]
    assert r["interpretacao"] == "negacao"
    assert orq.partida.store.carregar("s").negou_autoria is True


# ─── Cenário D: pergunta já respondida ───────────────────────────────────────

def test_pergunta_ja_respondida_nao_e_repetida():
    orq, _ = novo()
    repete = (30, "E onde exatamente o senhor estava entre 20h00 e 20h35?")
    prov = Roteiro([(30, "Alguém viu o senhor lá?"), repete, repete])
    jogar(orq, prov, "Eu estava no beco fumando, sozinho, das 20:05 às 20:35.")
    r = jogar(orq, prov, "Ja disse, no beco.")

    assert "20h00 e 20h35" not in r["texto_detetive"]
    e = orq.partida.store.carregar("s")
    assert e.temas["paradeiro_janela"]["respondido"] is True
    # o prompt do 2o turno lista o tema como ja esclarecido
    assert "Já esclarecido" in prov.prompts[1]
    assert "Onde Daniel esteve entre 20:00 e 20:35" in prov.prompts[1]


def test_pergunta_com_outras_palavras_tambem_e_pega():
    orq, _ = novo()
    prov = Roteiro([(30, "Entendo."),
                    (30, "Onde o senhor estava às 20:15, quando Otávio morreu?"),
                    (30, "Me diga onde o senhor estava às 20:15.")])
    jogar(orq, prov, "Eu estava fumando no beco das 20:05 às 20:35.")
    r = jogar(orq, prov, "Foi isso.")
    assert "20:15" not in r["texto_detetive"]


def test_retomar_assunto_quando_o_jogador_traz_algo_novo_sobre_ele():
    orq, _ = novo()
    prov = Roteiro([(30, "Entendi."),
                    (30, "Onde o senhor estava entre 20h00 e 20h35, exatamente? Pode repetir o horário que saiu?")])
    jogar(orq, prov, "Eu estava no beco fumando, sozinho, das 20:05 às 20:35.")
    r = jogar(orq, prov, "Na verdade fiquei no beco só até 20:20, saí antes.")   # corrige: toca o tema
    assert "20h00 e 20h35" in r["texto_detetive"]       # perguntar de novo e legitimo


# ─── Cenário E: contradição ──────────────────────────────────────────────────

def test_mudar_de_versao_e_contradicao_detectada_pelo_sistema():
    orq, _ = novo()
    prov = Roteiro([(30, "Certo."), (30, "Certo.")])
    jogar(orq, prov, "Eu não matei a vítima.")
    r = jogar(orq, prov, "Tá bom, fui eu, eu matei.")
    assert "MUDOU DE VERSÃO" in prov.prompts[1]
    assert r["status_investigacao"]["detectou_mentira"] is True
    assert orq.memory.get_contradictions("s")


# ─── Cenário F: resposta ambígua/evasiva ─────────────────────────────────────

def test_resposta_ambigua_pede_esclarecimento_sem_inventar():
    orq, _ = novo()
    prov = Roteiro([(30, "Alguém confirma?")])
    jogar(orq, prov, "hm")
    assert "Tipo: AMBIGUA" in prov.prompts[0]
    assert "Peça esclarecimento direto" in prov.prompts[0]


def test_evasiva_nao_conta_como_resposta_e_pode_ser_reperguntada_com_limite():
    orq, _ = novo()
    pergunta = "Onde o senhor estava entre 20h00 e 20h35?"
    prov = Roteiro([(30, pergunta)] * 8)
    jogar(orq, prov, "oi")
    for _ in range(2):
        jogar(orq, prov, "não sei")
    e = orq.partida.store.carregar("s")
    assert e.temas["paradeiro_janela"]["respondido"] is False       # evasiva nao responde
    r = jogar(orq, prov, "não sei de novo")
    # insistiu demais no mesmo tema: o sistema avanca para outro ponto
    assert "20h00 e 20h35" not in r["texto_detetive"]


# ─── Cenário G: conhecimento do suspeito ─────────────────────────────────────

def test_prompt_separa_o_que_o_suspeito_sabe_e_proibe_presumir():
    orq, _ = novo()
    prov = Roteiro([(30, "Certo.")])
    jogar(orq, prov, "Eu estava no beco.")
    p = prov.prompts[0]
    assert "O que o suspeito de fato viveu" in p
    assert "não presuma" in p
    # o detetive nao conhece o nome do culpado como "evidencia"
    assert "EVIDÊNCIAS EM SUAS MÃOS (pode citar)" in p
    assert "VERDADE DO CASO (CONFIDENCIAL — NUNCA REVELE)" in p


def test_modelo_pequeno_nao_recebe_a_verdade_nem_o_conhecimento_do_suspeito():
    orq, _ = novo()
    prov = Roteiro([(30, "Certo.")])
    prov.suporta_contexto_confidencial = False
    jogar(orq, prov, "Eu estava no beco.")
    assert "VERDADE DO CASO" not in prov.prompts[0]
    assert "Marcelo" not in prov.prompts[0]


# ─── Cenário H: histórico extenso ────────────────────────────────────────────

def test_historico_extenso_nao_gera_ciclo_de_perguntas_repetidas():
    orq, _ = novo()
    # Modelo "teimoso": sempre devolve a mesma pergunta sobre o horario.
    teimoso = (35, "Onde o senhor estava entre 20h00 e 20h35, senhor Reis?")
    prov = Roteiro([teimoso] * 60)
    falas = ["Eu estava no beco fumando das 20:05 às 20:35, sozinho.",
             "Discuti com o Otávio por causa das horas extras.",
             "Saí pela recepção, o crachá mostra.",
             "Não vi ninguém, só ouvi um barulho.",
             "A porta lateral ficava destrancada, acho.",
             "Não sei do pé de cabra.",
             "Já disse tudo."]
    textos = []
    for f in falas:
        textos.append(jogar(orq, prov, f)["texto_detetive"])
    # depois da 1a resposta, nenhuma fala volta a perguntar o horario ja esclarecido
    assert all("20h00 e 20h35" not in t for t in textos[1:])
    # e as perguntas seguintes avancam por temas diferentes
    assert len(set(textos[1:])) >= 3


# ─── Cenário I: partida encerrada ────────────────────────────────────────────

def test_partida_encerrada_nao_reinicia_nem_muda_o_estado_final():
    orq, _ = novo(SEM_TETO)
    prov = Roteiro([(100, "Acabou.")])
    jogar(orq, prov, "Eu matei mesmo.")
    antes = orq.partida.store.carregar("s").to_dict()

    r = jogar(orq, prov, "Não, espera, eu não matei!")
    depois = orq.partida.store.carregar("s").to_dict()
    assert r["ignorado"] is True
    assert r["status_investigacao"]["fim_de_jogo"] is True
    assert antes == depois


# ─── Revelação progressiva ───────────────────────────────────────────────────

def test_evidencias_tem_turno_de_liberacao_no_prompt():
    orq, _ = novo()
    prov = Roteiro([(30, "Certo.")])
    jogar(orq, prov, "Eu estava no beco.")
    p = prov.prompts[0]
    assert "(só cite a partir do turno 5)" in p              # pé-de-cabra
    assert "Evidências que você já pode citar neste turno: laudo_horario, cracha_saida" in p
    assert "objeto_ausente" not in p.split("Evidências que você já pode citar neste turno:")[1]


# ─── Pergunta de abertura, resposta curta e timeout repetindo a pergunta certa ─

def test_resposta_curta_a_pergunta_de_abertura_a_deixa_esclarecida():
    orq, _ = novo()
    # "em casa" (2 palavras) responde ao "Onde voce estava...?" que o jogo mostra de abertura.
    ruim = (30, "Onde o senhor estava entre 20h00 e 20h35, e alguém pode confirmar?")
    prov = Roteiro([(30, "Certo, e depois?"), ruim, ruim])
    jogar(orq, prov, "em casa")
    e = orq.partida.store.carregar("s")
    assert e.temas["paradeiro_janela"]["respondido"] is True
    r = jogar(orq, prov, "fiquei lendo")
    assert "20h00 e 20h35" not in r["texto_detetive"]


def test_timeout_repete_a_ultima_pergunta_do_detetive_e_nao_um_tema_antigo():
    orq, rel = novo()
    prov = Roteiro([(30, "Entendi. Quem mais trabalhava naquele turno com o senhor?")])
    jogar(orq, prov, "em casa")
    j = {"interacao_id": orq.partida.store.carregar("s").janela["id"]}
    rel.avanca(61)
    r = run(orq.expirar_janela("s", j["interacao_id"]))["resposta"]
    assert r["texto_detetive"].endswith("Quem mais trabalhava naquele turno com o senhor?")
    assert "20h00" not in r["texto_detetive"]


def test_timeout_logo_na_abertura_repete_a_pergunta_de_abertura():
    orq, rel = novo()
    j = run(orq.abrir_janela("s"))
    rel.avanca(61)
    r = run(orq.expirar_janela("s", j["interacao_id"]))["resposta"]
    assert r["texto_detetive"].endswith("Onde você estava na noite do dia 14 de março?")


def test_o_tom_da_fala_e_avaliado_pelo_modelo_via_instrucao_no_prompt():
    """Nao ha lista de palavroes no codigo: o prompt manda o modelo avaliar o tom."""
    orq, _ = novo()
    prov = Roteiro([(30, "Cuidado com o tom, senhor Reis.")])
    jogar(orq, prov, "eu matei mesmo")
    assert "Avalie você mesmo o TOM da fala" in prov.prompts[0]
    assert "agressivo, ofensivo, debochado ou provocador" in prov.prompts[0]


def test_nao_ha_lista_de_xingamentos_no_codigo():
    import app.services.interpretador as m
    assert not hasattr(m, "_HOSTIL")
