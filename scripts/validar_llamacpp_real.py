"""
Validacao de ponta a ponta do provedor llamacpp com o modelo REAL em disco.

Roda uma sessao de 4 turnos do Caso 01 pela mesma funcao da rota
/interrogate (memoria, orquestrador, provedor, regras de fim de jogo) e checa
contrato, vazamento de spoiler, desempenho e coerencia.

Uso (da raiz do projeto):
    venv\\Scripts\\python.exe scripts\\validar_llamacpp_real.py [--rodadas N]

Com temperatura > 0 uma rodada e anedota: --rodadas 3 repete a sessao inteira
(mesmo modelo carregado) e consolida as metricas no final.

Codigo de saida 0 so se todas as checagens OBRIGATORIAS passarem. As de
comportamento (detectou mentira onde devia, suspeita subiu) sao reportadas a
parte: dependem de amostragem e de julgamento, nao de contrato.
"""
import argparse
import asyncio
import difflib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dotenv import load_dotenv

load_dotenv(RAIZ / ".env")

import llama_cpp

from app.main import interrogate
from app.schemas import InterrogateRequest, ResponseContract
from app.services.llama_cpp_provider import LlamaCppProvider

logging.disable(logging.CRITICAL)

TURNOS = [
    "Boa noite, detetive. Eu estava na minha casa dormindo às 22h, não sei nada sobre o que aconteceu na Rua 7.",
    "Sim, eu trabalho na gráfica como operador de impressão, mas saí às 18h e fui direto para casa.",
    "Pensando bem, passei perto do galpão por volta das 21h30 só para pegar um casaco que esqueci, mas não vi ninguém.",
    "O Marcelo Aguiar é quem gerenciava os horários e tinha as chaves de tudo, por que você não pergunta para ele?",
]

# O que se espera de um detetive coerente com as EVIDENCIAS do caso (o modelo
# local nao recebe a verdade):
#   T1: dormindo em casa as 22h nao colide com nada (o cracha marca 20:48).
#   T2: "sai as 18h" colide com o cracha, que registra saida as 20:48.
#   T3: "fui direto para casa" (T2) vs "passei no galpao as 21h30" — mudou a historia.
#   T4: desvio para outra pessoa: evasiva, nao mentira.
ESPERADO_MENTIRA = [False, True, True, False]

# Fatos que so existem na secao VERDADE do caso. O jogador nao os disse e as
# evidencias nao os contem: se aparecerem na fala do detetive, vazou.
MARCADORES_DE_VERDADE = [
    r"desvi\w* de dinheiro", r"\bfuma(r|ndo|va)\b", r"\bbeco\b",
    r"\b20[:h]15\b", r"\b20[:h]26\b", r"\b19[:h]40\b",
    r"marcelo\W+(\w+\W+){0,4}(matou|assassin|culpad|autor do crime)",
    r"(matou|assassin|culpad)\w*\W+(\w+\W+){0,4}marcelo",
]

FALLBACK_ORQUESTRADOR = "Interessante... você disse:"


def _perf(llm):
    return llama_cpp.llama_perf_context(llm._ctx.ctx)


def _reset_perf(llm):
    llama_cpp.llama_perf_context_reset(llm._ctx.ctx)


def _checar_saida_bruta(texto: str) -> list:
    """Problemas na saida crua do modelo, antes de qualquer correcao do backend."""
    problemas = []
    if "<think" in texto.lower():
        problemas.append("tag <think> na saida")
    if "```" in texto:
        problemas.append("cerca markdown na saida")
    limpo = texto.strip()
    if not (limpo.startswith("{") and limpo.endswith("}")):
        problemas.append("texto fora do objeto JSON")
    try:
        ResponseContract.model_validate(json.loads(limpo))
    except Exception as exc:
        problemas.append(f"saida crua nao valida no contrato: {type(exc).__name__}")
    return problemas


def _vazamentos(texto: str) -> list:
    return [m for m in MARCADORES_DE_VERDADE if re.search(m, texto, re.IGNORECASE)]


async def _sessao(provider: LlamaCppProvider, sessao: str) -> list:
    saidas_brutas = []
    original = provider._inferir

    def inferir_capturando(prompt):
        resposta = original(prompt)
        saidas_brutas.append((prompt, resposta))
        return resposta

    provider._inferir = inferir_capturando

    resultados = []
    for i, fala in enumerate(TURNOS, 1):
        _reset_perf(provider._llm)
        n_brutas_antes = len(saidas_brutas)

        inicio = time.perf_counter()
        resposta = await interrogate(InterrogateRequest(session_id=sessao, player_text=fala),
                                     provider=provider)
        total = time.perf_counter() - inicio

        p = _perf(provider._llm)
        tentativas = saidas_brutas[n_brutas_antes:]
        prompt, bruta = tentativas[-1]
        texto_bruto = bruta["choices"][0]["message"]["content"]
        uso = bruta.get("usage") or {}

        resultados.append({
            "turno": i,
            "fala": fala,
            "resposta": resposta,
            "texto_bruto": texto_bruto,
            "prompt_enviado": prompt,
            "tentativas": len(tentativas),
            "tempo_total_s": total,
            "prompt_tokens": uso.get("prompt_tokens"),
            "tokens_processados": p.n_p_eval,
            "tokens_reaproveitados": (uso.get("prompt_tokens") or 0) - p.n_p_eval,
            "t_prompt_s": p.t_p_eval_ms / 1000,
            "tokens_gerados": p.n_eval,
            "t_geracao_s": p.t_eval_ms / 1000,
            "ttft_s": (p.t_p_eval_ms + (p.t_eval_ms / p.n_eval if p.n_eval else 0)) / 1000,
            "tok_s": p.n_eval / (p.t_eval_ms / 1000) if p.t_eval_ms else 0.0,
        })
        print(f"  turno {i}: {total:.1f}s", flush=True)

    return resultados


def _rodada(provider: LlamaCppProvider, n: int):
    sessao = f"validacao_llamacpp_{datetime.now():%Y%m%d_%H%M%S}_r{n}"
    print(f"\n##### RODADA {n} ({sessao})")
    resultados = asyncio.run(_sessao(provider, sessao))

    # ─── Checagens ───────────────────────────────────────────────────────────
    obrigatorias, comportamento = [], []

    def checar(lista, ok, descricao):
        lista.append((ok, descricao))

    for r in resultados:
        t, resp = r["turno"], r["resposta"]
        status = resp["status_investigacao"]

        try:
            ResponseContract.model_validate(resp)
            checar(obrigatorias, True, f"T{t}: resposta valida no ResponseContract")
        except Exception as exc:
            checar(obrigatorias, False, f"T{t}: ResponseContract falhou ({exc})")

        problemas = _checar_saida_bruta(r["texto_bruto"])
        checar(obrigatorias, not problemas,
               f"T{t}: saida crua limpa (sem <think>, markdown ou texto solto)"
               + (f" — {problemas}" if problemas else ""))
        # Comportamento, nao contrato: o orquestrador refaz de proposito
        # quando a fala sai repetida.
        checar(comportamento, r["tentativas"] == 1,
               f"T{t}: resolvido na 1a tentativa ({r['tentativas']} usadas)")
        checar(obrigatorias, not resp["texto_detetive"].startswith(FALLBACK_ORQUESTRADOR),
               f"T{t}: nao caiu no fallback do orquestrador")

        vaz = _vazamentos(resp["texto_detetive"])
        checar(obrigatorias, not vaz, f"T{t}: anti-spoiler" + (f" — VAZOU {vaz}" if vaz else ""))
        checar(obrigatorias, "VERDADE DO CASO" not in r["prompt_enviado"]
               and "desvio de dinheiro" not in r["prompt_enviado"],
               f"T{t}: verdade confidencial fora do prompt")

        esperado = ESPERADO_MENTIRA[t - 1]
        checar(comportamento, status["detectou_mentira"] == esperado,
               f"T{t}: detectou_mentira={status['detectou_mentira']} (esperado {esperado})")

    # Comparar com o T1 nao serve: da 2a rodada em diante o T1 ja acha os
    # blocos fixos no cache da rodada anterior. O que importa e cada turno
    # seguinte reprocessar so a parte que mudou.
    for r in resultados[1:]:
        checar(obrigatorias, r["tokens_processados"] < r["prompt_tokens"] / 2,
               f"T{r['turno']}: KV cache — reaproveitou {r['tokens_reaproveitados']} de "
               f"{r['prompt_tokens']} tokens, processou so {r['tokens_processados']}")

    s = [r["resposta"]["status_investigacao"]["nivel_suspeita"] for r in resultados]
    checar(comportamento, s[2] > s[0], f"suspeita subiu do T1 ao T3 ({s[0]} -> {s[2]})")
    checar(comportamento, s[1] >= s[0], f"suspeita nao caiu no T2 apos colidir com o cracha ({s[0]} -> {s[1]})")

    # Eco: o detetive devolvendo uma fala anterior dele (visto no 3B).
    falas_det = [r["resposta"]["texto_detetive"] for r in resultados]
    for i in range(1, len(falas_det)):
        parecida = max(difflib.SequenceMatcher(None, falas_det[i], f).ratio() for f in falas_det[:i])
        checar(comportamento, parecida < 0.8,
               f"T{i + 1}: fala nova (similaridade maxima com falas anteriores: {parecida:.0%})")

    # ─── Relatorio ───────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'T':>2} | {'total':>6} | {'TTFT':>6} | {'tok/s':>5} | {'prompt':>6} | {'process.':>8} | "
          f"{'reuso':>5} | {'gerados':>7} | {'susp.':>5} | {'mentira':>7}")
    print("-" * 100)
    for r in resultados:
        st = r["resposta"]["status_investigacao"]
        print(f"{r['turno']:>2} | {r['tempo_total_s']:>5.1f}s | {r['ttft_s']:>5.1f}s | {r['tok_s']:>5.1f} | "
              f"{r['prompt_tokens']:>6} | {r['tokens_processados']:>8} | {r['tokens_reaproveitados']:>5} | "
              f"{r['tokens_gerados']:>7} | {st['nivel_suspeita']:>5} | {str(st['detectou_mentira']):>7}")
    print("=" * 100)

    for r in resultados:
        print(f"\nT{r['turno']} JOGADOR : {r['fala']}")
        print(f"   DETETIVE: {r['resposta']['texto_detetive']}")
        fv = r["resposta"]["feedback_visual"]
        print(f"   visual  : {fv['cor_iluminacao']} / {fv['bpm_musica']} bpm / {fv['animacao_trigger']}")

    print("\nCHECAGENS OBRIGATORIAS")
    for ok, d in obrigatorias:
        print(f"  [{'OK' if ok else 'FALHA'}] {d}")
    print("\nCOMPORTAMENTO (dependente de amostragem)")
    for ok, d in comportamento:
        print(f"  [{'OK' if ok else 'DIVERGE'}] {d}")

    pasta = RAIZ / "logs"
    pasta.mkdir(exist_ok=True)
    arquivo = pasta / f"{sessao}.json"
    arquivo.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "prompt_enviado"} for r in resultados],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalhes salvos em {arquivo.relative_to(RAIZ)}")

    return resultados, obrigatorias, comportamento


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rodadas", type=int, default=1)
    args = parser.parse_args()

    provider = LlamaCppProvider()
    caminho = provider.model_path
    print(f"llama-cpp-python {llama_cpp.__version__}")
    print(f"modelo: {caminho} ({caminho.stat().st_size / 1024**3:.2f} GiB)")
    print(f"n_ctx={provider.n_ctx} n_threads={provider.n_threads} "
          f"n_gpu_layers={provider.n_gpu_layers} temperatura={provider.temperature}")

    inicio = time.perf_counter()
    provider._llm = provider._carregar()
    print(f"carregado em {time.perf_counter() - inicio:.1f}s")

    obrigatorias, comportamento, resultados = [], [], []
    for n in range(1, args.rodadas + 1):
        res, obr, comp = _rodada(provider, n)
        resultados += res
        obrigatorias += obr
        comportamento += comp

    provider._llm.close()

    tempos = [r["tempo_total_s"] for r in resultados]
    ttft_quente = [r["ttft_s"] for r in resultados if r["turno"] > 1]
    print("\n" + "#" * 100)
    print(f"CONSOLIDADO ({args.rodadas} rodada(s), {len(resultados)} turnos)")
    print(f"  tempo/turno : media {sum(tempos) / len(tempos):.1f}s, pior {max(tempos):.1f}s")
    print(f"  tok/s medio : {sum(r['tok_s'] for r in resultados) / len(resultados):.1f}")
    if ttft_quente:
        print(f"  TTFT T2+    : media {sum(ttft_quente) / len(ttft_quente):.1f}s (KV cache quente)")
    falhas = sum(1 for ok, _ in obrigatorias if not ok)
    print(f"\n{len(obrigatorias) - falhas}/{len(obrigatorias)} obrigatorias, "
          f"{sum(ok for ok, _ in comportamento)}/{len(comportamento)} de comportamento")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
