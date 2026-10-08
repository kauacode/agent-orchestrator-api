"""
Gera o backend executavel (dist/guilty-backend/) que vai junto com o jogo.

Uso (da raiz do projeto), depois de QUALQUER mudanca no backend:
    venv\\Scripts\\python.exe scripts\\empacotar_backend.py

Depois, o build do Unity copia dist/guilty-backend/ para a pasta do jogo
(ver BackendBuildCopy.cs no projeto Unity). O codigo-fonte continua sendo a
fonte da verdade: o executavel e so um produto gerado, refeito a cada mudanca.

Etapas: instala dependencias -> PyInstaller -> confere que nada sensivel
entrou no pacote -> sobe o executavel e testa /health e /providers.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIST = RAIZ / "dist" / "guilty-backend"
EXE = DIST / "guilty-backend.exe"
PORTA_TESTE = 8765


def _rodar(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=RAIZ, check=True)


def _conferir_conteudo():
    proibidos = [p for p in DIST.rglob("*")
                 if p.name == ".env" or p.suffix in (".gguf", ".part")]
    if proibidos:
        sys.exit(f"ERRO: arquivos que nao podem ir para o jogo: {proibidos}")
    if not (DIST / "_internal" / "app" / "content" / "cases").is_dir():
        sys.exit("ERRO: os casos (app/content/cases) nao entraram no pacote.")
    tamanho = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"Pacote: {DIST} ({tamanho / 1024**2:.0f} MB, sem .env e sem modelos)")


def _get(caminho):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORTA_TESTE}{caminho}", timeout=5) as r:
        return json.loads(r.read())


def _teste_de_fumaca():
    pasta = Path(tempfile.mkdtemp(prefix="guilty_backend_teste_"))
    env = {**os.environ,
           "GUILTY_PORT": str(PORTA_TESTE),
           "GUILTY_MODELS_DIR": str(pasta / "models"),
           "GUILTY_PARENT_PID": str(os.getpid())}
    # Sem variaveis do desenvolvedor: o teste tem que ver o que o jogador ve.
    for var in ("GEMINI_API_KEY", "LLM_TYPE", "LLAMA_CPP_MODEL_PATH"):
        env.pop(var, None)

    inicio = time.perf_counter()
    proc = subprocess.Popen([str(EXE)], cwd=pasta, env=env)
    try:
        while True:
            try:
                saude = _get("/health")
                break
            except OSError:
                if proc.poll() is not None:
                    sys.exit(f"ERRO: o executavel encerrou com codigo {proc.returncode}.")
                if time.perf_counter() - inicio > 60:
                    sys.exit("ERRO: o executavel nao respondeu /health em 60s.")
                time.sleep(0.5)

        print(f"/health em {time.perf_counter() - inicio:.1f}s: {saude}")
        assert saude.get("servico") == "guilty-backend", saude

        provedores = {p["id"]: p for p in _get("/providers")["providers"]}
        print("/providers:", {k: ("baixado" if v["baixado"] else "nao baixado") for k, v in provedores.items()})
        assert "gemini" in provedores and len(provedores) >= 2
        # Prova de que o .env de desenvolvimento nao vazou para o executavel.
        assert provedores["gemini"]["chave_no_servidor"] is False, "o executavel leu uma chave do Gemini!"
        print("Teste de fumaca OK.")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem-instalar", action="store_true",
                        help="pula o pip install (mais rapido quando nada mudou nas dependencias)")
    args = parser.parse_args()

    py = sys.executable
    if not args.sem_instalar:
        _rodar([py, "-m", "pip", "install", "-q", "-r", "requirements.txt",
                "-r", "requirements-llamacpp.txt", "pyinstaller"])
    _rodar([py, "-m", "PyInstaller", "guilty-backend.spec", "--noconfirm", "--clean"])
    _conferir_conteudo()
    _teste_de_fumaca()


if __name__ == "__main__":
    main()
