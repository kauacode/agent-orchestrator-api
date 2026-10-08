"""
Ponto de entrada do backend quando ele roda como executavel (PyInstaller),
iniciado pelo proprio jogo (BackendLauncher.cs no Unity).

Variaveis de ambiente que o Unity passa:
  GUILTY_PORT        porta HTTP (padrao 8000)
  GUILTY_MODELS_DIR  onde ficam/baixam os .gguf (pasta de dados do usuario)
  GUILTY_PARENT_PID  PID do jogo: se ele morrer, o backend morre junto

Em desenvolvimento continua valendo o jeito de sempre:
  venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000
mas este arquivo tambem funciona: venv\\Scripts\\python.exe run_server.py
"""
import os
import sys
import threading
import time


def _vigiar_processo_pai(pid: int) -> None:
    """
    Encerra o backend quando o jogo fecha — inclusive quando ele trava ou e
    morto pelo Gerenciador de Tarefas, caso em que o Unity nao tem chance de
    encerrar o filho. Sem isto, o backend orfao segurava a porta e ~3 GB de RAM.
    """
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        INFINITE = 0xFFFFFFFF
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            # O pai ja nao existe (ou nao temos permissao): nao ha quem esperar.
            os._exit(0)
        # Bloqueia a thread ate o processo do jogo terminar. Nao usa CPU.
        kernel32.WaitForSingleObject(handle, INFINITE)
        os._exit(0)
    else:
        # os.kill(pid, 0) so testa existencia em POSIX. No Windows ele MATA o
        # processo — por isso o ramo acima usa a API nativa.
        while True:
            try:
                os.kill(pid, 0)
            except OSError:
                os._exit(0)
            time.sleep(2)


def main() -> None:
    # Console do Windows em cp1252 quebra os logs com acento.
    for fluxo in (sys.stdout, sys.stderr):
        if fluxo is not None and hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace")

    pai = os.getenv("GUILTY_PARENT_PID")
    if pai and pai.isdigit():
        threading.Thread(target=_vigiar_processo_pai, args=(int(pai),), daemon=True).start()

    import uvicorn

    from app.main import app

    porta = int(os.getenv("GUILTY_PORT", "8000"))
    # 127.0.0.1 e nao 0.0.0.0: o backend so atende o jogo da propria maquina.
    # Exposto na rede, qualquer um na mesma Wi-Fi gastaria a cota do Gemini
    # com a chave do jogador.
    uvicorn.run(app, host="127.0.0.1", port=porta, log_level="info")


if __name__ == "__main__":
    main()
