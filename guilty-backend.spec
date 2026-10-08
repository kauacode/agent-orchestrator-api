# -*- mode: python ; coding: utf-8 -*-
#
# Empacota o backend num executavel que o jogo inicia sozinho.
# Nao rode o PyInstaller direto: use scripts\empacotar_backend.bat, que
# tambem confere o resultado.
#
# Modo "onedir" (pasta com o .exe + _internal\) e nao "onefile": o onefile
# descompacta ~200 MB numa pasta temporaria A CADA abertura do jogo, o que
# somava segundos ao boot e deixava lixo no %TEMP% quando o jogo travava.
#
# console=True de proposito: o Unity abre o processo com a janela escondida,
# e sem console o sys.stdout vira None e o log do uvicorn quebra na partida.
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("app/content", "app/content")]
binaries = []
hiddenimports = []

# llama_cpp carrega a llama.dll (e as ggml-*.dll) por ctypes em tempo de
# execucao — o PyInstaller nao enxerga isso sozinho.
for pacote in ("llama_cpp",):
    d, b, h = collect_all(pacote)
    datas += d
    binaries += b
    hiddenimports += h

# uvicorn e google.genai escolhem implementacoes por import dinamico.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("google.genai")

a = Analysis(
    ["run_server.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Nada disso e usado pelo backend; so incharia a pasta do jogo.
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "huggingface_hub", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="guilty-backend",
    console=True,
    upx=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="guilty-backend",
)
