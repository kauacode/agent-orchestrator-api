import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from filelock import FileLock

class ContextMemoryManager:
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _session_md_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}_depoimento.md"

    def _session_version_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}_version.json"

    def append_turn(self, session_id: str, text: str, role: str = "") -> None:
        """
        Grava uma fala no histórico da sessão.

        `role` identifica quem falou (SUSPEITO / DETETIVE). Sem isso, o arquivo
        era uma lista plana de linhas e, quando voltava como contexto para o
        modelo, ele tinha que adivinhar quem tinha dito o quê — em sessões
        reais isso fazia o detetive atribuir ao jogador falas dele mesmo.
        Arquivos antigos, sem o marcador, continuam sendo lidos normalmente.
        """
        path = self._session_md_path(session_id)
        timestamp = datetime.now().isoformat()
        marcador = f"({role.upper()}) " if role else ""

        lock_path = path.with_suffix('.lock')
        with FileLock(lock_path):
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {marcador}- {text}\n")

        self._update_version(session_id)

    def load_session_md(self, session_id: str) -> str:
        path = self._session_md_path(session_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def get_sliding_window(self, session_id: str, window_size: int = 3) -> str:
        conteudo = self.load_session_md(session_id)
        if not conteudo:
            return "Nenhuma interação anterior."

        linhas = [linha.strip() for linha in conteudo.splitlines() if linha.strip()]

        turnos = []
        turno_atual = []

        for linha in linhas:
            if linha.startswith('['):
                if turno_atual:
                    turnos.append('\n'.join(turno_atual))
                turno_atual = [linha]
            else:
                turno_atual.append(linha)

        if turno_atual:
            turnos.append('\n'.join(turno_atual))

        ultimos_turnos = turnos[-window_size:] if len(turnos) > window_size else turnos

        return '\n\n'.join(f"Turno {i+1}: {t}" for i, t in enumerate(ultimos_turnos))

    def summarize(self, session_id: str, max_lines: int = 5) -> str:
        md = self.load_session_md(session_id)
        linhas = [l for l in md.splitlines() if l.strip()]

        ultimas_linhas = linhas[-max_lines:] if len(linhas) > max_lines else linhas

        vistos = set()
        linhas_unicas = []

        for linha in ultimas_linhas:
            texto_limpo = self._texto_sem_cabecalho(linha)
            if texto_limpo not in vistos:
                vistos.add(texto_limpo)
                linhas_unicas.append(linha)

        return "\n".join(linhas_unicas)

    @staticmethod
    def _texto_sem_cabecalho(linha: str) -> str:
        """
        Remove '[timestamp] (PAPEL) - ' e devolve só a fala.

        Usado para deduplicar. Procura o primeiro '- ' DEPOIS do ']' em vez do
        literal '] - ': com o marcador de papel no meio, o formato virou
        '] (SUSPEITO) - ' e a busca antiga nunca casava — o timestamp entrava
        na comparação e duas falas idênticas nunca eram vistas como repetidas.
        """
        fim_colchete = linha.find(']')
        if fim_colchete == -1:
            return linha.strip()

        resto = linha[fim_colchete + 1:]
        sep = resto.find('- ')
        return (resto[sep + 2:] if sep != -1 else resto).strip()

    def get_prompt_context(self, session_id: str) -> str:
        """
        Monta só o HISTÓRICO da sessão.

        Os fatos do crime saíram daqui: agora vêm do case_repository, montados
        pelo PromptOrchestrator. Antes este método procurava um
        data/fatos_crime.json que nunca existiu, e havia dois lugares diferentes
        tentando injetar fatos no mesmo prompt.
        """
        resumo = self.summarize(session_id, max_lines=5)
        janela = self.get_sliding_window(session_id, window_size=3)

        partes = []

        if resumo:
            partes.append(f"RESUMO DA SESSÃO (últimas interações):\n{resumo}")

        if janela and janela != "Nenhuma interação anterior.":
            partes.append(f"JANELA DESLIZANTE (últimos turnos completos):\n{janela}")

        if not partes:
            return "=== HISTÓRICO DA SESSÃO ===\nPrimeiro turno: nada foi dito ainda."

        return "=== HISTÓRICO DA SESSÃO ===\n" + '\n\n'.join(partes)

    def get_dialogo(self, session_id: str, max_falas: int = 6) -> List[tuple]:
        """
        Últimas falas como (papel, texto), sem timestamp.

        Para modelos pequenos. O get_prompt_context manda cada fala duas vezes
        (resumo e janela se sobrepõem) e com timestamp ISO: no Qwen 2.5 3B isso
        fez o detetive copiar palavra por palavra a própria fala anterior, que
        era a coisa mais repetida do prompt. Também é ~1/3 dos tokens.
        """
        falas = []
        for linha in self.load_session_md(session_id).splitlines():
            if not linha.strip():
                continue
            fim_colchete = linha.find(']')
            resto = linha[fim_colchete + 1:].strip() if fim_colchete != -1 else linha
            papel = ""
            if resto.startswith('(') and ')' in resto:
                papel = resto[1:resto.index(')')]
            falas.append((papel, self._texto_sem_cabecalho(linha)))
        return falas[-max_falas:]

    def finalizar_sessao(self, session_id: str) -> bool:
        """
        Apaga tudo da partida: historico, suspeita/contradicoes e o .lock.

        Cada partida e unica — uma partida nova nunca continua a anterior. Sem
        isto os arquivos ficavam para tras, e uma sessao reaproveitada por
        engano herdava a suspeita alta e ja nascia perdida.
        """
        md = self._session_md_path(session_id)
        removeu = False
        for caminho in (md, self._session_version_path(session_id), md.with_suffix('.lock')):
            try:
                if caminho.exists():
                    caminho.unlink()
                    removeu = True
            except OSError:
                pass
        return removeu

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """`max_age_days=0` apaga todas as sessoes."""
        apagados = 0
        limite = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)

        for arquivo in self.storage_dir.glob("*_depoimento.md"):
            if arquivo.stat().st_mtime <= limite:
                self.finalizar_sessao(arquivo.stem.replace('_depoimento', ''))
                apagados += 1

        return apagados

    def _update_version(self, session_id: str) -> None:
        caminho = self._session_version_path(session_id)
        
        info = {
            "session_id": session_id,
            "last_updated": datetime.now().isoformat(),
            "turn_count": len(self.load_session_md(session_id).splitlines())
        }

        try:
            if caminho.exists():
                with open(caminho, 'r', encoding='utf-8') as f:
                    atual = json.load(f)
                
                if type(atual) is dict:
                    if 'suspeita_history' in atual:
                        info['suspeita_history'] = atual['suspeita_history']
                    if 'contradictions' in atual:
                        info['contradictions'] = atual['contradictions']

            with open(caminho, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)
        except Exception:
            pass

    def append_suspicion(self, session_id: str, nivel_suspeita: int) -> None:
        caminho = self._session_version_path(session_id)
        dados = {
            "session_id": session_id,
            "last_updated": datetime.now().isoformat(),
            "turn_count": len(self.load_session_md(session_id).splitlines()),
            "suspeita_history": []
        }

        try:
            if caminho.exists():
                with open(caminho, 'r', encoding='utf-8') as f:
                    existente = json.load(f)
                
                dados.update(existente)
                if 'suspeita_history' not in dados:
                    dados['suspeita_history'] = []

            if 'suspeita_history' not in dados:
                dados['suspeita_history'] = []
                
            dados['suspeita_history'].append({
                "timestamp": datetime.now().isoformat(),
                "nivel": int(nivel_suspeita)
            })

            with open(caminho, 'w', encoding='utf-8') as f:
                json.dump(dados, f, indent=2)
        except Exception:
            pass

    def get_recent_suspicions(self, session_id: str, n: int = 3) -> List[int]:
        caminho = self._session_version_path(session_id)
        if not caminho.exists():
            return []
            
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            historico = dados.get('suspeita_history', [])
            valores = [int(item.get('nivel', 0)) for item in historico]
            return valores[-n:]
        except Exception:
            return []

    def append_contradiction(self, session_id: str, turno_id: int) -> None:
        caminho = self._session_version_path(session_id)
        dados = {
            "session_id": session_id,
            "last_updated": datetime.now().isoformat(),
            "turn_count": len(self.load_session_md(session_id).splitlines()),
            "contradictions": []
        }

        try:
            if caminho.exists():
                with open(caminho, 'r', encoding='utf-8') as f:
                    existente = json.load(f)
                
                dados.update(existente)
                if 'contradictions' not in dados:
                    dados['contradictions'] = []

            if 'contradictions' not in dados:
                dados['contradictions'] = []

            dados['contradictions'].append({
                "turno": int(turno_id),
                "timestamp": datetime.now().isoformat()
            })

            with open(caminho, 'w', encoding='utf-8') as f:
                json.dump(dados, f, indent=2)
        except Exception:
            pass

    def get_contradictions(self, session_id: str) -> List[Dict[str, Any]]:
        caminho = self._session_version_path(session_id)
        if not caminho.exists():
            return []
            
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            return dados.get('contradictions', [])
        except Exception:
            return []

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        caminho = self._session_version_path(session_id)
        if caminho.exists():
            try:
                with open(caminho, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "session_id": session_id,
            "last_updated": None,
            "turn_count": 0
        }