import asyncio
import re
import copy
import difflib
import json
import logging
from typing import Dict, Any, List, Optional

from app.schemas import ResponseContract
from app.services import game_rules as gr
from app.services.case_repository import get_case
from app.services.interpretador import (
    AMBIGUA, CONFISSAO, EVASIVA, INFORMACAO, NEGACAO, Interpretacao, Tema,
    carregar_temas, classificar,
)
from app.services.llm_providers import LLMGenerationError, LLMProvider
from app.services.frases_detetive import escolher_frase
from app.services.partida_service import (
    JanelaInvalida, PartidaService, PartidaStore,
)

logger = logging.getLogger(__name__)

class PromptOrchestrator:
    def __init__(self, memory):
        self.memory = memory
        # Espelho do turno de cada sessao (a fonte da verdade e o estado
        # persistido pelo PartidaService). Fica por compatibilidade.
        self.turn_counter = {}
        self.exemplos = self._get_few_shot_examples()
        self.partida = PartidaService(PartidaStore(str(memory.storage_dir)))
        # Um lock por sessao: resposta e timeout da MESMA partida nunca correm
        # juntos (FIFO — quem chegou primeiro decide).
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._locks[session_id] = asyncio.Lock()
        return lock

    def _semear_abertura(self, estado: gr.EstadoPartida, caso: Optional[Dict[str, Any]]) -> None:
        """
        A fala de abertura (cliente) ja e uma pergunta. Registra-a no estado do
        turno 0: sem isso a 1a resposta do jogador ("em casa") nao era ligada a
        pergunta nenhuma, o tema ficava em aberto e o detetive o repetia depois.
        """
        abertura = (caso or {}).get("abertura")
        if not abertura or estado.turno != 0 or estado.ultima_pergunta:
            return
        estado.ultima_pergunta = abertura.get("pergunta", "")
        tema = abertura.get("tema")
        if tema:
            estado.tema_pergunta_atual = tema
            estado.temas.setdefault(tema, {"perguntado": 0, "respondido": False, "evasivas": 0,
                                           "resposta": "", "textos": []})["perguntado"] += 1

    def esquecer_sessao(self, session_id: str) -> None:
        """Chamado ao finalizar a partida: solta lock, contador e estado."""
        self._locks.pop(session_id, None)
        self.turn_counter.pop(session_id, None)
        self.partida.store.apagar(session_id)

    def _get_few_shot_examples(self) -> str:
        """
        Exemplos deliberadamente ABSTRATOS.

        A versão anterior citava uma biblioteca e um horário de 20:00. Como o
        arquivo de fatos nunca existiu, esses detalhes inventados eram a única
        informação concreta de crime no prompt inteiro — e o modelo passava a
        tratá-los como o caso real, improvisando em cima deles. Com fatos de
        verdade no prompt, os exemplos voltam a fazer só o trabalho deles:
        ensinar o formato do raciocínio, sem competir com o caso.
        """
        return """
Exemplo A - Contradição com evidência:
Uma evidência em suas mãos afirma X. O suspeito afirma algo incompatível com X.
Análise: contradição direta. detectou_mentira = true, suspeita sobe muito.

Exemplo B - Contradição com ele mesmo:
O suspeito afirmou antes que estava sozinho. Agora afirma que havia alguém com ele.
Análise: a história mudou. detectou_mentira = true, suspeita sobe.

Exemplo C - Lacuna preenchida de forma plausível:
O suspeito dá um detalhe verificável que explica um período em aberto e não colide
com nenhuma evidência.
Análise: não é mentira. detectou_mentira = false, suspeita desce.

Exemplo D - Evasiva:
O suspeito desconversa ou se recusa a responder.
Análise: não é mentira, mas não colabora. detectou_mentira = false, suspeita sobe pouco.
"""

    # Um exemplo só, para modelos pequenos rodando em CPU: cada token de entrada
    # custa tempo antes da primeira palavra sair. Ele mostra os DOIS desfechos:
    # a primeira versão só mostrava o de mentira, e o Qwen 2.5 3B marcou
    # detectou_mentira = true em 4 de 4 turnos, inclusive nos sem contradição.
    EXEMPLO_COMPACTO = """
Exemplo: uma evidência afirma X.
- Suspeito afirma algo incompatível com X → detectou_mentira = true, suspeita sobe muito.
- Suspeito diz algo que não colide com nenhuma evidência nem com o que já disse
  (mesmo sem álibi, mesmo desviando) → detectou_mentira = false.
"""

    # ─── Blocos do prompt montados a partir do caso ──────────────────────────

    def _bloco_identidade(self, caso: Dict[str, Any]) -> str:
        papel = caso.get('papel_do_jogador', {})
        vitima = caso.get('vitima', {})
        contexto = caso.get('contexto_geral')
        return f"""CASO: {caso.get('titulo', 'Sem título')}
{('CONTEXTO: ' + contexto) if contexto else ''}
VÍTIMA: {vitima.get('nome', 'Desconhecida')}, {vitima.get('idade', '?')} anos — {vitima.get('funcao', '')}

QUEM ESTÁ NA SUA FRENTE:
- Nome: {papel.get('nome', 'Desconhecido')}
- Função: {papel.get('funcao', '')}
- Situação: {papel.get('situacao', '')}
- {papel.get('instrucao_ao_detetive', '')}"""

    def _bloco_verdade(self, caso: Dict[str, Any]) -> str:
        """
        A verdade do caso. Fica no prompt para o modelo conseguir JULGAR as
        falas, com proibição explícita de revelá-la — é o que separa um detetive
        que percebe a mentira de um que só desconfia de tudo.
        """
        verdade = caso.get('verdade', {})
        linha = "\n".join(
            f"  {item.get('hora', '??')} — {item.get('fato', '')}"
            for item in verdade.get('linha_do_tempo', [])
        )
        culpado_e_o_jogador = verdade.get('jogador_e_culpado')

        return f"""=== VERDADE DO CASO (CONFIDENCIAL — NUNCA REVELE) ===
Esta seção existe para você AVALIAR o que o suspeito diz. Você NÃO pode citá-la,
insinuá-la nem deixá-la transparecer. Só pode mencionar o que estiver na seção
de EVIDÊNCIAS. Se o suspeito perguntar como você sabe de algo, cite a evidência.

O que aconteceu: {verdade.get('o_que_aconteceu', 'Desconhecido')}
Culpado real: {verdade.get('culpado', 'Desconhecido')}
O suspeito à sua frente é o culpado? {'SIM' if culpado_e_o_jogador else 'NÃO'}

Motivações: {json.dumps(verdade.get('motivacoes', {}), ensure_ascii=False)}
Segredos (não revele antes da hora): {'; '.join(verdade.get('segredos_que_nao_devem_ser_revelados_antes_da_hora', [])) or '(nenhum)'}
O que o suspeito de fato viveu (use só para JULGAR; não presuma que ele disse algo que não disse): {'; '.join(caso.get('conhecimento_do_suspeito', {}).get('sabe', [])) or '(não informado)'}
Linha do tempo real:
{linha or '  (não informada)'}"""

    def _bloco_evidencias(self, caso: Dict[str, Any]) -> str:
        def _linha(e):
            # Texto FIXO (não depende do turno atual): mantém o prefixo do
            # prompt estável para o KV cache. Quem diz o que já pode ser citado
            # neste turno é o bloco dinâmico ESTADO DA INVESTIGAÇÃO.
            turno = int(e.get('a_partir_do_turno', 1))
            quando = f" (só cite a partir do turno {turno})" if turno > 1 else ""
            return f"  - [{e.get('fonte', '?')}] {e.get('descricao', '')}{quando}"

        itens = "\n".join(_linha(e) for e in caso.get('evidencias_conhecidas', []))
        lacunas = "\n".join(f"  - {l}" for l in caso.get('lacunas', []))

        return f"""=== EVIDÊNCIAS EM SUAS MÃOS (pode citar) ===
{itens or '  (nenhuma)'}

=== O QUE VOCÊ AINDA NÃO SABE (arranque isto dele) ===
{lacunas or '  (nada)'}"""

    def _bloco_suspeita(self, caso: Dict[str, Any]) -> str:
        regras = caso.get('regras_de_suspeita', {})
        linhas = "\n".join(
            f"  - {chave.replace('_', ' ')}: {valor}"
            for chave, valor in regras.items()
            if chave != 'observacao'
        )
        obs = regras.get('observacao', '')

        return f"""=== COMO MEXER NO NÍVEL DE SUSPEITA ===
Parta do NÍVEL DE SUSPEITA ATUAL (informado junto da fala do jogador) e ajuste:
{linhas or '  (sem regras definidas)'}
{f'  ATENÇÃO: {obs}' if obs else ''}"""

    def _instrucoes(self, incluir_verdade: bool) -> str:
        # Sem a seção de verdade no prompt, mandar "compare com a VERDADE DO
        # CASO" é pedir ao modelo pequeno que compare com algo que não existe
        # — e ele inventa.
        if incluir_verdade:
            comparar = "Compare a fala do suspeito com as EVIDÊNCIAS e com a VERDADE DO CASO"
            nao_revelar = "Nunca revele a VERDADE DO CASO; cite apenas evidências"
        else:
            comparar = "Compare a fala do suspeito com as EVIDÊNCIAS"
            nao_revelar = "Não afirme saber o que não está nas evidências"

        return f"""INSTRUÇÕES:
1. {comparar}
2. Compare também com o que ele mesmo já disse nesta sessão
3. Só marque detectou_mentira quando houver colisão real — não ter álibi não é mentira
4. Pressione as lacunas: peça horário, local e quem pode confirmar
5. {nao_revelar}
6. Responda à FALA DO JOGADOR atual com frases novas; não repita falas suas nem cumprimente de novo
6b. Avalie você mesmo o TOM da fala. Se o suspeito for agressivo, ofensivo, debochado ou provocador, reaja em personagem — irritação contida e autoridade firme, em uma frase curta que mostre que você notou — antes de seguir com a pergunta. Se for cooperativo ou nervoso, ajuste o tom de acordo. Nunca responda como se a fala fosse neutra quando não for
7. Fale pouco: no máximo 2 ou 3 frases curtas, terminando em UMA pergunta
8. Retorne APENAS um objeto JSON válido, sem blocos de código markdown

{self.REGRA_DO_TEMPO}"""

    # Texto FIXO sobre o tempo de resposta. Fica nas instruções (parte fixa do
    # prompt, que o llama.cpp reaproveita entre turnos) e não no bloco de tempo
    # do turno: repetido ali, eram ~150 palavras reprocessadas a cada resposta.
    # O bloco do turno traz só os valores medidos.
    REGRA_DO_TEMPO = (
        "SOBRE O TEMPO MEDIDO PELO SISTEMA (quando o bloco de tempo aparecer): o nível de tempo de "
        "resposta representa exclusivamente quanto tempo o jogador levou para responder à pergunta "
        "atual. Ele não representa, por si só, culpa, mentira, confissão, nervosismo ou contradição. "
        "O conteúdo da resposta, sua coerência com as evidências, as contradições e as informações "
        "verificáveis continuam sendo os principais fatores para avaliar a suspeita. Use a demora "
        "apenas como um sinal contextual secundário e fraco. Não aumente a suspeita automaticamente "
        "apenas porque o tempo de resposta foi médio ou alto, nem trate a demora como evidência "
        "factual de culpa. O sistema já aplica sozinho um pequeno ajuste pelo tempo: NÃO o inclua no "
        "nível_suspeita que você propõe. Se o nível for alto, você pode apenas deixar o tom um pouco "
        "mais pressionador, sem inventar fatos."
    )

    def _historico_compacto(self, session_id: str, player_text: str) -> str:
        """Diálogo enxuto para modelos pequenos (ver ContextMemoryManager.get_dialogo)."""
        falas = self.memory.get_dialogo(session_id, max_falas=12)
        # A rota grava a fala atual antes de chamar o orquestrador; ela já vai
        # em FALA DO JOGADOR, logo abaixo.
        if falas and falas[-1] == ("SUSPEITO", player_text):
            falas = falas[:-1]
        if not falas:
            return "=== HISTÓRICO DA SESSÃO ===\nPrimeiro turno: nada foi dito ainda."

        # Só as declarações do suspeito: é contra elas que se checa contradição.
        # As falas do detetive ficam de fora de propósito — medido no Qwen 2.5
        # 3B, qualquer fala dele citada no prompt (até com "não repita" do
        # lado) voltava copiada palavra por palavra em vários turnos.
        declaracoes = [t for papel, t in falas if papel == "SUSPEITO"][-5:]
        if not declaracoes:
            return "=== HISTÓRICO DA SESSÃO ===\nPrimeiro turno: nada foi dito ainda."

        linhas = "\n".join(f'- "{t}"' for t in declaracoes)
        return ("=== HISTÓRICO DA SESSÃO ===\n"
                "Você já interrogou o suspeito nesta sessão (não o cumprimente de novo).\n"
                f"O que ele já declarou:\n{linhas}")

    def _suspeita_atual(self, session_id: str,
                        estado: Optional[gr.EstadoPartida] = None) -> str:
        # Fica DEPOIS do histórico, nunca no bloco de regras: muda a cada turno
        # e, no prefixo, invalidaria o KV cache do runtime local.
        if estado is not None:
            if estado.turno == 0:
                return "NÍVEL DE SUSPEITA ATUAL: 0 (início do interrogatório)"
            return f"NÍVEL DE SUSPEITA ATUAL: {estado.suspeita}"
        recentes = self.memory.get_recent_suspicions(session_id, n=1)
        if not recentes:
            return "NÍVEL DE SUSPEITA ATUAL: 0 (início do interrogatório)"
        return f"NÍVEL DE SUSPEITA ATUAL: {recentes[-1]}"

    # ─── Blocos DINAMICOS (depois de "FALA DO JOGADOR") ──────────────────────
    # Mudam a cada turno, entao ficam no fim: o prefixo fixo do prompt continua
    # identico entre turnos e o KV cache do llama.cpp segue valendo.

    def _bloco_analise(self, interp: Interpretacao, fatos: Dict[str, Any],
                       temas: List[Tema], estado: gr.EstadoPartida) -> str:
        por_id = {t.id: t for t in temas}
        instrucao = {
            CONFISSAO: ("O suspeito AFIRMOU QUE COMETEU O CRIME (confissão — ainda NÃO comprovada). "
                        "Isto NÃO é uma negação. Reaja diretamente a ela. NÃO volte a perguntar "
                        "horário, álibi ou paradeiro como se ele tivesse negado. Peça detalhes que "
                        "só o autor saberia (como, com o quê, onde a vítima estava, o que fez depois) "
                        "e confronte-os com as evidências."),
            NEGACAO: ("O suspeito NEGOU a autoria. Reconheça a negação e responda a ela com a "
                      "evidência ou a lacuna mais relevante."),
            EVASIVA: ("O suspeito se esquivou ou disse que não sabe. Isso NÃO é uma resposta e NÃO é "
                      "mentira. Insista no máximo uma vez, de outro jeito, ou passe ao próximo ponto."),
            AMBIGUA: ("A fala é vaga, curta ou hipotética. Peça esclarecimento direto; não invente o "
                      "que ele quis dizer."),
            INFORMACAO: ("O suspeito trouxe informação. Use-a: compare com as evidências e com o que "
                         "ele já disse antes de seguir."),
        }[interp.tipo]
        linhas = [f"=== ANÁLISE DA ÚLTIMA FALA (feita pelo sistema; confie nela) ===",
                  f"Tipo: {interp.tipo.upper()}. {instrucao}"]
        if fatos.get("mudou_versao"):
            linhas.append("ATENÇÃO: ele MUDOU DE VERSÃO sobre a autoria (antes disse o contrário). "
                          "É uma contradição com o que ele mesmo declarou; explore-a.")
        pendente = por_id.get(estado.tema_pergunta_atual) if estado.tema_pergunta_atual else None
        if pendente is not None:
            linhas.append(
                f"Sua pergunta anterior era sobre \"{pendente.descricao}\": "
                + ("a fala a RESPONDE; não a repita." if fatos.get("respondeu_pendente")
                   else "a fala NÃO a respondeu."))
        return "\n".join(linhas)

    def _bloco_estado(self, estado: gr.EstadoPartida, temas: List[Tema],
                      caso: Optional[Dict[str, Any]]) -> str:
        por_id = {t.id: t for t in temas}
        esclarecidos, sem_resposta, abertos = [], [], []
        for t in temas:
            info = estado.temas.get(t.id)
            if info and info["respondido"]:
                esclarecidos.append(f'  - {t.descricao}: "{info["resposta"]}"')
            elif info and info["perguntado"] > 0:
                sem_resposta.append(f"  - {t.descricao}")
            else:
                abertos.append(f"  - {t.descricao}")
        proximo = self.partida.proximo_tema(estado, temas)

        turno = estado.turno + 1
        liberadas = [e.get('id') for e in (caso or {}).get('evidencias_conhecidas', [])
                     if int(e.get('a_partir_do_turno', 1)) <= turno]
        linhas = [
            "=== ESTADO DA INVESTIGAÇÃO ===",
            f"Este é o turno {turno} de no máximo {gr.REGRAS.MAX_TURNOS}. Suspeita atual: {estado.suspeita}.",
            "Já esclarecido (NÃO pergunte de novo, a menos que ele traga algo novo sobre isso):",
            "\n".join(esclarecidos) or "  (nada ainda)",
            "Perguntado e ainda sem resposta (pode insistir, de outro jeito):",
            "\n".join(sem_resposta) or "  (nada)",
            "Ainda não abordado:",
            "\n".join(abertos) or "  (nada)",
        ]
        if proximo is not None:
            linhas.append(f"Foco sugerido para esta resposta: {proximo.descricao}.")
        if liberadas:
            linhas.append("Evidências que você já pode citar neste turno: " + ", ".join(liberadas) + ".")
        return "\n".join(linhas)

    def build_prompt(self, session_id: str, player_text: str,
                     caso: Optional[Dict[str, Any]] = None,
                     incluir_verdade: bool = True,
                     compacto: bool = False,
                     estado: Optional[gr.EstadoPartida] = None,
                     interp: Optional[Interpretacao] = None,
                     fatos: Optional[Dict[str, Any]] = None,
                     diretiva: str = "",
                     tempo_resposta_s: Optional[float] = None) -> str:
        """
        `incluir_verdade=False` omite a seção confidencial do caso.

        Existe porque modelos pequenos não conseguem sustentar a proibição de
        não revelá-la: medido no Llama 3.2 3B local, com o prompt real, a
        verdade vazou em 5 de 8 respostas — uma delas dizendo ao suspeito o
        nome do culpado. Quem decide é a capability do provedor
        (LLMProvider.suporta_contexto_confidencial), não uma configuração à
        parte, para não existir combinação de env que mande a verdade para um
        modelo que a repete.

        `compacto=True` troca os quatro exemplos por um só, para modelos
        pequenos em CPU, onde o tamanho da entrada domina o tempo de resposta.

        `compacto=True` também troca o histórico pelo diálogo enxuto
        (_historico_compacto).

        Ordem dos blocos: tudo que é fixo dentro de uma partida (caso, regras,
        exemplos, instruções, formato JSON) vem ANTES do histórico, que muda a
        cada turno. Os
        runtimes locais (llama.cpp e Ollama) reaproveitam o processamento do
        maior prefixo igual ao da chamada anterior — com o histórico no meio,
        tudo depois dele era reprocessado em todo turno.
        """
        if compacto:
            contexto = self._historico_compacto(session_id, player_text)
            exemplos = self.EXEMPLO_COMPACTO
        else:
            contexto = self.memory.get_prompt_context(session_id)
            exemplos = self.exemplos

        if caso is None:
            # Sem caso carregado o jogo continua, mas sem base factual — que é
            # exatamente o estado anterior a esta mudança. Loga alto porque é
            # degradação silenciosa: a partida "funciona" e a detecção não.
            logger.error("Nenhum caso disponível: o detetive vai interrogar sem fatos.")
            return self._build_prompt_sem_caso(contexto, player_text, exemplos)

        if incluir_verdade:
            bloco_verdade = self._bloco_verdade(caso)
        else:
            bloco_verdade = (
                "=== VOCÊ NÃO TEM A SOLUÇÃO DO CASO ===\n"
                "Você só dispõe das evidências abaixo. Detecte contradição apenas\n"
                "contra elas ou contra o que o próprio suspeito já disse nesta sessão.\n"
                "Não afirme saber onde ele esteve nem quem é o culpado."
            )

        return f"""Você é um detetive experiente conduzindo um interrogatório.

{self._bloco_identidade(caso)}

{bloco_verdade}

{self._bloco_evidencias(caso)}

{self._bloco_suspeita(caso)}

EXEMPLOS DE ANÁLISE:
{exemplos}

{self._instrucoes(incluir_verdade)}

Responda com JSON puro (sem ```json):
{{
  "texto_detetive": "<escreva aqui a sua fala, em segunda pessoa>",
  "status_investigacao": {{
    "nivel_suspeita": 0-100,
    "detectou_mentira": true/false
  }},
  "feedback_visual": {{
    "cor_iluminacao": "#HEXCODE",
    "bpm_musica": 60-140,
    "animacao_trigger": "NomeDaAnimacao"
  }}
}}

{contexto}

{self._suspeita_atual(session_id, estado)}
FALA DO JOGADOR: "{player_text}"
{self._sufixo_dinamico(estado, interp, fatos, caso, diretiva, tempo_resposta_s)}"""

    def _bloco_tempo(self, tempo_resposta_s: Optional[float]) -> str:
        """
        Contexto temporal da resposta atual, classificado pelo BACKEND
        (game_rules.classificar_tempo) — o modelo nunca calcula nem recebe do
        cliente. E um sinal contextual fraco; o efeito numerico e do sistema.
        """
        nivel = gr.classificar_tempo(tempo_resposta_s)
        if nivel is None:
            return ""
        r = gr.REGRAS
        return f"""=== TEMPO DE RESPOSTA (medido pelo sistema) ===
tempo_resposta_segundos: {tempo_resposta_s:.1f}
nivel_tempo_resposta: {nivel}  (baixo: menos de {r.TEMPO_MEDIO_A_PARTIR_S:.0f}s; médio: de {r.TEMPO_MEDIO_A_PARTIR_S:.0f}s a menos de {r.TEMPO_ALTO_A_PARTIR_S:.0f}s; alto: de {r.TEMPO_ALTO_A_PARTIR_S:.0f}s a {r.PRAZO_RESPOSTA_S:.0f}s)
status_tempo_resposta: dentro_do_prazo
(Como usar este sinal: ver SOBRE O TEMPO MEDIDO PELO SISTEMA, nas instruções.)"""

    def _sufixo_dinamico(self, estado, interp, fatos, caso, diretiva: str,
                         tempo_resposta_s: Optional[float] = None) -> str:
        if estado is None or interp is None:
            return ""
        temas = carregar_temas(caso)
        partes = [self._bloco_analise(interp, fatos or {}, temas, estado),
                  self._bloco_estado(estado, temas, caso)]
        bloco_tempo = self._bloco_tempo(tempo_resposta_s)
        if bloco_tempo:
            partes.append(bloco_tempo)
        if diretiva:
            partes.append(f"CORREÇÃO OBRIGATÓRIA: {diretiva}")
        return "\n" + "\n\n".join(partes) + "\n"

    def _build_prompt_sem_caso(self, contexto: str, player_text: str,
                               exemplos: str) -> str:
        """Degradação para quando a pasta de casos está vazia ou corrompida."""
        return f"""Você é um detetive experiente conduzindo um interrogatório.
Não há fatos do caso disponíveis: conduza o interrogatório pedindo detalhes
verificáveis e NÃO acuse o suspeito de mentir, porque você não tem contra o que
comparar.

EXEMPLOS DE ANÁLISE:
{exemplos}

{contexto}

FALA DO JOGADOR: "{player_text}"

Responda com JSON puro (sem ```json):
{{
  "texto_detetive": "<escreva aqui a sua fala, em segunda pessoa>",
  "status_investigacao": {{
    "nivel_suspeita": 0-100,
    "detectou_mentira": false
  }},
  "feedback_visual": {{
    "cor_iluminacao": "#HEXCODE",
    "bpm_musica": 60-140,
    "animacao_trigger": "NomeDaAnimacao"
  }}
}}
"""

    # Sessao que nunca existe na memoria: o prompt dela e identico ao de um 1o
    # turno real ate a fala do jogador — e esse prefixo que o aquecimento cacheia.
    SESSAO_AQUECIMENTO = "__aquecimento__"

    async def aquecer(self, provider: LLMProvider, case_id: Optional[str] = None) -> None:
        """Adianta carga e cache do provedor (ver LlamaCppProvider.aquecer)."""
        prompt = self.build_prompt(
            self.SESSAO_AQUECIMENTO, "", get_case(case_id),
            incluir_verdade=provider.suporta_contexto_confidencial,
            compacto=not provider.suporta_contexto_confidencial,
        )
        await provider.aquecer(prompt)

    @staticmethod
    def _limitar_faixas(resultado: Dict[str, Any]) -> None:
        """
        Prende os numeros na faixa que o jogo entende. O prompt pede 0-100 e
        60-140, mas nada impede o modelo de mandar 150 — o ResponseContract so
        exige int, e a gramatica do llama.cpp ignora minimum/maximum. Fora da
        faixa, a barra de suspeita da HUD estourava.
        """
        status = resultado.get('status_investigacao', {})
        visual = resultado.get('feedback_visual', {})
        try:
            status['nivel_suspeita'] = max(0, min(100, int(status.get('nivel_suspeita', 0))))
        except (TypeError, ValueError):
            status['nivel_suspeita'] = 0
        try:
            visual['bpm_musica'] = max(60, min(140, int(visual.get('bpm_musica', 90))))
        except (TypeError, ValueError):
            visual['bpm_musica'] = 90

    async def analyze(self, session_id: str, player_text: str,
                      provider: LLMProvider,
                      case_id: Optional[str] = None,
                      interacao_id: Optional[str] = None,
                      chegada: Optional[float] = None) -> Dict[str, Any]:
        """
        Roda um turno de interrogatorio.

        Recebe o provedor por parametro em vez de guardar um no construtor: assim
        a regra de jogo nao sabe qual modelo esta atendendo, e testar nao exige
        mexer em variavel de ambiente — basta passar um MockProvider.

        LLMUnavailableError NAO e capturado aqui de proposito: servico fora do ar
        e problema de infraestrutura, e a rota traduz isso em HTTP 503. Engolir
        aqui devolveria um mock silencioso e o jogador nunca saberia que o
        modelo esta desligado. Quando isso acontece NADA da partida muda (o
        estado so e gravado no fim) e a janela de resposta e reaberta inteira:
        a falha de infraestrutura nao pode custar tempo ao jogador.

        `chegada` e o instante em que a requisicao entrou na API — medido ANTES
        da fila do lock e da IA, para que o tempo de processamento nao seja
        confundido com o tempo que o jogador levou para responder.
        """
        chegada = self.partida.agora() if chegada is None else chegada
        async with self._lock(session_id):
            return await self._analisar(session_id, player_text, provider,
                                        case_id, interacao_id, chegada)

    async def _analisar(self, session_id: str, player_text: str,
                        provider: LLMProvider, case_id: Optional[str],
                        interacao_id: Optional[str], chegada: float) -> Dict[str, Any]:
        store = self.partida.store
        estado = store.carregar(session_id, case_id)

        # Reenvio da mesma interacao: devolve o que ja foi entregue, sem efeito.
        if interacao_id and interacao_id in estado.processadas:
            logger.info(f"Interacao {interacao_id} ja processada; devolvendo a mesma resposta.")
            return copy.deepcopy(estado.processadas[interacao_id])

        # Partida encerrada: nada muda (nem historico, nem contadores).
        if estado.encerrada:
            return self._resposta_de_partida_encerrada(estado)

        caso = get_case(case_id or estado.case_id)
        temas = carregar_temas(caso)
        if estado.case_id is None and caso:
            estado.case_id = caso.get("case_id")
        self._semear_abertura(estado, caso)

        # Resposta que chegou depois do prazo: vale o timeout, nao o texto.
        if self.partida.janela_vencida(estado, interacao_id, chegada):
            logger.info(f"Resposta tardia na interacao {interacao_id}; aplicando timeout.")
            return self._aplicar_timeout(estado, caso, temas)

        tempo = self.partida.tempo_de_resposta(estado, interacao_id, chegada)
        janela_original = copy.deepcopy(estado.janela)
        confessou_antes = estado.confessou

        # Trabalha numa COPIA: so vira estado oficial se o turno terminar bem.
        trabalho = copy.deepcopy(estado)
        interp = classificar(player_text, temas, pergunta_pendente=True)
        fatos = self.partida.registrar_fala_do_jogador(trabalho, interp, player_text)
        if trabalho.janela and trabalho.janela["id"] == interacao_id:
            trabalho.janela["estado"] = "respondida"

        try:
            resultado = await self._gerar_fala(session_id, player_text, provider,
                                               caso, temas, trabalho, interp, fatos, tempo)
        except Exception:
            # Falha de infraestrutura: reabre a janela com o prazo inteiro.
            if janela_original and janela_original["id"] == interacao_id:
                agora = self.partida.agora()
                janela_original["aberta_em"] = agora
                janela_original["prazo_em"] = agora + gr.REGRAS.PRAZO_RESPOSTA_S
                estado.janela = janela_original
                try:
                    store.salvar(estado)
                except OSError as exc:
                    logger.error(f"Nao foi possivel reabrir a janela: {exc}")
            raise

        penalidade = gr.penalidade_por_tempo(tempo) if tempo is not None else 0
        piso = 0
        if interp.tipo == CONFISSAO:
            piso = (gr.REGRAS.PISO_CONFISSAO_REPETIDA if confessou_antes
                    else gr.REGRAS.PISO_PRIMEIRA_CONFISSAO)

        status = resultado.setdefault("status_investigacao", {})
        if fatos["mudou_versao"]:
            # Contradicao com a propria fala anterior: fato do sistema, nao
            # opiniao do modelo.
            status["detectou_mentira"] = True

        resposta = self.partida.montar_resposta(
            trabalho, resultado, penalidade_tempo=penalidade,
            tempo_resposta_s=tempo, interpretacao=interp, piso_suspeita=piso,
        )
        resposta["interacao_id"] = interacao_id or ""
        self.partida.registrar_pergunta_do_detetive(trabalho, resposta["texto_detetive"], temas)
        self.partida.guardar_entrega(trabalho, interacao_id or "", resposta)

        store.salvar(trabalho)
        self.turn_counter[session_id] = trabalho.turno
        self._gravar_historico(session_id, player_text, resposta, trabalho)
        return resposta

    async def _gerar_fala(self, session_id: str, player_text: str, provider: LLMProvider,
                          caso: Optional[Dict[str, Any]], temas: List[Tema],
                          trabalho: gr.EstadoPartida, interp: Interpretacao,
                          fatos: Dict[str, Any],
                          tempo: Optional[float] = None) -> Dict[str, Any]:
        """
        Pede a fala ao modelo e a valida: contrato, repeticao literal e
        repeticao de TEMA ja esclarecido. Devolve o dicionario do contrato com
        a suspeita ainda como o modelo propôs (quem fecha o turno e o PartidaService).
        """
        # A capability do provedor decide se a verdade do caso entra no prompt.
        # Quem não aguenta a verdade é modelo pequeno, que também ganha o
        # prompt compacto: uma flag só, para não existir combinação incoerente.
        incluir_verdade = provider.suporta_contexto_confidencial

        def montar(diretiva: str = "") -> str:
            return self.build_prompt(
                session_id, player_text, caso,
                incluir_verdade=incluir_verdade, compacto=not incluir_verdade,
                estado=trabalho, interp=interp, fatos=fatos, diretiva=diretiva,
                tempo_resposta_s=tempo,
            )

        prompt = montar()
        tentativas = 0
        resultado = None
        repetida = None          # fala identica a uma anterior (ainda serve em ultimo caso)
        tema_repetido = None     # fala valida, mas pergunta de novo um tema esclarecido
        corrigido_por_tema = False

        # tenta gerar e faz retry se o json vier fora do contrato
        while tentativas <= 2:
            try:
                dados = await provider.generate(prompt)
            except LLMGenerationError as exc:
                logger.warning(f"Falha recuperavel do provedor {provider.nome}: {exc}")
                tentativas += 1
                await asyncio.sleep(0.1)
                continue

            try:
                # id_turno e provisorio; o definitivo vem do estado da partida
                dados['id_turno'] = trabalho.turno + 1
                self._completar_campos_do_sistema(dados)
                ResponseContract.model_validate(dados)
            except Exception:
                tentativas += 1
                await asyncio.sleep(0.1)
                continue

            texto = self._sem_metadados(dados['texto_detetive'])
            dados['texto_detetive'] = texto

            if self._fala_e_exemplo(texto):
                # Visto na validacao com o Qwen 2.5 3B: a fala veio literalmente
                # "<escreva aqui a sua fala, em segunda pessoa>", copiada do
                # modelo de JSON do prompt. Contrato valido, mas o jogador veria
                # o placeholder na prancheta.
                logger.warning(f"Provedor {provider.nome} copiou o exemplo do prompt; tentando de novo.")
                tentativas += 1
                continue

            repetidos = self.partida.temas_repetidos(trabalho, texto, temas, fatos)
            if repetidos:
                # Pergunta de novo algo ja esclarecido (ou ja insistido demais).
                # Uma nova tentativa, com a correcao explicita; se o modelo
                # insistir, a fala e substituida por uma deterministica.
                por_id = {t.id: t for t in temas}
                nomes = ", ".join(por_id[i].descricao for i in repetidos)
                logger.warning(f"Provedor {provider.nome} repetiu tema ja esclarecido: {nomes}")
                tema_repetido = dados
                if not corrigido_por_tema:
                    corrigido_por_tema = True
                    proximo = self.partida.proximo_tema(trabalho, temas)
                    alvo = (f" Em vez disso, avance para: {proximo.descricao}." if proximo
                            else " Em vez disso, reaja ao que ele disse ou peça um detalhe novo.")
                    prompt = montar(f"NÃO pergunte de novo sobre: {nomes} (já esclarecido).{alvo}")
                    tentativas += 1
                    continue
                break

            if repetida is None and self._repete_fala_anterior(session_id, texto):
                # Contrato ok, mas o jogador veria a mesma fala de novo. Uma
                # nova tentativa só: em CPU cada uma custa ~20s, e o ApiClient
                # do Unity desiste em 90s. Se repetir de novo, a fala repetida
                # ainda é melhor que o fallback genérico.
                logger.warning(f"Provedor {provider.nome} repetiu uma fala anterior; tentando de novo.")
                repetida = dados
                tentativas += 1
                continue

            resultado = dados
            break

        if resultado is None and tema_repetido is not None:
            # Mantem a leitura de suspeita do modelo, troca so a fala.
            resultado = tema_repetido
            resultado['texto_detetive'] = self._fala_deterministica(caso, temas, trabalho, interp, fatos)
        resultado = resultado or repetida

        # se der erro em todas as tentativas, usa a resposta fallback
        if not resultado:
            resultado = self._fallback_response(trabalho, caso, temas, interp, fatos)

        if trabalho.turno >= 1:
            resultado['texto_detetive'] = self._sem_saudacao(resultado.get('texto_detetive', ''))
        self._limitar_faixas(resultado)
        return resultado

    # Termos da análise INTERNA que o modelo às vezes escreve dentro da fala.
    _METADADOS = re.compile(
        r"detectou[_ ]?mentira|n[íi]vel[_ ]?(de[_ ])?suspeita|status_investigacao|"
        r"congelar_input|fim_de_jogo|feedback_visual|cor_iluminacao|bpm_musica|"
        r"animacao_trigger|id_turno|suspeita\s+(sobe|desce|aumenta|diminui|cai)",
        re.IGNORECASE)
    _FRASES = re.compile(r"[^.!?]+[.!?]*")

    @classmethod
    def _sem_metadados(cls, texto: str) -> str:
        """
        Tira da fala as frases com a análise interna. Visto em jogo com o Qwen
        2.5 3B: a fala terminava em "Detectou_mentira: true, suspeita sobe +25."
        e o jogador lia isso na prancheta. Sobra só o que o detetive diria;
        se não sobrar nada, _fala_e_exemplo recusa e a fala é refeita.
        """
        if not texto or not cls._METADADOS.search(texto):
            return texto
        frases = [f.strip() for f in cls._FRASES.findall(texto) if f.strip()]
        return " ".join(f for f in frases if not cls._METADADOS.search(f))

    @staticmethod
    def _fala_e_exemplo(texto: str) -> bool:
        """Fala vazia ou copiada do modelo de JSON do prompt (placeholder)."""
        limpo = (texto or "").strip()
        return (not limpo
                or "escreva aqui a sua fala" in limpo.lower()
                or (limpo.startswith("<") and limpo.endswith(">")))

    @staticmethod
    def _completar_campos_do_sistema(dados: Dict[str, Any]) -> None:
        """
        congelar_input e fim_de_jogo sao decididos pelo SISTEMA (PartidaService)
        e id_turno pelo estado da partida: o modelo nao os gera mais. Eram ~25
        tokens por resposta escritos so para serem sobrescritos — no Qwen em CPU,
        segundos por turno. Aqui entram com valor neutro so para o contrato
        validar; o valor final continua vindo do sistema.
        """
        status = dados.get('status_investigacao')
        if isinstance(status, dict):
            status.setdefault('congelar_input', False)
            status.setdefault('fim_de_jogo', False)

    _SAUDACAO = re.compile(
        r"^\s*(boa\s+(noite|tarde)|bom\s+dia|ol[áa])\b[\s,.!]*"
        r"((senhor|sr\.?)\s+\w+|daniel)?[\s,.!]*", re.IGNORECASE)

    @classmethod
    def _sem_saudacao(cls, texto: str) -> str:
        """
        "Boa noite, senhor Reis." no começo de TODA fala, apesar da instrução:
        visto em 2 de 3 turnos nas medições com o Qwen. A partir do 2º turno o
        cumprimento sai — só se sobrar uma fala de verdade depois dele.
        """
        resto = cls._SAUDACAO.sub("", texto, count=1)
        if resto == texto or len(resto.strip()) < 15:
            return texto
        resto = resto.strip()
        return resto[0].upper() + resto[1:]

    def _fala_deterministica(self, caso: Optional[Dict[str, Any]], temas: List[Tema],
                             estado: gr.EstadoPartida, interp: Interpretacao,
                             fatos: Dict[str, Any]) -> str:
        """
        Fala montada sem modelo: reage ao TIPO da declaracao e segue para o
        proximo ponto em aberto. Garante progresso quando o modelo trava num
        tema ja esclarecido. So usa fatos do proprio caso.
        """
        nome = "senhor"
        if caso:
            sobrenome = caso.get("papel_do_jogador", {}).get("nome", "").split()[-1:]
            if sobrenome:
                nome = f"senhor {sobrenome[0]}"

        reacao = {
            CONFISSAO: (f"O {nome} está dizendo que matou Otávio Lisboa? Então me conte como foi: "
                        "com o quê, onde na gráfica e o que fez logo depois."
                        if caso else "O senhor está dizendo que cometeu o crime? Então me conte como foi."),
            NEGACAO: f"Entendo que o {nome} nega. Mas preciso que isso se sustente.",
            EVASIVA: f"Esquivar-se não ajuda, {nome}.",
            AMBIGUA: f"Não ficou claro, {nome}. Seja direto.",
            INFORMACAO: "Anotado.",
        }[interp.tipo]
        if interp.tipo == CONFISSAO:
            return reacao

        proximo = self.partida.proximo_tema(estado, temas)
        if proximo is not None and proximo.pergunta_modelo:
            return f"{reacao} {proximo.pergunta_modelo}"
        return f"{reacao} Há mais alguma coisa que o senhor queira acrescentar ao seu depoimento?"

    # ─── Timer ───────────────────────────────────────────────────────────────

    async def abrir_janela(self, session_id: str, case_id: Optional[str] = None) -> Dict[str, Any]:
        """O jogador acionou 'responder': comeca a contar os 60 s (relogio do servidor)."""
        async with self._lock(session_id):
            estado = self.partida.store.carregar(session_id, case_id)
            self._semear_abertura(estado, get_case(case_id or estado.case_id))
            j = estado.janela
            if (j and j["estado"] == "aberta"
                    and self.partida.agora() > j["prazo_em"] + gr.REGRAS.TOLERANCIA_RESPOSTA_S):
                # O cliente sumiu sem avisar o vencimento: o prazo estourou de
                # verdade, entao o timeout e aplicado agora (uma vez) em vez de
                # a janela ser descartada de graça.
                caso = get_case(case_id or estado.case_id)
                self._aplicar_timeout(estado, caso, carregar_temas(caso))
            janela = self.partida.abrir_janela(estado)   # JanelaInvalida se encerrada
            self.partida.store.salvar(estado)
            return janela

    async def expirar_janela(self, session_id: str, interacao_id: str,
                             case_id: Optional[str] = None) -> Dict[str, Any]:
        """
        O prazo acabou sem resposta. Idempotente e imune a corrida com a
        resposta: se a interacao ja foi resolvida (respondida ou expirada), devolve
        o que ja foi entregue e `aplicado=False`.
        """
        async with self._lock(session_id):
            estado = self.partida.store.carregar(session_id, case_id)

            if interacao_id in estado.processadas:
                return {"aplicado": False,
                        "resposta": copy.deepcopy(estado.processadas[interacao_id])}
            if estado.encerrada:
                return {"aplicado": False, "resposta": self._resposta_de_partida_encerrada(estado)}

            self.partida.validar_pedido_de_timeout(estado, interacao_id)  # JanelaInvalida

            if estado.janela["estado"] != "aberta":
                return {"aplicado": False,
                        "resposta": self._resposta_de_partida_encerrada(estado)}

            caso = get_case(case_id or estado.case_id)
            return {"aplicado": True,
                    "resposta": self._aplicar_timeout(estado, caso, carregar_temas(caso))}

    def _aplicar_timeout(self, estado: gr.EstadoPartida, caso: Optional[Dict[str, Any]],
                         temas: List[Tema]) -> Dict[str, Any]:
        """Fecha a janela como expirada, UMA vez, sem chamar o modelo."""
        self._semear_abertura(estado, caso)
        indice, frase = escolher_frase(estado.ultima_frase_impaciencia)
        estado.ultima_frase_impaciencia = indice
        # A pergunta que ficou sem resposta volta; sem ela, o proximo ponto.
        # Repete a pergunta que ficou no ar (a ultima do detetive), nao um tema
        # novo: era o que fazia o timeout "voltar" a uma pergunta antiga.
        texto = frase
        if estado.ultima_pergunta:
            texto = f"{frase} {estado.ultima_pergunta}"
        else:
            alvo = self.partida.proximo_tema(estado, temas)
            if alvo is not None and alvo.pergunta_modelo:
                texto = f"{frase} {alvo.pergunta_modelo}"
        resposta = self.partida.resposta_de_timeout(estado, texto, frase)
        self.partida.registrar_pergunta_do_detetive(estado, texto, temas)
        self.partida.store.salvar(estado)
        self.turn_counter[estado.session_id] = estado.turno
        try:
            self.memory.append_turn(estado.session_id, "O jogador não respondeu dentro do prazo.", role="sistema")
            self.memory.append_turn(estado.session_id, texto, role="detetive")
            self.memory.append_suspicion(estado.session_id, estado.suspeita)
        except OSError as exc:
            logger.error(f"Historico do timeout nao gravado: {exc}")
        return resposta

    # ─── Auxiliares ──────────────────────────────────────────────────────────

    def _resposta_de_partida_encerrada(self, estado: gr.EstadoPartida) -> Dict[str, Any]:
        """O que ja foi entregue no fim da partida, marcado como ignorado."""
        if estado.ultima_resposta:
            resposta = copy.deepcopy(estado.ultima_resposta)
        else:  # nao deveria ocorrer: partida encerrada sempre tem ultima resposta
            resposta = self.partida.montar_resposta(
                copy.deepcopy(estado), self._fallback_base(estado.suspeita))
        resposta["ignorado"] = True
        resposta.setdefault("status_investigacao", {})["fim_de_jogo"] = True
        return resposta

    def _gravar_historico(self, session_id: str, player_text: str,
                          resposta: Dict[str, Any], estado: gr.EstadoPartida) -> None:
        """Historico textual e metadados, so DEPOIS de o turno ter terminado bem."""
        try:
            self.memory.append_turn(session_id, player_text, role="suspeito")
            self.memory.append_turn(session_id, resposta.get("texto_detetive", ""), role="detetive")
            self.memory.append_suspicion(session_id, estado.suspeita)
            if resposta.get("status_investigacao", {}).get("detectou_mentira"):
                self.memory.append_contradiction(session_id, resposta["id_turno"])
        except OSError as exc:
            logger.error(f"Historico do turno nao gravado: {exc}")

    def _repete_fala_anterior(self, session_id: str, texto: str) -> bool:
        """
        Fala nova quase idêntica a uma das últimas do detetive.

        Medido no Qwen 2.5 3B: mesmo com "não repita" no prompt, 1 em cada ~12
        turnos saía cópia exata da fala anterior. Instrução não resolve modelo
        pequeno; checar a saída resolve, e vale para qualquer provedor.
        """
        anteriores = [t for papel, t in self.memory.get_dialogo(session_id, max_falas=8)
                      if papel == "DETETIVE"]
        return any(
            difflib.SequenceMatcher(None, texto, anterior).ratio() >= self.LIMIAR_REPETICAO
            for anterior in anteriores[-3:]
        )

    # 0.85 separa cópia (100%, ou só a pontuação mudou) de fala que reaproveita
    # um trecho — "Senhor Reis, onde o senhor estava..." é normal se repetir.
    LIMIAR_REPETICAO = 0.85

    def _fallback_base(self, suspeita: int) -> Dict[str, Any]:
        return {
            "id_turno": 0,
            "texto_detetive": "",
            "status_investigacao": {
                "nivel_suspeita": suspeita,
                "congelar_input": False,
                "detectou_mentira": False,
                "fim_de_jogo": False
            },
            "feedback_visual": {
                "cor_iluminacao": "#FFFF00",
                "bpm_musica": 100,
                "animacao_trigger": "Thinking"
            }
        }

    def _fallback_response(self, estado: gr.EstadoPartida, caso, temas: List[Tema],
                           interp: Interpretacao, fatos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ultimo recurso quando o modelo nao deu nenhuma resposta aproveitavel.
        A suspeita NAO se mexe (antes caia num 30 arbitrario) e a fala reage ao
        tipo da declaracao em vez do eco generico "Interessante... continue".
        """
        base = self._fallback_base(estado.suspeita)
        base["texto_detetive"] = self._fala_deterministica(caso, temas, estado, interp, fatos)
        return base
