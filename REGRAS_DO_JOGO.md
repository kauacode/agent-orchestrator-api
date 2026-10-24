# Regras do jogo (fonte da verdade: `app/services/game_rules.py`)

Todas as regras são **determinísticas** e vivem no backend. O modelo de IA só propõe uma fala e um
nível de suspeita; o sistema decide o resto. Os números ficam na dataclass `Regras`. O Unity só
exibe: o texto do tutorial (`TutorialOverlay.cs`) espelha estes números e deve mudar junto.

Escala: `nivel_suspeita` é inteiro de 0 a 100.

## Turno
Um turno = uma fala do jogador processada **ou** um prazo que venceu. Ajustes do mesmo turno
(proposta do modelo + tempo + confissão/timeout) são somados antes de fechar e contam uma vez só.
Estado em `data/<sessao>_estado.json` (sobrevive a reinício). Reenviar a mesma `interacao_id`
devolve a mesma resposta; o Unity ignora respostas cujo `id_turno` já foi aplicado.

## Encerramento (precedência, nesta ordem)
1. Suspeita = **100** → derrota `suspeita_maxima`, imediata, em qualquer turno.
2. Suspeita ≥ **80** em **3 turnos consecutivos** → derrota `suspeita_sustentada`.
3. Turno ≥ **8** e suspeita ≤ **30** → vitória `liberado`.
4. **Turno 10** (`MAX_TURNOS`, absoluto): suspeita ≥ **60** → derrota, senão vitória (`limite_de_turnos`).

O turno 10 encerra a partida dentro de `fechar_turno`; um turno 11 nunca é processado (chamadas
depois do fim devolvem a última resposta com `ignorado: true`, sem IA nem histórico).
*Decisão de balanceamento do corte de 60%*: com o limite menor há só 3 turnos (8–10) em que a
vitória por "liberado" é possível. Sobreviver aos 10 turnos sem o detetive ter reunido suspeita
alta (< 60) é, por coerência com a regra 3, uma vitória; ≥ 60 significa que ele tinha caso.
O valor foi mantido (não era uma simples troca de 12 por 10) e fica em `LIMITE_DERROTA_SUSPEITA`.

## Timer de 60 s por turno
* O prazo começa quando a **pergunta do detetive fica disponível**: o backend abre a janela dentro
  da própria resposta (`proxima_interacao_id`, `proxima_restante_s`), depois do processamento da IA.
  A 1ª pergunta (fala de abertura, do cliente) abre a janela por `POST /session/{id}/interacao`,
  chamado quando o tutorial é fechado.
* Relógio do servidor. O tempo de resposta é medido na **chegada** da requisição (fila do lock e
  tempo de IA não contam), limitado ao prazo.
* `POST /session/{id}/interacao/{iid}/expirar`: o servidor confere o próprio relógio (409 se
  cedo; tolerância 1,5 s), aplica o timeout **uma vez** e **não** chama a IA. Resposta tardia
  (> 60 s + 2 s de folga) vira o timeout. Resposta e expiração simultâneas são serializadas por
  um lock por sessão; o perdedor vira no-op (`aplicado: false`).
* Se o cliente some e a janela vence sem aviso, o timeout é aplicado ao reabrir a janela.
* Falha de infraestrutura (IA fora do ar) não muda a partida e reabre a janela com os 60 s inteiros.
* **Pause**: não congela o timer (decisão: o relógio é do servidor e pausar não pode dar tempo
  extra para pensar). O tutorial avisa.

## Faixas do tempo de resposta (só para respostas válidas)
| Faixa | Intervalo | Efeito na suspeita |
|---|---|---|
| baixo | t < 30 s | 0 |
| médio | 30 s ≤ t < 45 s | +1 |
| alto | 45 s ≤ t ≤ 60 s | +2 |
| timeout | sem resposta no prazo | ver abaixo (não é uma faixa) |

* Classificação determinística e sem arredondar antes (`classificar_tempo`); nenhuma chamada de IA.
* A IA recebe `tempo_resposta_segundos`, `nivel_tempo_resposta` e `status_tempo_resposta` com a
  instrução de que é um sinal contextual fraco, **não** prova de culpa/mentira/nervosismo, e de
  **não** incluir o efeito do tempo na suspeita que propõe.
* Separação de efeitos: suspeita final = proposta semântica do modelo (presa a +40/−25) **+** efeito
  da faixa (0/+1/+2, aplicado uma vez pelo sistema), com o piso da confissão (85/100) por cima. O efeito antigo
  contínuo de até +5 foi **removido** (era excessivo para um sinal secundário e duplicaria o novo).

## Timeout
* Penalidade `PENALIDADE_TIMEOUT = +15`, uma vez, sem somar o efeito de faixa (não há resposta).
* O detetive diz uma de 10 frases de impaciência (`frases_detetive.py`), nunca a mesma duas vezes
  seguidas, seguida da pergunta que ficou sem resposta. Sem chamada de IA.
* Avança o turno; respeita 100%, sequência de 80% e o turno 10. Não abre janela se a partida acabou.

## Confissão e perguntas repetidas
* `interpretador.py` classifica cada fala (confissão / negação / evasiva / informação / ambígua) por
  frases e escopo de negação. A 1ª confissão leva a suspeita a **pelo menos 85**
  (`PISO_PRIMEIRA_CONFISSAO`, começa a sequência de 80%) e **não encerra** a partida: o detetive
  ainda confronta os detalhes. Confirmar a confissão leva a **100** (`PISO_CONFISSAO_REPETIDA`)
  e encerra. Negar e depois confessar (ou o inverso) marca contradição pelo sistema.
* Tema já esclarecido não é perguntado de novo, a menos que o jogador traga algo novo sobre ele.
  Fala repetida é recusada, o modelo recebe uma correção e, se insistir, o sistema a substitui.

## Regras: o que já existia × o que foi acrescentado
| Regra | Origem | Onde | Situação |
|---|---|---|---|
| Vitória turno ≥ 8 e suspeita ≤ 30 | projeto (Unity) | `game_rules.avaliar_fim` | preservada, movida ao backend |
| Derrota "3 turnos ≥ 90" | projeto | — | substituída pelo pedido (100% imediato; 80% × 3) |
| Faixa 0–100, bpm 60–140 | projeto | `_limitar_faixas` | preservada |
| 100% / 80%×3 | pedido, rodada 1 | `avaliar_fim`, `fechar_turno` | preservadas |
| Teto 10 turnos | pedido, rodada 2 | `Regras.MAX_TURNOS` | novo valor (era 12, nossa proposta) |
| Corte 60% no limite | rodada 1 (nossa) | `LIMITE_DERROTA_SUSPEITA` | mantido; justificado acima |
| Variação +40/−25 por turno | rodada 1 (nossa) | `suspeita_do_turno` | mantida: vem das âncoras do caso ("+25 a +40", "−10 a −25") e impede saltos arbitrários; não conflita com as demais |
| Confissão: piso de 85, confirmar = 100 | rodada 1 (nossa), revista | `PISO_PRIMEIRA_CONFISSAO` / `PISO_CONFISSAO_REPETIDA` | substitui o antigo "+25 uma vez" (`BONUS_CONFISSAO` não existe mais): "eu matei" valia menos que uma mentira |
| Timeout +15 | rodada 1 (nossa) | `PENALIDADE_TIMEOUT` | mantido |
| Penalidade de tempo contínua até +5 | rodada 1 (nossa) | — | **removida**; substituída pelas faixas 0/+1/+2 |
| Recusa de fala repetitiva | rodada 1 (nossa) | `_gerar_fala` | mantida |
