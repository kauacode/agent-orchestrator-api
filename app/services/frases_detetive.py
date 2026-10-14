"""
Falas prontas do detetive para quando o jogador deixa o tempo acabar.

Ficam aqui, num lugar so, e NAO passam pelo modelo: um timeout nao tem fala do
jogador para analisar, e gastar uma chamada de IA (15-30 s em CPU) para gerar
uma frase de impaciencia seria o oposto do que a frase comunica.
"""
import random
from typing import Optional, Tuple

FRASES_IMPACIENCIA = (
    "Vai me responder ou não?",
    "Estou esperando. Responda à pergunta.",
    "Não tenho o dia inteiro. Vamos!",
    "Pare de enrolar e responda.",
    "Você está dificultando bastante as coisas para si mesmo.",
    "Não teste a minha paciência. Responda!",
    "Por que está demorando tanto para responder?",
    "Não vou perguntar a mesma coisa o dia inteiro.",
    "Chega de silêncio. Quero uma resposta.",
    "Você está começando a me irritar. Colabore!",
)


def escolher_frase(ultimo_indice: Optional[int],
                   rng: Optional[random.Random] = None) -> Tuple[int, str]:
    """
    (indice, frase) sorteada entre as que NAO foram a ultima usada: dois
    timeouts seguidos nunca repetem a mesma fala (ha 10 alternativas).
    """
    rng = rng or random
    candidatos = [i for i in range(len(FRASES_IMPACIENCIA)) if i != ultimo_indice]
    indice = rng.choice(candidatos)
    return indice, FRASES_IMPACIENCIA[indice]
