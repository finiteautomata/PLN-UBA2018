"""Helpers module."""
from collections import defaultdict

def generate_ngrams_for_sentence(n, sentence):
    """
    Genera n-gramas y n-1 gramas
    """
    m = len(sentence)
    ngrams = []

    """
    Los n-1 primeros tokens tengo que rellenarlos
    """
    for i in range(min(n-1, len(sentence))):
        ngram = ['<s>'] * (n-(i+1)) + sentence[0:i+1]
        ngrams.append(tuple(ngram))

    for i in range(max(n-2, 0), len(sentence)-n+1):
        ngrams.append(tuple(sentence[i:i+n]))

    if n > 1:
        ngram = sentence[m-(n-1):m] + ['</s>']
        ngrams.append(tuple(ngram))
    else:
        ngrams.append(('</s>', ))

    return ngrams

def generate_ngrams(n, sents):
    """
    Generar n-gramas a partir de las sentencias
    """
    ngrams = []

    for sent in sents:
        ng = generate_ngrams_for_sentence(n, sent)
        ngrams += ng
    return ngrams

def count_all_grams(n, sents):
    count = defaultdict(int)
    for k in range(1, n+1):
        ngrams = generate_ngrams(k, sents)
        for ngram in ngrams:
            count[ngram] += 1
        if k == 1:
            count[()] = len(ngrams)
    return count
