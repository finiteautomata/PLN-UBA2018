# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        result = 0.0
        for i, sent in enumerate(sents):
            lp = self.sent_log_prob(sent)
            if lp == -math.inf:
                return lp
            result += lp
        return result

    def cross_entropy(self, sents):
        log_prob = self.log_prob(sents)
        n = sum(len(sent) + 1 for sent in sents)  # count '</s>' events
        e = - log_prob / n
        return e

    def perplexity(self, sents):
        return math.pow(2.0, self.cross_entropy(sents))


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        ngrams, nminusonegrams = self._generate_ngrams(n, sents)
        count = defaultdict(int)

        for ngram in ngrams + nminusonegrams:
            count[ngram] += 1

        self._count = count
        """
        Probabilidades de tokens iniciales:
        """
        self._initial_probs = defaultdict(float)

        for sent in sents:
            gram = tuple(sent[:n-1])
            self._initial_probs[gram] += 1 / len(sents)


    def _generate_ngrams_for_sentence(self, n, sentence):
        """
        Genera n-gramas y n-1 gramas
        """
        m = len(sentence)
        ngrams = []

        """
        Los n-1 primeros tokens tengo que rellenarlos
        """
        ngram = ['<s>'] + sentence[0:n-1]
        ngrams.append(tuple(ngram))

        for i in range(max(n-2, 0), len(sentence)-n+1):
            ngrams.append(tuple(sentence[i:i+n]))


        if n > 1:
            ngram = sentence[m-(n-1):m] + ['</s>']
            ngrams.append(tuple(ngram))
        else:
            ngrams.append(('</s>', ))

        nminusonegrams = [ngram[:-1] for ngram in ngrams if ngram != ('<s>',)]

        return ngrams, nminusonegrams

    def _generate_ngrams(self, n, sents):
        """
        Generar n-gramas a partir de las sentencias
        """
        ngrams, nminusonegrams = [], []

        for sent in sents:
            ng, nminusoneg = self._generate_ngrams_for_sentence(n, sent)
            ngrams += ng
            nminusonegrams += nminusoneg
        return ngrams, nminusonegrams


    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        prev_tokens = prev_tokens or ()
        num = self._count.get(prev_tokens + (token,), 0)
        quot = self._count.get(prev_tokens, 0)

        if quot == 0:
            return 0
        else:
            return num / quot
        # WORK HERE!!

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        sent += ['</s>']
        prev_tokens = tuple(sent[:self._n-1])

        prob = self._initial_probs[prev_tokens]
        for i in range(self._n-1, len(sent)):
            token = sent[i]
            next_prob = self.cond_prob(token, prev_tokens)
            prob *= next_prob

            if self._n > 1:
                prev_tokens = prev_tokens[1:] + (token,)
        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return 0# log(self.sent_prob(sent), 2)
