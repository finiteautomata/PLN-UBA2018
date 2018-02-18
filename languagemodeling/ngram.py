# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math
import pandas as pd
from .helpers import generate_ngrams
from . import BEGIN_MARKER, END_MARKER

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

        ngrams = generate_ngrams(n, sents)
        relative_counts = defaultdict(lambda: defaultdict(float))
        count = defaultdict(int)

        for ngram in ngrams:
            prev_tokens, token = ngram[:-1], ngram[-1]
            count[ngram] += 1
            count[prev_tokens] += 1
            relative_counts[prev_tokens][token] += 1

        self._count = dict(count)
        self._generate_probs(relative_counts)


    def _generate_probs(self, relative_counts):
        """
        Generate probability dictionary from relative counts
        """
        self._probs = {}
        for prev_token, count_vector in relative_counts.items():
            total = sum(count_vector.values())
            self._probs[prev_token] = {}
            for token, count in count_vector.items():
                self._probs[prev_token][token] = count / total

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

        try:
            return self._probs[prev_tokens][token]
        except KeyError as e:
            # Uso excepcion porque el sparse tira eso..
            return .0

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        sent += [END_MARKER]
        prev_tokens = tuple([BEGIN_MARKER] * (self._n-1))

        prob = 1
        for i in range(0, len(sent)):
            token = sent[i]
            next_prob = self.cond_prob(token, prev_tokens)
            prob *= next_prob

            if self._n > 1:
                prev_tokens = prev_tokens[1:] + (token,)
        return prob

    def cond_prob_density(self, prev_tokens):
        """
        Returns conditional probability density on prev_tokens
        """

        return self._probs[prev_tokens]

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        sent += [END_MARKER]
        prev_tokens = tuple([BEGIN_MARKER] * (self._n-1))

        logprob = 0
        for i in range(0, len(sent)):
            token = sent[i]
            next_prob = self.cond_prob(token, prev_tokens)
            if next_prob == 0:
                logprob = float("-inf")
                break

            logprob += math.log2(next_prob)

            if self._n > 1:
                prev_tokens = prev_tokens[1:] + (token,)
        return logprob


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        self._voc = voc = set(tok for sent in sents for tok in sent)
        self._voc.add(END_MARKER)
        self._V = len(voc)
        super().__init__(n, sents)

    def _generate_probs(self, relative_counts):
        """
        Generate probability dictionary from relative counts
        """
        self._probs = {}
        for prev_token, count_vector in relative_counts.items():
            total = sum(count_vector.values())
            self._probs[prev_token] = {}
            for token, count in count_vector.items():
                self._probs[prev_token][token] =\
                    (count + 1) / (total + self._V)

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            # if prev_tokens not given, assume 0-uple:
            prev_tokens = ()

        default = None
        try:
            default = 1 / (self._count[prev_tokens] + self._V)
            return self._probs[prev_tokens][token]
        except KeyError as e:
            return default or (1 / self._V)

    def cond_prob_density(self, prev_tokens):
        """
        Returns conditional probability density on prev_tokens
        """


        default = 1 / (self.count(prev_tokens) + self._V)
        ret = {v:default for v in self._voc}
        try:
            ret.update(self._probs[prev_tokens])
        except KeyError as e:
            pass
        return ret
