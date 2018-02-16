# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math

BEGIN_MARKER = '<s>'
END_MARKER = '</s>'

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

    def _generate_ngrams_for_sentence(self, n, sentence):
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

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        sent += [END_MARKER]
        prev_tokens = tuple([BEGIN_MARKER] * (self._n-1))

        logprob = 0
        for i in range(self._n-1, len(sent)):
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
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = voc = set()
        # WORK HERE!!

        self._V = len(voc)  # vocabulary size

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
        assert len(prev_tokens) == n - 1

        # WORK HERE!!


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        # WORK HERE!!
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            # WORK HERE!!

            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # use grid search to choose gamma
            min_gamma, min_p = None, float('inf')

            # WORK HERE!! TRY DIFFERENT VALUES BY HAND:
            for gamma in [100 + i * 50 for i in range(10)]:
                self._gamma = gamma
                p = self.perplexity(held_out_sents)
                print('  {} -> {}'.format(gamma, p))

                if p < min_p:
                    min_gamma, min_p = gamma, p

            print('  Choose gamma = {}'.format(min_gamma))
            self._gamma = min_gamma

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        # WORK HERE!! (JUST A RETURN STATEMENT)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            # if prev_tokens not given, assume 0-uple:
            prev_tokens = ()
        assert len(prev_tokens) == n - 1

        # WORK HERE!!
        # SUGGESTED STRUCTURE:
        tokens = prev_tokens + (token,)
        prob = 0.0
        cum_lambda = 0.0  # sum of previous lambdas
        for i in range(n):
            # i-th term of the sum
            if i < n - 1:
                # COMPUTE lambdaa AND cond_ml.
                pass
            else:
                # COMPUTE lambdaa AND cond_ml.
                # LAST TERM: USE ADD ONE IF NEEDED!
                pass

            prob += lambdaa * cond_ml
            cum_lambda += lambdaa

        return prob
