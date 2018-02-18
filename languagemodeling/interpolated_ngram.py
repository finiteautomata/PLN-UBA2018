"""InterpolatedNGram model."""
from collections import defaultdict
from .helpers import generate_ngrams, count_all_grams
from .ngram import LanguageModel, AddOneNGram, NGram

class InterpolatedNGram(LanguageModel):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).k
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        train_sents = held_out_sents = None
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
        sents_to_count = train_sents or sents

        self._count = dict(count_all_grams(n, sents_to_count))
        self._addone = addone

        print('Creating models..')
        self._create_models(train_sents)

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

    def _create_models(self, sents):
        model_class = AddOneNGram if self._addone else NGram
        self._models = {
            order:model_class(order, sents) for order in range(1, self._n+1)
        }

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
                lambdaa = self.count(prev_tokens) / (self.count(prev_tokens) + self._gamma)
                lambdaa *= (1 - cum_lambda)
                cond_ml = self._models[self._n-i].cond_prob(token, prev_tokens)
            else:
                lambdaa = 1 - cum_lambda
                cond_ml = self._models[1].cond_prob(token)
            prob += lambdaa * cond_ml
            cum_lambda += lambdaa

        return prob
