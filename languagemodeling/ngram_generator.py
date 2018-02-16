import numpy as np
from .ngram import EMPTY_TOKEN

class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._model = model
        # WORK HERE!!

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self._model._n

        sent = []
        prev_tokens = ['<s>'] * (n - 1)
        token = self.generate_token(tuple(prev_tokens))

        while token != '</s>':
            sent.append(token)
            prev_tokens += [token]
            prev_tokens = prev_tokens[1:]
            token = self.generate_token(tuple(prev_tokens))

        return sent

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if not prev_tokens:
            prev_tokens = ()

        prob_vector = self._model._probs[prev_tokens]
        return np.random.choice(
            [*prob_vector.keys()],
            p=[*prob_vector.values()]
        )
