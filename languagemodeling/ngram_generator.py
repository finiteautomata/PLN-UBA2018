import numpy as np

class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._model = model

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
        try:
            prob_vector = self._model.cond_prob_density(prev_tokens)
            p = np.array([*prob_vector.values()])
            return np.random.choice(
                [*prob_vector.keys()],
                p=p
            )
        except ValueError as e:
            # Error numérico, intentamos subsanarlo
            s = sum(p)
            print("Error numérico. La suma de probabilidades es {}".format(s))
            p /= s
            return np.random.choice(
                [*prob_vector.keys()],
                p=p
            )
