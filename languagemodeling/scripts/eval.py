"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import math
import nltk


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    test_corpus = nltk.corpus.PlaintextCorpusReader(
        "data/",
        ".*test\.txt",
    )
    sents = test_corpus.sents()
    print("Testing at {} sents".format(len(sents)))

    # compute the cross entropy
    log_prob = model.log_prob(sents)
    n = sum(len(sent) + 1 for sent in sents)  # count '</s>' event
    e = - log_prob / n
    p = math.pow(2.0, e)

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(e))
    print('Perplexity: {}'.format(p))
