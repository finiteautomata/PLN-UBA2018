"""Train a sequence tagger.

Usage:
  train.py [-m <model>] [-n <n>] [-c <clf>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: badbase]:
                  badbase: Bad baseline
                  base: Baseline
                  memm: Maximum Entropy Markov Model
  -n <n>        Order of the model (if needed).
  -c <clf>      Classifier to use if the model is a MEMM [default: svm]:
                  maxent: Maximum Entropy (i.e. Logistic Regression)
                  svm: Support Vector Machine
                  mnb: Multinomial Bayes
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from tagging.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger, BadBaselineTagger
from tagging.memm import MEMM
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

models = {
    'badbase': BadBaselineTagger,
    'base': BaselineTagger,
    'memm': MEMM,
}


classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}



if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    print("Loading corpus...")
    corpus = SimpleAncoraCorpusReader('data/ancora/', files)
    sents = corpus.tagged_sents()

    # train the model
    print("Training model...")
    model_class = models[opts['-m']]
    # USEFUL FOR MODELS WITH PARAMETERS:
    if opts['-n']:
        n = int(opts['-n'])
        if opts['-m'] == 'memm':
            clf = classifiers[opts['-c']]()
            model = model_class(n, sents, clf=clf)
        else:
            model = model_class(n, sents)
    else:
        # only for baselines
        model = model_class(sents)

    # save it
    print("Saving model...")
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
