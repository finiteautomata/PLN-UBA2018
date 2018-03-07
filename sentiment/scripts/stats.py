"""Draw a learning curve for a Sentiment Analysis model.

Usage:
  curve.py [-m <model>] [-c <clf>]
  curve.py -h | --help

Options:
  -m <model>    Model to use [default: basemf]:
                  basemf: Most frequent sentiment
                  clf: Machine Learning Classifier
  -c <clf>      Classifier to use if the model is a MEMM [default: svm]:
                  maxent: Maximum Entropy (i.e. Logistic Regression)
                  svm: Support Vector Machine
                  mnb: Multinomial Bayes
  -h --help     Show this screen.
"""
from docopt import docopt
from sentiment.tass import InterTASSReader, GeneralTASSReader


def stats_for(name, reader):
    target = list(reader.y())
    n = len(list(reader.tweets()))
    P = target.count('P')
    N = target.count('N')
    NEU = target.count('NEU')
    NONE = target.count('NONE')
    print("=" * 80)
    print('Estadísticas de %s' % name)
    print('#tweets: {}'.format(n))
    print('P: cantidad {} frecuencia {}'.format(P, P / n))
    print('N: cantidad {} frecuencia {}'.format(N, N / n))
    print('NEU: cantidad {} frecuencia {}'.format(NEU, NEU / n))
    print('NONE: cantidad {} frecuencia {}'.format(NONE, NONE / n))

    print("=" * 80)


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load training corpus
    intertass = InterTASSReader('data/TASS/InterTASS/tw_faces4tassTrain1000rc.xml')
    gentass = GeneralTASSReader('data/TASS/GeneralTASS/general-tweets-train-tagged.xml', simple=True)

    stats_for('Train InterTASS', intertass)
    stats_for('Train GeneralTASS', gentass)
