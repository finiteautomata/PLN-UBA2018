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
    print("=" * 80)
    print('Estadísticas de %s' % name)
    print('Cantidad total de tweets: {}'.format(len(list(reader.tweets()))))
    print('Cantidad de tweets con polaridad P: {}'.format(target.count('P')))
    print('Cantidad de tweets con polaridad N: {}'.format(target.count('N')))
    print('Cantidad de tweets con polaridad NEG: {}'.format(target.count('NEU')))
    print('Cantidad de tweets con polaridad NONE: {}'.format(target.count('NONE')))
    print("=" * 80)


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load training corpus
    intertass = InterTASSReader('data/TASS/InterTASS/tw_faces4tassTrain1000rc.xml')
    reader_2 = GeneralTASSReader('data/TASS/GeneralTASS/general-tweets-train-tagged.xml', simple=True)

    stats_for('Train InterTASS', intertass)
