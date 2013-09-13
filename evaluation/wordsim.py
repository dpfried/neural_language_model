import csv
from scipy.spatial.distance import cosine
import gzip, cPickle

from scipy.stats import spearmanr

def run(model, include_synsets, normalize_components, wordsim_root):
    def get_embedding(word):
        return model.get_embedding(word, include_synsets=include_synsets, normalize_components=normalize_components)

    with open(wordsim_root) as csvfile:
        # discard headr
        lines = list(csv.reader(csvfile))[1:]

    words = [line[:2] for line in lines]
    human_scores = [float(line[-1]) for line in lines]

    model_scores = [1 - cosine(get_embedding(word1.lower()), get_embedding(word2.lower()))
                    for word1, word2 in words]

    rho, p = spearmanr(model_scores, human_scores)
    return rho, p


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="model file to be used for semeval.py script")
    parser.add_argument('--all_synsets', action='store_true',)
    parser.add_argument('--top_synset', action='store_true',)
    parser.add_argument('--normalize_components', action='store_true',)
    parser.add_argument('--wordsim_root', help="folder containing wordsim353 csv file", default="/home/dfried/data/wordsim/combined.csv")
    args = parser.parse_args()

    with gzip.open(args.model) as f:
        model = cPickle.load(f)

    if args.all_synsets:
        include_synsets='all'
    elif args.top_synset:
        include_synsets='top'
    else:
        include_synsets=None

    rho, p = run(model, include_synsets, args.normalize_components, args.wordsim_root)

    print 'rho: %f\tp: %f' % (rho, p)
