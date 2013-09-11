import csv
from scipy.spatial.distance import cosine
import gzip, cPickle

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

    def get_embedding(word):
        if args.all_synsets:
            include_synsets='all'
        elif args.top_synset:
            include_synsets='top'
        else:
            include_synsets=None
        return model.get_embedding(word, include_synsets=include_synsets, normalize_components=args.normalize_components)

    with open(args.wordsim_root) as csvfile:
        # discard headr
        lines = list(csv.reader(csvfile))[1:]

    words = [line[:2] for line in lines]
    human_scores = [float(line[-1]) for line in lines]

    model_scores = [1 - cosine(get_embedding(word1.lower()), get_embedding(word2.lower()))
                    for word1, word2 in words]

    from scipy.stats import spearmanr

    rho, p = spearmanr(model_scores, human_scores)

    print 'rho: %f\tp: %f' % (rho, p)
