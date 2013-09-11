from query import load_classifier_and_ngrams,  maps
import csv
from scipy.spatial.distance import cosine
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="model file to be used for semeval.py script")
    parser.add_argument('--all_synsets', action='store_true',)
    parser.add_argument('--top_synset', action='store_true',)
    parser.add_argument('--wordsim_root', help="folder containing wordsim353 csv file", default="/home/dfried/data/wordsim/combined.csv")
    args = parser.parse_args()

    classifier, ngram_reader = load_classifier_and_ngrams(args.model)

    id_map, reverse_map = maps(ngram_reader)
    E = classifier.embedding_layer.embedding

    if args.all_synsets or args.top_synset:
        from nltk.corpus import wordnet as wn
        S = classifier.synset_embedding_layer.embedding
        synsets = ['NONE'] + list(wn.all_synsets())
        id_to_synset = dict(enumerate(synsets))
        synset_to_id = dict((synset, index) for (index, synset) in enumerate(synsets))
        def get_embedding(word):
            word_synsets = wn.synsets(word)
            if not word_synsets:
                indices = [0]
            elif args.all_synsets:
                indices = [synset_to_id[synset] for synset in word_synsets]
            elif args.top_synset:
                indices = [synset_to_id[word_synsets[0]]]
            synset_vector = S[indices].mean(0)
            if word not in id_map:
                print 'warning: %s not in vocab' % word
            return np.concatenate([E[id_map[word]], synset_vector])
    else:
        def get_embedding(word):
            if word not in id_map:
                print 'warning: %s not in vocab' % word
            return E[id_map[word]]

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
