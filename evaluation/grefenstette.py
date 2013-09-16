import numpy as np
from scipy.spatial.distance import cosine
import gzip, cPickle
from scipy.stats import spearmanr
from semantic_network import *
from semantic_network import _default_word

def parse_line(line):
    tokens = line.lower().strip().split()
    words = tokens[:4]
    score = float(tokens[-1])
    return words, score

def parse_file(filename):
    with open(filename) as f:
        return [parse_line(line) for line in f]

def make_verb_applicability_fn(get_embedding_fn):
    def verb_applicability(verb, noun1, noun2):
        return sum(get_embedding_fn(w) for w in [verb, noun1, noun2])
    def compare_verbs(verb1, verb2, noun1, noun2):
        return 1 - cosine(verb_applicability(verb1, noun1, noun2),
                          verb_applicability(verb2, noun1, noun2))
    return verb_applicability, compare_verbs

def run(model, include_synsets, normalize_components, verb_file):
    parsed_lines = parse_file(verb_file)

    def get_embedding(word):
        return model.get_embedding(word, include_synsets=include_synsets, normalize_components=normalize_components)

    verb_applicability, compare_verbs = make_verb_applicability_fn(get_embedding)

    model_scores = []
    human_scores = []
    for tokens, human_score in parsed_lines:
        verb1, noun1, noun2, verb2 = tokens
        human_scores.append(human_score)
        model_score = compare_verbs(verb1, verb2, noun1, noun2)
        model_scores.append(model_score)
        # print tokens, human_score, model_score

    # print np.row_stack([human_scores, model_scores]).T[:10,:]

    rho, p = spearmanr(model_scores, human_scores)
    return rho, p


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--all_synsets', action='store_true',)
    parser.add_argument('--top_synset', action='store_true',)
    parser.add_argument('--normalize_components', action='store_true',)
    parser.add_argument('--verb_file', default='/home/dfried/code/verb_disambiguation')
    args = parser.parse_args()

    with gzip.open(args.model_file) as f:
        model = cPickle.load(f)

    if args.all_synsets:
        include_synsets='all'
    elif args.top_synset:
        include_synsets='top'
    else:
        include_synsets=None

    rho, p = run(model, include_synsets, args.normalize_components, args.verb_file)

    print 'rho: %f\tp: %f' % (rho, p)
