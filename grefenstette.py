import numpy as np
from query import cosine_similarity, load_classifier_and_ngrams, maps

def parse_line(line):
    tokens = line.strip().split()
    words = tokens[:4]
    score = float(tokens[-1])
    return words, score

def parse_file(filename):
    with open(filename) as f:
        return [parse_line(line) for line in f]

def make_verb_applicability_fn(classifier, ngram_reader):
    id_map, reverse_map = maps(ngram_reader)
    E = classifier.embedding_layer.embedding
    def verb_applicability(verb, noun1, noun2):
        for word in [verb, noun1, noun2]:
            if word not in id_map:
                print "warning: %s not in vocabulary" % word
        # return E[id_map[verb]] + E[id_map[noun1]] + E[id_map[noun2]]
        return sum(E[id_map[w]] for w in [verb, noun1, noun2])
    def compare_verbs(verb1, verb2, noun1, noun2):
        return cosine_similarity(verb_applicability(verb1, noun1, noun2),
                                 verb_applicability(verb2, noun1, noun2))
    return verb_applicability, compare_verbs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--verb_file', default='/home/dfried/code/verb_disambiguation')
    args = parser.parse_args()

    parsed_lines = parse_file(args.verb_file)

    classifier, ngram_reader = load_classifier_and_ngrams(args.model_file)
    verb_applicability, compare_verbs = make_verb_applicability_fn(classifier, ngram_reader)

    model_scores = []
    human_scores = []
    for tokens, human_score in parsed_lines:
        verb1, noun1, noun2, verb2 = tokens
        human_scores.append(human_score)
        model_score = compare_verbs(verb1, verb2, noun1, noun2)
        model_scores.append(model_score)
        print tokens, human_score, model_score

    print np.row_stack([human_scores, model_scores]).T[:10,:]

    from scipy.stats import spearmanr

    rho, p = spearmanr(model_scores, human_scores)

    print 'rho: %f\tp: %f' % (rho, p)
