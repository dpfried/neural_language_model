import numpy as np
from query import cosine_similarity
import gzip

def read_turian_embeddings(embedding_file):
    embeddings = {}
    with gzip.open(embedding_file) as f:
        for line in f:
            tokens = line.strip().split()
            embeddings[tokens[0]] = np.array(map(float, tokens[1:]))
    return embeddings

def parse_line(line):
    tokens = line.strip().split()
    words = tokens[:4]
    score = float(tokens[-1])
    return words, score

def parse_file(filename):
    with open(filename) as f:
        return [parse_line(line) for line in f]

def make_verb_applicability_fn(turian_embeddings):
    def verb_applicability(verb, noun1, noun2):
        for word in [verb, noun1, noun2]:
            if word not in turian_embeddings:
                print "warning: %s not in vocabulary" % word
        # return E[id_map[verb]] + E[id_map[noun1]] + E[id_map[noun2]]
        return sum(turian_embeddings[w] for w in [verb, noun1, noun2])
    def compare_verbs(verb1, verb2, noun1, noun2):
        return cosine_similarity(verb_applicability(verb1, noun1, noun2),
                                 verb_applicability(verb2, noun1, noun2))
    return verb_applicability, compare_verbs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--turian_embeddings', default='/home/dfried/data/embeddings-scaled.EMBEDDING_SIZE=50.txt.gz')
    parser.add_argument('--verb_file', default='/home/dfried/code/verb_disambiguation')
    args = parser.parse_args()

    parsed_lines = parse_file(args.verb_file)

    turian_embeddings = read_turian_embeddings(args.turian_embeddings)
    verb_applicability, compare_verbs = make_verb_applicability_fn(turian_embeddings)

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
