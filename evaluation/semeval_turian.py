#!/usr/bin/env python
import subprocess
import os
from glob import glob
import numpy as np
import sys
from query import make_analogy_fns
from grefenstette_turian import read_turian_embeddings, cosine_similarity

def attr_dict(filename):
    with open(filename) as f:
        keys_and_vals = [line.split(':') for line in f if line.strip()]
    return dict((key.strip(), val.strip()) for key, val in keys_and_vals)

def parse_correlation(filename):
    attrs = attr_dict(filename)
    return float(attrs['Spearman Correlation'])

def parse_accuracy(filename):
    attrs = attr_dict(filename)
    return float(attrs['Overall Accuracy'].strip('%')) / 100.

def get_paradigms(question_file):
    from itertools import islice
    with open(question_file) as f:
        lines = list(islice(f, 4, 8))
        if not(lines[-1].strip()): # possibly only 3 golden examples
            lines = lines[:-1]
        pairs = [line.lower().strip().split(':') for line in lines]
    return pairs

def get_examples(answer_file):
    with open(answer_file) as f:
        return [line.lower().strip().strip('"').split(':') for line in f]


def make_analogy_fns(turian_embeddings):
    def analogy_fn(word1, word2):
        if word1 not in turian_embeddings:
            print "warning: %s not in vocabulary" % word1
            word1 = '*UNKNOWN*'
        if word2 not in turian_embeddings:
            print "warning: %s not in vocabulary" % word2
            word2 = '*UNKNOWN*'
        return turian_embeddings[word1] - turian_embeddings[word2]
    def choose_best(reference_analogy, other_pairs):
        # reference_analogy = analogy_fn(word1, word2)
        other_analogies = [analogy_fn(w1, w2) for (w1, w2) in other_pairs]
        scores = [cosine_similarity(reference_analogy, other) for other in other_analogies]
        return list(reversed(sorted(zip(scores, other_pairs))))
    return analogy_fn, choose_best



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder', help="folder to write results to")
    parser.add_argument('--turian_embeddings', default='/home/dfried/data/embeddings-scaled.EMBEDDING_SIZE=50.txt.gz')
    parser.add_argument('--semeval_root', help="folder containing semeval data", default="/home/dfried/code/semeval")
    args = parser.parse_args()

    turian_embeddings = read_turian_embeddings(args.turian_embeddings)
    analogy_fn, choose_best = make_analogy_fns(turian_embeddings)

    def semeval_path(suffix):
        return os.path.join(args.semeval_root, suffix)

    sets_by_folder = {
        # semeval_path('Training'): "10a 1a 2c 2h 3a 3c 4c 5d 5i 7a".split(),
        semeval_path('Testing'): "1b 1c 1d 1e 2a 2b 2d 2e 2f 2g 2i 2j 3b 3d 3e 3f 3g 3h 4a 4b 4d 4e 4f 4g 4h 5a 5b 5c 5e 5f 5g 5h 6a 6b 6c 6d 6e 6f 6g 6h 7b 7c 7d 7e 7f 7g 7h 8a 8b 8c 8d 8e 8f 8g 8h 9a 9b 9c 9d 9e 9f 9g 9h 9i 10b 10c 10d 10e 10f".split()
    }

    def nlm_scaled_path(suffix):
        return os.path.join(args.output_folder, 'ModelScaled-%s.txt' % suffix)

    def turker_scaled_path(suffix):
        return os.path.join(args.output_folder, 'TurkerScaled-%s.txt' % suffix)

    def spearman_results_path(suffix):
        return os.path.join(args.output_folder, 'SpearmanResults-%s.txt' % suffix)

    def maxdiff_scaled_path(suffix):
        return os.path.join(args.output_folder, 'MaxDiff-%s.txt' % suffix)

    def maxdiff_results_path(suffix):
        return os.path.join(args.output_folder, 'MaxDiffResults-%s.txt' % suffix)

    for semeval_data_folder, sets_to_run in sets_by_folder.items():
        # only test on testing
        for s in sets_to_run:
            print '-------'
            print s

            answer_file = os.path.join(semeval_data_folder, 'Phase1Answers', 'Phase1Answers-%s.txt' % s)
            question_file = os.path.join(semeval_data_folder, 'Phase1Questions', 'Phase1Questions-%s.txt' % s)

            paradigms = get_paradigms(question_file)
            examples = get_examples(answer_file)

            paradigm_analogies = [analogy_fn(*pair) for pair in paradigms]
            # print zip(paradigms, paradigm_analogies)
            average_paradigm = sum(paradigm_analogies) / len(paradigm_analogies)

            rankings = choose_best(average_paradigm, examples)
            min_score = min(score for score, pair in rankings)
            # print rankings
            with open(nlm_scaled_path(s), 'w') as f:
                for score, (w1, w2) in rankings:
                    f.write('%0.2f "%s:%s"\n' % ((score - min_score)  * 100 + 1, w1, w2))

            subprocess.call([semeval_path("maxdiff_to_scale.pl"),
                             os.path.join(semeval_data_folder, 'Phase2Answers', 'Phase2Answers-%s.txt' % s),
                             turker_scaled_path(s)],
                            stdout=open(os.devnull, 'wb'))

            subprocess.call([semeval_path("score_scale.pl"),
                             turker_scaled_path(s),
                             nlm_scaled_path(s),
                             spearman_results_path(s)],
                            stdout=open(os.devnull, 'wb'))

            subprocess.call([semeval_path("scale_to_maxdiff.pl"),
                             os.path.join(semeval_data_folder, "Phase2Questions", "Phase2Questions-%s.txt" % s),
                             nlm_scaled_path(s),
                             maxdiff_scaled_path(s)],
                            stdout=open(os.devnull, 'wb'))

            subprocess.call([semeval_path("score_maxdiff.pl"),
                             os.path.join(semeval_data_folder, "Phase2Answers", "Phase2Answers-%s.txt" % s),
                             maxdiff_scaled_path(s),
                             maxdiff_results_path(s)],
                            stdout=open(os.devnull, 'wb'))

    # compute average correlation
    corrs = [parse_correlation(filename) for filename in glob(spearman_results_path('*'))]

    # compute average accuracy
    accuracy = [parse_accuracy(filename) for filename in glob(maxdiff_results_path('*'))]

    print 'average correlation: %f' % np.mean(corrs)
    print 'average accuracy: %f' % np.mean(accuracy)
