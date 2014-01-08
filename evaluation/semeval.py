#!/usr/bin/env python
import pandas
import subprocess
import os
from glob import glob
import numpy as np
from query import  make_analogy_fns, get_vocab_container
import gzip, cPickle
import tempfile
import re

DEFAULT_SEMEVAL_ROOT = "/home/dfried/code/semeval"

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
    with open(question_file) as f:
        paradigm_string = re.search('Consider the following word pairs:(.*)What relation best describes', f.read(), re.DOTALL).groups()[0]
        return re.findall('(\w*):(\w*)', paradigm_string)

def get_examples(answer_file):
    with open(answer_file) as f:
        return [line.strip().strip('"').split(':') for line in f]

def category_names(filename=os.path.join(DEFAULT_SEMEVAL_ROOT, 'subcategories-list.txt')):
    def parse_line(line):
        '''
        "1, a, CLASS-INCLUSION, Taxonomic" -> "1a, (CLASS-INCLUSION, Taxonomic)"
        '''
        tokens = [s.strip() for s in line.split(',')]
        return (tokens[0] + tokens[1]), (tokens[2], tokens[3])
    with open(filename) as f:
        return dict(parse_line(line) for line in f)

def run(embeddings, vocab_container,  semeval_root=DEFAULT_SEMEVAL_ROOT):
    analogy_fn, choose_best = make_analogy_fns(embeddings,
                                               vocab_container,
                                               )

    output_folder = tempfile.mkdtemp()

    def semeval_path(suffix):
        return os.path.join(semeval_root, suffix)

    sets_by_folder = {
        # semeval_path('Training'): "10a 1a 2c 2h 3a 3c 4c 5d 5i 7a".split(),
        semeval_path('Testing'): "1b 1c 1d 1e 2a 2b 2d 2e 2f 2g 2i 2j 3b 3d 3e 3f 3g 3h 4a 4b 4d 4e 4f 4g 4h 5a 5b 5c 5e 5f 5g 5h 6a 6b 6c 6d 6e 6f 6g 6h 7b 7c 7d 7e 7f 7g 7h 8a 8b 8c 8d 8e 8f 8g 8h 9a 9b 9c 9d 9e 9f 9g 9h 9i 10b 10c 10d 10e 10f".split()
    }

    def nlm_scaled_path(suffix):
        return os.path.join(output_folder, 'ModelScaled-%s.txt' % suffix)

    def turker_scaled_path(suffix):
        return os.path.join(output_folder, 'TurkerScaled-%s.txt' % suffix)

    def spearman_results_path(suffix):
        return os.path.join(output_folder, 'SpearmanResults-%s.txt' % suffix)

    def maxdiff_scaled_path(suffix):
        return os.path.join(output_folder, 'MaxDiff-%s.txt' % suffix)

    def maxdiff_results_path(suffix):
        return os.path.join(output_folder, 'MaxDiffResults-%s.txt' % suffix)

    cat_names = category_names(os.path.join(semeval_root, 'subcategories-list.txt'))
    results = []
    for semeval_data_folder, sets_to_run in sets_by_folder.items():
        # only test on testing
        for s in sets_to_run:
            # print '-------'
            # print s
            data = {}
            data['key'] = s
            data['category'], data['name'] = cat_names[s]

            answer_file = os.path.join(semeval_data_folder, 'Phase1Answers', 'Phase1Answers-%s.txt' % s)
            question_file = os.path.join(semeval_data_folder, 'Phase1Questions', 'Phase1Questions-%s.txt' % s)

            paradigms = get_paradigms(question_file)
            examples = get_examples(answer_file)

            paradigm_analogies = [analogy_fn(*pair) for pair in paradigms]
            # print zip(paradigms, paradigm_analogies)
            average_paradigm = sum(paradigm_analogies) / len(paradigm_analogies)

            rankings = choose_best(average_paradigm, examples)
            data['number_paradigms'] = len(paradigms)
            data['number_testing'] = len(examples)

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

            data['rho'] = parse_correlation(spearman_results_path(s))
            data['accuracy'] = parse_accuracy(maxdiff_results_path(s))

            results.append(data)

    data = pandas.DataFrame(results).set_index('key')

    # compute average correlation
    # corrs = { filename:parse_correlation(filename) for filename in glob(spearman_results_path('*')) }
    # # compute average accuracy
    # accuracy = { filename:parse_accuracy(filename) for filename in glob(maxdiff_results_path('*')) }
    # assert np.abs(np.mean(corrs.values()) - data.rho.mean()) < 1e-5
    # assert np.abs(np.mean(accuracy.values()) - data.accuracy.mean()) < 1e-5

    return data.rho.mean(), data.accuracy.mean(), data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="model file to be used for semeval.py script")
    # parser.add_argument('--output_folder', help="folder to write results to", default="/home/dfried/code/nlm/semeval/junk")
    parser.add_argument('--semeval_root', help="folder containing semeval data", default=DEFAULT_SEMEVAL_ROOT)
    args = parser.parse_args()

    with gzip.open(args.model) as f:
        model = cPickle.load(f)

    vocab_container = get_vocab_container(model)

    mean_cor, mean_acc = run(model.embeddings, vocab_container, args.semeval_root)

    print 'average correlation: %f' % mean_cor
    print 'average accuracy: %f' % mean_acc
