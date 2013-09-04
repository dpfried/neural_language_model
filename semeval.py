# coding: utf-8
import numpy as np

from query import load_classifier_and_ngrams,  make_analogy_fns

def questions_from_answers(answer_file):
    return answer_file.replace('Answers', 'Questions')

def get_paradigms(question_file):
    from itertools import islice
    with open(question_file) as f:
        lines = list(islice(f, 4, 8))
        if not(lines[-1].strip()): # possibly only 3 golden examples
            lines = lines[:-1]
        pairs = [line.strip().split(':') for line in lines]
    return pairs

def get_examples(answer_file):
    with open(answer_file) as f:
        return [line.strip().strip('"').split(':') for line in f]

if __name__ == "__main__":
    import sys
    try:
        answer_file = sys.argv[1]
        output_file = sys.argv[2]
    except:
        print 'usage: %s <answer_file>' % sys.argv[0]
        sys.exit(1)
    question_file = questions_from_answers(answer_file)
    paradigms = get_paradigms(question_file)
    examples = get_examples(answer_file)

    classifier, ngram_reader = load_classifier_and_ngrams('/home/dfried/code/nlm/new_sample/model-1.pkl.gz')
    analogy_fn, choose_best = make_analogy_fns(classifier, ngram_reader)

    paradigm_analogies = [analogy_fn(*pair) for pair in paradigms]
    print zip(paradigms, paradigm_analogies)
    average_paradigm = sum(paradigm_analogies) / len(paradigm_analogies)

    rankings = choose_best(average_paradigm, examples)
    min_score = min(score for score, pair in rankings)
    print rankings
    with open(output_file, 'w') as f:
        for score, (w1, w2) in rankings:
            f.write('%0.2f "%s:%s"\n' % ((score - min_score)  * 100 + 1, w1, w2))
