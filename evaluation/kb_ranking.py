import models
import os
import ngrams
from relational.wordnet_rels import SynsetToWord
import numpy as np
import gzip
import cPickle
import sys
from query import get_vocab_container
from nltk.corpus import wordnet as wn
import pandas

def make_testing_data(model, relationships=None, vocabulary=None, s2w=None, pct=0.98):
    if relationships is None:
        rel_path = os.path.join(model.other_params['base_dir'], 'relationships.pkl.gz')
        with gzip.open(rel_path) as f:
            relationships = cPickle.load(f)

    if vocabulary is None:
        reader = ngrams.NgramReader(model.other_params['ngram_filename'], model.other_params['vocab_size'])
        vocabulary = reader.word_array

    if s2w is None:
        s2w = SynsetToWord(vocabulary)

    training_size = int(relationships.N) * pct

    def usable_row(symbolic_row):
        sa, sb, rel = symbolic_row
        return s2w.usable(sa) and s2w.usable(sb)

    testing_data = filter(usable_row, map(relationships.indices_to_symbolic, relationships.data[training_size:]))
    return testing_data

def score_model(model, vocabulary, s2w, relationships, symbolic_testing_data):
    # replace model v embeddings with averaged embeddings
    # model.v_trainer.embedding_layer.embedding.set_value(model.averaged_embeddings())

    scores = []
    vocab_size = len(vocabulary)
    correct_synsets = []
    correct_indices = []

    # compute scorable synsets
    synsets_with_rep = filter(s2w.usable, wn.all_synsets())
    N = len(symbolic_testing_data)
    word_embeddings = model.averaged_embeddings()
    def embed_synset(syn):
        return word_embeddings[s2w.words_by_synset[syn]].mean(axis=0)
    synset_embeddings = [embed_synset(syn) for syn in synsets_with_rep]
    for datum_index, (s1, s2, rel) in enumerate(symbolic_testing_data):
        if s1 not in synsets_with_rep or s2 not in synsets_with_rep:
            print
            print 'failed sanity check for %s, %s' % (s1, s2)
        sys.stdout.write('\r%d / %d' % (datum_index, N))
        sys.stdout.flush()

        correct_synsets.append(s2)

        s1_embedding = embed_synset(s1)

        rel_index = relationships.relationships.index(rel)

        scores_for_row = []

        for i, (trial_synset, trial_embedding) in enumerate(zip(synsets_with_rep, synset_embeddings)):
            if trial_synset == s2:
                correct_indices.append(i)
            scores_for_row.append(model.v_trainer.score_embeddings(s1_embedding, trial_embedding, rel_index))
        scores.append(scores_for_row)
    print
    return np.array(scores), correct_synsets, correct_indices

def score_socher_set(model, socher_set, admm=True):
    if admm:
        # word_embeddings = model.averaged_embeddings()
        # score_fn = model.v_trainer.score_embeddings
        word_embeddings = model.v_trainer.embeddings
        score_fn = model.v_trainer.score_embeddings
    else:
        word_embeddings = model.embeddings
        score_fn = model.score_embeddings
    def average_indices(indices):
        # return word_embeddings[indices][0]
        return np.nan_to_num(word_embeddings[ indices ].mean(axis=0))
    def score_row(row):
        indices_1, rel_index, indices_2 , label = row
        embedding_1 = average_indices(indices_1)
        embedding_2 = average_indices(indices_2)
        return float(score_fn(embedding_1, embedding_2, rel_index))
    data = [list(row) + [score_row(row)] for row in socher_set]
    return pandas.DataFrame(data,columns=['indices_a', 'rel', 'indices_b', 'label', 'score'])

def classify(rows, threshold):
    return rows.score.map(lambda f: 1 if f >= threshold else -1)

def num_correct(rows, threshold):
    predicted = classify(rows, threshold)
    return (rows.label == predicted).sum()

def find_thresholds(scored_set):
    thresholds = {}
    for rel in set(scored_set.rel):
        this_rel_data = scored_set[scored_set.rel == rel]
        minv = this_rel_data.score.min()
        maxv = this_rel_data.score.max()
        val = max(np.arange(minv, maxv, (maxv - minv) / 1000), key = lambda f: num_correct(this_rel_data, f).sum())
        thresholds[rel] = val
    return thresholds

def accuracy(scored_set, thresholds):
    acc = {}
    total_correct = 0
    total_N = 0
    for rel in thresholds:
        this_rel_data = scored_set[scored_set.rel == rel]
        N = len(this_rel_data.score)
        total_N += N
        correct = num_correct(this_rel_data, thresholds[rel])
        total_correct += correct
        acc[rel] = float(correct) / N
    return acc, float(total_correct) / total_N

def ranks(scores, correct_indices):
    N, M = scores.shape
    return [M - np.where(np.argsort(scores[n]) == correct_indices[n])[0][0] for n in xrange(N)]

def test_socher(model, rels):
    admm = hasattr(model, 'v_trainer')
    print 'scoring set'
    scores = score_socher_set(model, rels.dev_words, admm=admm)
    print 'calculating thresholds'
    thresholds = find_thresholds(scores)
    print 'calculating accuracy'
    return accuracy(score_socher_set(model, rels.test_words,admm=admm), thresholds)

def test(model):
    relationship_path = os.path.join(model.other_params['base_dir'], 'relationships.pkl.gz')
    with gzip.open(relationship_path) as f:
        relationships = cPickle.load(f)

    vocabulary = get_vocab_container(model).word_array

    s2w = SynsetToWord(vocabulary)

    testing_data = make_testing_data(model, relationships, vocabulary, s2w)

    scores, correct_synsets, correct_indices = score_model(model, vocabulary, s2w, relationships, testing_data)

    return ranks(scores, correct_indices)
