import models
import os
import ngrams
from relational.synset_to_word import SynsetToWord
import numpy as np
import gzip
import cPickle
import sys
from query import get_vocab_container

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
    model.v_trainer.embedding_layer.embedding.set_value(model.averaged_embeddings())

    scores = []
    vocab_size = len(vocabulary)
    correct_synsets = []
    correct_indices = []
    N = len(symbolic_testing_data)
    for datum_index, (s1, s2, rel) in enumerate(symbolic_testing_data):
        sys.stdout.write('\r%d / %d' % (datum_index, N))
        sys.stdout.flush()
        w1 = s2w.words_by_synset[s1][0] # get the index of the most common word in the synset (contained in the vocab)
        rel_index = relationships.relationships.index(rel)
        scores_for_row = []
        correct_indices.append(s2w.words_by_synset[s2][0])
        correct_synsets.append(s2)
        for w2 in xrange(vocab_size):
            scores_for_row.append(model.v_trainer.score(w1, w2, rel_index))
        scores.append(scores_for_row)
    print
    return np.array(scores), correct_synsets, correct_indices

def ranks(scores, correct_indices):
    N, M = scores.shape
    return [M - np.where(np.argsort(scores[n]) == correct_indices[n])[0][0] for n in xrange(N)]

def test(model):
    relationship_path = os.path.join(model.other_params['base_dir'], 'relationships.pkl.gz')
    with gzip.open(relationship_path) as f:
        relationships = cPickle.load(f)

    vocabulary = get_vocab_container(model).word_array

    s2w = SynsetToWord(vocabulary)

    testing_data = make_testing_data(model, relationships, vocabulary, s2w)

    scores, correct_synsets, correct_indices = score_model(model, vocabulary, s2w, relationships, testing_data)

    return ranks(scores, correct_indices)