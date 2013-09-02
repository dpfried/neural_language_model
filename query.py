# coding: utf-8
import cPickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
import gzip
import ngrams
from collections import defaultdict

DEFAULT_NGRAM_FILE='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5'

def load_zipped_pickle(pickle_fname):
    with gzip.open(pickle_fname, 'rb') as f:
        return cPickle.load(f)

def load_classifier_and_ngrams(classifier_path, ngram_file=DEFAULT_NGRAM_FILE):
    classifier = load_zipped_pickle(classifier_path)
    ngram_reader = ngrams.NgramReader(ngram_file, vocab_size=classifier.vocab_size)
    return classifier, ngram_reader

def maps(ngram_reader):
    id_map = defaultdict(int, dict((word, index) for (index, word) in enumerate(ngram_reader.word_array)))
    reverse_map = defaultdict(lambda x: 'RARE', dict((value, key) for (key, value) in id_map.items()))
    return id_map, reverse_map

def cosine_similarity(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b,b))

def make_analogy_fns(classifier, ngram_reader):
    id_map, reverse_map = maps(ngram_reader)
    E = classifier.embedding_layer.embedding
    def analogy_fn(word1, word2):
        if word1 not in id_map:
            print "warning: %s not in vocabulary" % word1
        if word2 not in id_map:
            print "warning: %s not in vocabulary" % word2
        return E[id_map[word2]] - E[id_map[word1]]
    def choose_best(reference_analogy, other_pairs):
        # reference_analogy = analogy_fn(word1, word2)
        other_analogies = [analogy_fn(w1, w2) for (w1, w2) in other_pairs]
        scores = [cosine_similarity(reference_analogy, other) for other in other_analogies]
        return list(reversed(sorted(zip(scores, other_pairs))))
    return analogy_fn, choose_best

def make_query_fn(classifier, ngram_reader):
    embeddings = classifier.embedding_layer.embedding
    dist_matrix = squareform(pdist(embeddings, 'cosine'))
    id_map, reverse_map = maps(ngram_reader)
    reverse_map[0] = 'RARE'

    def query(word, n=10):
        if word not in id_map:
            raise Exception('%s not in vocabulary' % word)
        index = id_map[word]

        sorted_indices = np.argsort(dist_matrix[index,:])
        top_indices = sorted_indices[:n]
        print top_indices
        return [(reverse_map[i], dist_matrix[index,i]) for i in top_indices]
    return query
