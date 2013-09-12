# coding: utf-8
import cPickle
import numpy as np
from scipy.spatial.distance import pdist, cosine
import gzip
import ngrams
from collections import defaultdict

DEFAULT_NGRAM_FILE='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5'

def load_zipped_pickle(pickle_fname):
    with gzip.open(pickle_fname, 'rb') as f:
        return cPickle.load(f)

def load_model_and_ngrams(model_path, ngram_file=DEFAULT_NGRAM_FILE):
    model = load_zipped_pickle(model_path)
    ngram_reader = ngrams.NgramReader(ngram_file, vocab_size=model.vocab_size)
    return model, ngram_reader

def maps(ngram_reader):
    id_map = defaultdict(int, dict((word, index) for (index, word) in enumerate(ngram_reader.word_array)))
    reverse_map = defaultdict(lambda x: 'RARE', dict((value, key) for (key, value) in id_map.items()))
    return id_map, reverse_map

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def make_analogy_fns(model, include_synsets=None, normalize_components=False):
    def analogy_fn(word1, word2):
        return model.get_embedding(word2, include_synsets, normalize_components) - \
                model.get_embedding(word1, include_synsets, normalize_components)
    def choose_best(reference_analogy, other_pairs):
        # reference_analogy = analogy_fn(word1, word2)
        other_analogies = [analogy_fn(w1, w2) for (w1, w2) in other_pairs]
        scores = [cosine_similarity(reference_analogy, other) for other in other_analogies]
        return list(reversed(sorted(zip(scores, other_pairs))))
    return analogy_fn, choose_best

def top_indices_from_distances(distances, reverse_map, n=10):
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:n]
    print top_indices
    return [(reverse_map[i], distances[i]) for i in top_indices]

def query(model, ngram_reader, word, n=10):
    # TODO update this to handle synset embeddings
    id_map, reverse_map = maps(ngram_reader)
    if word not in id_map:
        raise Exception('%s not in vocabulary' % word)
    index = id_map[word]
    embeddings = classifier.embedding_layer.embedding
    this_embedding = embeddings[index]
    distances = cdist(this_embedding[np.newaxis,:], embeddings, 'cosine').flatten()
    return top_indices_from_distances(distances, reverse_map, n=n)
