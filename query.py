# coding: utf-8
import cPickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
import gzip
import ngrams

def load_zipped_pickle(pickle_fname):
    with gzip.open(pickle_fname, 'rb') as f:
        return cPickle.load(f)

def make_query_fn(classifier_path, ngram_file='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5'):
    ngram_reader = ngrams.NgramReader(ngram_file)
    classifier = load_zipped_pickle(classifier_path)
    embeddings = classifier.embedding_layer.embedding
    dist_matrix = squareform(pdist(embeddings, 'euclidean'))
    id_map = dict((word, index) for (index, word) in enumerate(ngram_reader.word_array))
    reverse_map = dict((value, key) for (key, value) in id_map.items())
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
