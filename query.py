# coding: utf-8
import cPickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
import gzip

def load_zipped_pickle(pickle_fname):
    with gzip.open(pickle_fname, 'rb') as f:
        return cPickle.load(f)

def make_query_fn(classifier_path, n_gram_path='data/n_grams.pkl.gz'):
    n_grams = load_zipped_pickle(n_gram_path)
    classifier = load_zipped_pickle(classifier_path)
    embeddings = classifier.embedding_layer.embedding
    dist_matrix = squareform(pdist(embeddings, 'euclidean'))
    id_map = n_grams.id_map
    reverse_map = dict((value, key) for (key, value) in id_map.items())
    reverse_map[0] = 'RARE'

    def query(word, n=10):
        if word not in n_grams.id_map:
            raise Exception('%s not in vocabulary' % word)
        index = id_map[word]

        sorted_indices = np.argsort(dist_matrix[index,:])
        top_indices = sorted_indices[:n]
        print top_indices
        return [(reverse_map[i], dist_matrix[index,i]) for i in top_indices]
    return query
