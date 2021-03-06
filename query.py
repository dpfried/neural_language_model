import numpy as np
from scipy.spatial.distance import cdist, cosine
import ngrams
from os.path import join
import gzip, cPickle

from config import DEFAULT_NGRAM_FILENAME

def get_vocab_container(model):
    try:
        ngram_filename = model.other_params['ngram_filename']
    except:
        ngram_filename = DEFAULT_NGRAM_FILENAME
    try:
        vocab_size = model.other_params['vocab_size']
    except:
        vocab_size = 50000
    return ngrams.NgramReader(ngram_filename, vocab_size=vocab_size)

## TODO: dedup this
def get_vocab(model):
    try:
        base_dir = model.other_params['base_dir']
        vocabulary_path = join(base_dir, 'vocabulary.pkl.gz')
        return load_model(vocabulary_path)
    except:
        try:
            ngram_filename = model.other_params['ngram_filename']
        except:
            ngram_filename = DEFAULT_NGRAM_FILENAME
        try:
            vocab_size = model.other_params['vocab_size']
        except:
            vocab_size = 50000
        return ngrams.NgramReader(ngram_filename, vocab_size=vocab_size)


def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def make_analogy_fns(embeddings, vocab_container):
    def index(word):
        return vocab_container.symbol_to_index[word]
    def analogy_fn(word1, word2):
        return embeddings[index(word2.lower())] - embeddings[index(word1.lower())]
    def choose_best(reference_analogy, other_pairs):
        # reference_analogy = analogy_fn(word1, word2)
        other_analogies = [analogy_fn(w1.lower(), w2.lower()) for (w1, w2) in other_pairs]
        scores = np.nan_to_num([cosine_similarity(reference_analogy, other) for other in other_analogies])
        return list(reversed(sorted(zip(scores, other_pairs))))
    return analogy_fn, choose_best

def top_indices_from_distances(distances, index_to_symbol, n=10):
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:n]
    print top_indices
    return [(index_to_symbol[i], distances[i]) for i in top_indices]

def query_embeddings(embeddings, vocab_container, word, n=10):
    index = vocab_container.symbol_to_index[word]
    this_embedding = embeddings[index]
    distances = cdist(this_embedding[np.newaxis,:], embeddings, 'cosine').flatten()
    return top_indices_from_distances(distances, vocab_container.index_to_symbol, n=n)

def query(model, vocab_container, word, n=10):
    embeddings = model.embeddings
    return query_embeddings(embeddings, vocab_container, word, n=n)

def averaged_query(model, vocab_container, word, n=10):
    embeddings = model.averaged_embeddings()
    return query_embeddings(embeddings, vocab_container, word, n=n)

def get_model(path, number=1000):
    with gzip.open(join(path, 'model-%s.pkl.gz' % number)) as f:
        return cPickle.load(f)

def get_rels(path):
    with gzip.open(join(path, 'relationships.pkl.gz')) as f:
        return cPickle.load(f)
