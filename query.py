import numpy as np
from scipy.spatial.distance import cdist, cosine
import ngrams

DEFAULT_NGRAM_FILE='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5'

def get_vocab_container(model):
    try:
        ngram_filename = model.other_params['ngram_filename']
    except:
        ngram_filename = DEFAULT_NGRAM_FILE
    try:
        vocab_size = model.other_params['vocab_size']
    except:
        vocab_size = 50000
    return ngrams.NgramReader(ngram_filename, vocab_size=vocab_size)

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def make_analogy_fns(model, vocab_container):
    def index(word):
        return vocab_container.symbol_to_index[word]
    def analogy_fn(word1, word2):
        return model.embeddings[index(word2.lower())] - model.embeddings[index(word1.lower())]
    def choose_best(reference_analogy, other_pairs):
        # reference_analogy = analogy_fn(word1, word2)
        other_analogies = [analogy_fn(w1.lower(), w2.lower()) for (w1, w2) in other_pairs]
        scores = [cosine_similarity(reference_analogy, other) for other in other_analogies]
        return list(reversed(sorted(zip(scores, other_pairs))))
    return analogy_fn, choose_best

def top_indices_from_distances(distances, index_to_symbol, n=10):
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:n]
    print top_indices
    return [(index_to_symbol[i], distances[i]) for i in top_indices]

def query(model, vocab_container, word, n=10):
    index = vocab_container.symbol_to_index[word]
    embeddings = model.embeddings
    this_embedding = embeddings[index]
    distances = cdist(this_embedding[np.newaxis,:], embeddings, 'cosine').flatten()
    return top_indices_from_distances(distances, vocab_container.index_to_symbol, n=n)
