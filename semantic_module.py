import numpy as np
import itertools
from collections import defaultdict
import os

MEMMAP_DTYPE = 'float64'

def fill_nan(arr):
    arr[np.isnan(arr)] = -np.inf

def memmapped_pairwise_sims(vocab_size, cached_similarity_filename, memmap_filename):
    """
    if memmap_filename is None, load cached_similarity_filename
    if memmap_filename exists, load it as a memmap
    if memmap_filename doesn't exist, load cached_similarity_filename, write it to memmap_filename, then load memmap_filename as a memmap
    """
    if not memmap_filename:
        arr =  np.load(cached_similarity_filename)
        fill_nan(arr)
    elif os.path.exists(memmap_filename):
        return np.memmap(memmap_filename, dtype=MEMMAP_DTYPE, mode='r', shape=(vocab_size, vocab_size))
    else:
        sims = np.load(cached_similarity_filename)
        fill_nan(sims)
        mmap = np.memmap(memmap_filename, dtype=MEMMAP_DTYPE, mode='w+', shape=(vocab_size, vocab_size))
        mmap[:] = sims[:]
        del sims
        mmap.flush()
        return mmap

class WordSimilarity(object):
    def __init__(self, vocabulary, cached_similarity_filename, memmap_filename=None):
        """
        vocabulary: a list of words

        cached_similarity_filename: a file containing a pickled numpy matrix with pairwise similarity values (between -1 and 1) for
        each word in the vocabulary list. Can be created through dump_wordnet_similarities
        """

        self.vocabulary = vocabulary
        self.symbol_to_word = defaultdict(lambda : '*UNKNOWN*', dict(enumerate(self.vocabulary)))
        self.symbol_to_word[0] = '*UNKNOWN*'
        self.word_to_symbol = defaultdict(int, dict((word, index) for index, word in enumerate(self.vocabulary)))
        self.N = len(self.vocabulary)
        self.word_pairwise_sims = memmapped_pairwise_sims(self.N, cached_similarity_filename, memmap_filename)
        r, c = self.word_pairwise_sims.shape
        if not (self.N == r == c):
            raise ValueError('size mismatch in vocabulary and pairwise similarity matrices: %d words in vocabulary, but pairwise sim matrix is %d by %d' % (self.N, r, c))

    def most_similar_words(self, word, top_n=10):
        index = self.word_to_symbol[word]
        indices_and_sims = self.most_similar_indices(index, top_n)
        return [(self.vocabulary[i], sim)
                for (i, sim) in indices_and_sims]

    def most_similar_indices(self, index, top_n=10):
        sims = self.word_pairwise_sims[index]
        return [(i, sims[i])
                for i in itertools.islice(reversed(np.argsort(sims)), top_n)]
