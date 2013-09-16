import numpy as np
import itertools
from collections import defaultdict

class WordSimilarity(object):
    def __init__(self, vocabulary, cached_similarity_filename):
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
        self.word_pairwise_sims = np.load(cached_similarity_filename)
        r, c = self.word_pairwise_sims.shape
        if not (self.N == r == c):
            raise ValueError('size mismatch in vocabulary and pairwise similarity matrices: %d words in vocabulary, but pairwise sim matrix is %d by %d' % (self.N, r, c))
        self.word_pairwise_sims[np.isnan(self.word_pairwise_sims)] = -np.inf

    def most_similar_words(self, word, top_n=10):
        index = self.word_to_symbol[word]
        indices_and_sims = self.most_similar_indices(index, top_n)
        return [(self.vocabulary[i], sim)
                for (i, sim) in indices_and_sims]

    def most_similar_indices(self, index, top_n=10):
        sims = self.word_pairwise_sims[index]
        return [(i, sims[i])
                for i in itertools.islice(reversed(np.argsort(sims)), top_n)]
