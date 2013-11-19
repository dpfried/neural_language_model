import math
import h5py
import numpy as np
from collections import defaultdict

from utils import sample_cumulative_discrete_distribution
UNKNOWN_WORD = '*UNKNOWN*'

class NgramReader(object):
    def __init__(self, filename, ngram_length=5, vocab_size=None, train_proportion=0.95, test_proportion=None):
        self.hd5_file = h5py.File(filename, 'r')
        self.ngram_length = ngram_length
        self.vocab_size = vocab_size
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion or (1 - train_proportion)

        # map the datasets in the file to attributes of the class
        if self.vocab_size:
            self.word_array = self.hd5_file['words'][:vocab_size]
            self.word_frequencies = self.hd5_file['word_frequencies'][:vocab_size]
        else:
            self.word_array = self.hd5_file['words']
            self.word_frequencies = self.hd5_file['word_frequencies']
        self.word_array[0] = UNKNOWN_WORD

        self.cumulative_word_frequencies = np.cumsum(self.word_frequencies)
        self.ngrams = self.hd5_file['%i_grams' % ngram_length]

        self.number_of_ngrams, cols = self.ngrams.shape

        if ngram_length != cols - 1:
            print 'size mismatch: ngram_length %i but ngram matrix has %i columns' % (ngram_length, cols - 1)

        # build a dictionary of words to indices for the top {vocab_size} words
        # in the dataset
        self.symbol_to_index = defaultdict(int,
                                           dict((word, index) for index, word in enumerate(self.word_array)))
        self.index_to_symbol = defaultdict(lambda : UNKNOWN_WORD,
                                           dict((index, word) for index, word in enumerate(self.word_array)))

    def process_block(self, ngram_matrix):
        """
        an ngram_matrix is a (instances x (n + 1)) matrix containing integers
        which designate the tokens in each of the n words of each n gram. The
        final column (at index n) is the frequency of that n-gram.

        This function returns a matrix which has those tokens which are above
        the vocabulary size replaced by zeros. The frequency column is retained,
        and an additional column, which is the cumulative sum of the
        frequency column within this block, to be used in ngram sampling
        """
        rows, cols = ngram_matrix.shape
        new_block = np.empty((rows, cols + 1), dtype=np.uint64)
        zero_mask = ngram_matrix[:, :-1] >= self.vocab_size
        copy_mask = np.logical_not(zero_mask)
        new_block[zero_mask] = 0
        new_block[copy_mask] = ngram_matrix[copy_mask]
        new_block[:,-2] = ngram_matrix[:, -1]
        new_block[:,-1] = np.cumsum(ngram_matrix[:, -1])
        return new_block

    def testing_block(self):
        num_rows = int(self.test_proportion * self.number_of_ngrams)
        return self.process_block(self.ngrams[-num_rows:])

    def training_block(self, fraction, block_size=100000):
        start_row = 0
        end_row = int(self.train_proportion * self.number_of_ngrams)
        num_normal_blocks = (end_row - start_row) / block_size
        leftover_rows = (end_row - start_row) % block_size
        main_part_fraction = 1 - float(leftover_rows) / (end_row - start_row)
        if fraction < main_part_fraction:
            index = int(math.floor(fraction / main_part_fraction * num_normal_blocks))
            first_index = int(block_size * index)
            last_index = int(block_size * (index + 1))
            block = self.ngrams[start_row + first_index : start_row + last_index]
        else:
            block = self.ngrams[end_row - leftover_rows : end_row]
        return self.process_block(block)

    def to_words(self, ngram_matrix):
        return [[self.word_array[index] for index in tokens[:-1]] + [tokens[-1]]
                for tokens in self.zero_out_rare_words(ngram_matrix)]

    def add_noise_to_symbols(self, symbols, column_index=None, rng=None, max_tries=5):
        seq_length = symbols.shape[0]

        if column_index is None:
            column_index = seq_length / 2

        tries = 0
        replacement_word = symbols[column_index]
        while tries < max_tries:
            tries += 1
            replacement_word = sample_cumulative_discrete_distribution(self.cumulative_word_frequencies)
            if replacement_word != 0 and replacement_word != symbols[column_index]:
                break
        assert replacement_word < self.vocab_size

        noisy = symbols.copy()
        noisy[column_index] = replacement_word
        return noisy

    def contrastive_symbols_from_row(self, row, replacement_column_index=None, rng=None):
        if replacement_column_index is None:
            replacement_column_index = self.ngram_length / 2
        """
        given a row of an ngram matrix reprsenting a training ngram, return a
        list of the symbols corresponding to words in that ngram, a corrupted
        list of the same symbols, and the frequency of the original ngram in
        the training corpus
        """
        # last two columns are reserved for frequency of ngram and cumulative
        # frequency, respectively
        correct_symbols = row[:-2]
        ngram_frequency = row[-2]
        # get a list of symbols representing a corrupted ngram
        error_symbols = self.add_noise_to_symbols(correct_symbols, column_index=replacement_column_index, rng=rng)
        return correct_symbols, error_symbols, ngram_frequency
