import numpy as np
import h5py
from ngrams_reader import dictionaries_for_files, ngrams_for_files, DTYPE, DATA_BY_SIZE
from glob import glob
import sys
from collections import defaultdict
import math
from utils import sample_cumulative_discrete_distribution

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
            self.word_frequencies = self.hd5_file

        self.cumulative_word_frequencies = np.cumsum(self.word_frequencies)
        self.ngrams = self.hd5_file['%i_grams' % ngram_length]

        self.number_of_ngrams, cols = self.ngrams.shape

        if ngram_length != cols - 1:
            print 'size mismatch: ngram_length %i but ngram matrix has %i columns' % (ngram_length, cols - 1)

        # build a dictionary of words to indices for the top {vocab_size} words
        # in the dataset
        self.id_map = defaultdict(int, dict((word, index) for index, word in enumerate(self.word_array)))

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
        zero_mask = ngram_matrix[:, :-1] > self.vocab_size
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
        assert replacement_word <= self.vocab_size

        noisy = symbols.copy()
        noisy[column_index] = replacement_word
        return noisy

def build_hd5(hd5_filename, ngram_file_glob, ngram_length=5, row_chunksize=10000):
    print '...first pass: building frequency counts and word to token map for files in %s' % ngram_file_glob
    frequency_counts, id_map, n_rows = dictionaries_for_files(ngram_file_glob)
    print '...creating hd5 file'

    hd5_file = h5py.File(hd5_filename, 'w')
    hd5_file.attrs['contained_files'] = ngram_file_glob

    cols = ngram_length + 1

    words, frequencies = zip(*frequency_counts.items())

    word_array = np.array(['RARE'] + list(words))
    frequency_array = np.array([0] + list(frequencies))

    hd5_file.create_dataset("words", data = word_array)
    hd5_file.create_dataset("word_frequencies", data=frequency_array)

    hd5_file.flush()

    ngram_dset = hd5_file.create_dataset("%d_grams" % ngram_length,
                                   shape=(n_rows, cols),
                                   dtype=DTYPE,
                                   maxshape=(None, cols),
                                   chunks=(row_chunksize, cols))

    fnames = glob(ngram_file_glob)
    read_rows = 0
    flush_every=20
    for file_count, filename in enumerate(glob(ngram_file_glob)):
        sys.stdout.write('reading file %i / %i (%s)\r' % (file_count, len(fnames), filename))
        sys.stdout.flush()
        try:
            ngram_mat = ngrams_for_files(filename, frequency_counts=frequency_counts, id_map=id_map)
            this_rows, _ = ngram_mat.shape
            ngram_dset[read_rows:read_rows + this_rows, :] = ngram_mat
            read_rows += this_rows
            if (file_count + 1) % flush_every == 0:
                hd5_file.flush()
        except IOError as e:
            print 'for file %s' % filename
            print e

    print
    print "complete.. writing file"
    hd5_file.close()

if __name__ == "__main__":
    import time
    for i, file_glob in enumerate(DATA_BY_SIZE):
        print 'size %i' % i
        old_time = time.clock()
        build_hd5('/cl/nldata/books_google_ngrams_eng/5grams_size%i.hd5' % i, file_glob)
        print 'elapsed time: %i seconds' % (time.clock() - old_time)
        print
