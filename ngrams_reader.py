import gzip
from glob import glob
import numpy as np

DTYPE = np.uint32

def parse_line(line):
    token_string, count_string = line.strip().lower().split('\t')
    if token_string.startswith('"'):
        token_string = token_string[1:]
    if token_string.endswith('"'):
        token_string = token_string[:-1]
    token_string_dd = token_string.replace('""', '"')
    return token_string_dd.split(), int(count_string)

def line_iterator(glob_path):
    for fname in glob(glob_path):
        try:
            with gzip.open(fname) as f:
                for line in f:
                    yield line
        except IOError as e:
            print 'for file %s' % fname
            print e

import nltk

def build_frequency_counts(lines, frequency_counts=None):
    N = 0
    if frequency_counts is None:
        frequency_counts = nltk.FreqDist()
    for line in lines:
        N += 1
        tokens, count = parse_line(line)
        for token in tokens:
            frequency_counts[token] += count
    return frequency_counts, N

def build_id_map(frequency_counts, num_words=None):
    """given a FreqDist, return a dictionary mapping words to
    integer indices (by frequency), limited by num_words
    if given. if num_words is None, will map all words
    from the FreqDist"""
    # FreqDist.items() is sorted descending
    word_to_tokens = dict((word, i + 1)
                          for i, (word, count) in enumerate(frequency_counts.items()[:num_words]))
    # 0 is reserved for RARE, so return a default dict that maps unseen words
    # to the token 0
    from collections import defaultdict
    return defaultdict(int, word_to_tokens)

def matrix_rep(lines, frequency_counts, id_map):
    def line_to_row(line):
        tokens, count = parse_line(line)
        indices = [id_map[token] for token in tokens]
        return indices + [count]
    return np.array([line_to_row(line) for line in lines], dtype=DTYPE)

def dictionaries_for_files(glob_path, num_words=None):
    frequency_counts, N = build_frequency_counts(line_iterator(glob_path))
    id_map = build_id_map(frequency_counts, num_words=num_words)
    return frequency_counts, id_map, N

def ngrams_for_files(glob_path, frequency_counts, id_map):
    return matrix_rep(line_iterator(glob_path), frequency_counts, id_map)

DATA_BY_SIZE = ['/cl/nldata/books_google_ngrams_eng/5gms/5gm-100.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-10?.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-1??.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-*.gz']

def add_noise_to_symbols(symbols, vocab_size, column_index=None, rng=None):
    """symbols: a vector"""
    seq_length = symbols.shape[0]

    if rng is None:
        rng = np.random

    if column_index is None:
        column_index = seq_length / 2

    noisy = symbols.copy()
    noisy_mod = (noisy[column_index] + rng.randint(1, vocab_size)) % vocab_size
    noisy[column_index] = noisy_mod
    return noisy

class NGramsContainer(object):
    def __init__(self, glob_path, num_words=None):
        self.glob_path = glob_path
        self.num_words = num_words
        self.construct()

    def construct(self):
        self.frequency_counts, self.id_map, self.n_examples = dictionaries_for_files(self.glob_path, num_words=self.num_words)
        self.token_count_matrix = ngrams_for_files(self.glob_path, self.frequency_counts, self.id_map)
        n_examples, width = self.token_count_matrix.shape
        self.n_gram_length = width - 1
        # use max instead of length because defaultdict adds values on get,
        # even if not present in the dictionary
        self.vocab_size = max(self.id_map.values()) + 1 # add one for unseen word

    def average_sparsity(self):
        return (self.token_count_matrix[:, :-1] == 0).sum() / float(self.n_examples * self.n_gram_length)

    def get_data(self, rng=None, train_proportion=0.95, test_proportion=None):
        if rng is not None:
            view = self.token_count_matrix[rng.permutation(self.n_examples)]
        else:
            view = self.token_count_matrix

        if test_proportion is None:
            test_proportion = 1 - train_proportion

        num_train = int(train_proportion * self.n_examples)
        num_test = int(test_proportion * self.n_examples)
        train, test = view[:num_train], view[-num_test:]
        return train, test
