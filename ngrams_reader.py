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
    return (line for fname in glob(glob_path) for line in gzip.open(fname))

import nltk

def build_frequency_counts(lines):
    frequency_counts = nltk.FreqDist()
    for line in lines:
        tokens, count = parse_line(line)
        for token in tokens:
            frequency_counts[token] += count
    return frequency_counts

def build_id_map(frequency_counts, num_words=None):
    """given a FreqDist, return a dictinoary mapping words to
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

def matrix_rep(lines, frequency_counts=None, id_map=None, num_words=None):
    if frequency_counts is None:
        lines = list(lines)
        frequency_counts = build_frequency_counts(lines)
    if id_map is None:
        id_map = build_id_map(frequency_counts, num_words=num_words)
    def line_to_row(line):
        tokens, count = parse_line(line)
        indices = [id_map[token] for token in tokens]
        return indices + [count]
    return np.array([line_to_row(line) for line in lines], dtype=DTYPE), frequency_counts, id_map

def read_data(glob_path, num_words=None):
    frequency_counts = build_frequency_counts(line_iterator(glob_path))
    id_map = build_id_map(frequency_counts, num_words=num_words)
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
    noisy[column_index] += rng.randint(1, vocab_size)
    return np.mod(noisy, vocab_size)

def add_noise_to_column(train_X, vocab_size, column_index=None, rng=None):
    train_X = np.array(train_X)
    if train_X.ndim > 1:
        n_examples, seq_length = train_X.shape
    else:
        n_examples, seq_length = 1, len(train_X)
        # make this vector into the row of a one-row matrix
        train_X = train_X[None, :]
    if rng is None:
        rng = np.random

    if column_index is None:
        column_index = seq_length / 2

    # replace the word in column_index with a different word by adding a
    # random offset to it, modulo the size of the vocabulary
    noise_to_add = rng.randint(1, vocab_size, size=n_examples).astype(DTYPE)
    noisy_column = np.mod(train_X[:, column_index] + noise_to_add, vocab_size)

    # substitute this column back into the data matrix by appending the columns
    # together.
    # TODO: Does this mean memory is shared?
    return np.column_stack([train_X[:, : column_index], noisy_column, train_X[:, column_index + 1 :]])

class NGramsContainer(object):
    def __init__(self, glob_path, num_words=None):
        self.glob_path = glob_path
        self.num_words = num_words
        self.construct()

    def construct(self):
        self.token_count_matrix, self.frequency_counts, id_map = read_data(self.glob_path, num_words=self.num_words)
        self.id_map = id_map
        n_examples, width = self.token_count_matrix.shape
        self.n_examples = n_examples
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
        # exclude the counts which are the last column
        train, test = view[:num_train, :-1], view[-num_test:, :-1]
        return train, test
