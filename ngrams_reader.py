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

def build_id_map(frequency_counts):
    return dict((word, i) for i, word in enumerate(frequency_counts.keys()))

def matrix_rep(lines, frequency_counts=None, id_map=None):
    if frequency_counts is None:
        lines = list(lines)
        frequency_counts = build_frequency_counts(lines)
    if id_map is None:
        id_map = build_id_map(frequency_counts)
    def line_to_row(line):
        tokens, count = parse_line(line)
        indices = [id_map[token] for token in tokens]
        return indices + [count]
    return np.array([line_to_row(line) for line in lines], dtype=DTYPE), frequency_counts, id_map

def read_data(glob_path):
    frequency_counts = build_frequency_counts(line_iterator(glob_path))
    id_map = build_id_map(frequency_counts)
    return matrix_rep(line_iterator(glob_path), frequency_counts, id_map)


DATA_BY_SIZE = ['/cl/nldata/books_google_ngrams_eng/5gms/5gm-100.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-10?.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-1??.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-*.gz']

def add_noise_to_column(train_X, vocab_size, column_index=None, rng=None):
    n_examples, seq_length = train_X.shape
    if rng is None:
        rng = np.random
    noise_to_add = rng.randint(1, vocab_size, size=n_examples).astype(DTYPE)
    if column_index is None:
        column_index = seq_length / 2
    return np.column_stack([train_X[:, : column_index], train_X[:, column_index] + noise_to_add, train_X[:, column_index + 1 :]])

class NGramsContainer(object):
    def __init__(self, glob_path):
        self.glob_path = glob_path
        self.construct()

    def construct(self):
        self.token_count_matrix, self.frequency_counts, self.id_map = read_data(self.glob_path)
        n_examples, width = self.token_count_matrix.shape
        self.n_examples = n_examples
        self.n_gram_length = width - 1
        self.vocab_size = len(self.frequency_counts)

    def get_data(self, rng=None, train_proportion=0.95, nonsense_generator=add_noise_to_column):
        if rng is not None:
            view = self.token_count_matrix[rng.permutation(self.n_examples)]
        else:
            view = self.token_count_matrix

        num_train = int(train_proportion * self.n_examples)
        train, test = view[:num_train], view[-num_train:]
        # exclude the counts
        train_X = train[:, 0:-1]
        train_Y = np.array([1] *self.n_examples)

        test_X = nonsense_generator(train_X, self.vocab_size, rng=rng)
        test_Y = np.array([0] * self.n_examples)

        return [(train_X, train_Y), (test_X, test_Y)]
