import sys

import gzip
import h5py
import nltk
import numpy as np
from collections import defaultdict
from glob import glob

DTYPE = np.uint32

DATA_BY_SIZE = ['/cl/nldata/books_google_ngrams_eng/5gms/5gm-100.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-10?.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-1??.gz',
                '/cl/nldata/books_google_ngrams_eng/5gms/5gm-*.gz']

def line_iterator(glob_path):
    for fname in glob(glob_path):
        try:
            with gzip.open(fname) as f:
                for line in f:
                    yield line
        except IOError as e:
            print 'for file %s' % fname
            print e

def parse_line(line):
    token_string, count_string = line.strip().lower().split('\t')
    if token_string.startswith('"'):
        token_string = token_string[1:]
    if token_string.endswith('"'):
        token_string = token_string[:-1]
    token_string_dd = token_string.replace('""', '"')
    return token_string_dd.split(), int(count_string)

def ngrams_for_files(glob_path, frequency_counts, id_map):
    return matrix_rep(line_iterator(glob_path), frequency_counts, id_map)

def matrix_rep(lines, frequency_counts, id_map):
    def line_to_row(line):
        tokens, count = parse_line(line)
        indices = [id_map[token] for token in tokens]
        return indices + [count]
    return np.array([line_to_row(line) for line in lines], dtype=DTYPE)

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
    return defaultdict(int, word_to_tokens)


def dictionaries_for_files(glob_path, num_words=None):
    frequency_counts, N = build_frequency_counts(line_iterator(glob_path))
    id_map = build_id_map(frequency_counts, num_words=num_words)
    return frequency_counts, id_map, N

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
