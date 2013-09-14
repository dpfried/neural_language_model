from wordnet_similarity import WunschPaths, make_similarity_matrix, scaled_lch_similarity
from ngrams import NgramReader
from nltk.corpus import wordnet as wn
import numpy as np
import gzip, cPickle
from functools import partial


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='filename to dump the similarity matrix to')
    parser.add_argument('--vocab_size', type=int, default=50000, help='size of the vocabulary')
    parser.add_argument('--wunsch_paths', help='zipped pickle of the WunschPaths, precomputed')
    parser.add_argument('--ngram_file', default='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5')
    parser.add_argument('--reduction_function', default='max', help='"max" or "mean"')
    args = parser.parse_args()

    if args.reduction_function == 'max':
        reduction_fn = np.max
    elif args.reduction_function == 'mean':
        reduction_fn = np.mean
    else:
        print 'unknown function %s, using np.max' % args.reduction_function
        reduction_fn = np.max
    reader = NgramReader(args.ngram_file)
    if args.wunsch_paths:
        with gzip.open(args.wunsch_paths) as f:
            paths = cPickle.load(f)
    else:
        paths = WunschPaths(wn.all_synsets())

    sim_fn = partial(scaled_lch_similarity, paths)
    similarity_matrix = make_similarity_matrix(reader.word_array[:args.vocab_size],
                                                similarity_fn=sim_fn,
                                                reduction_fn=reduction_fn)

    print 'writing to file %s' % args.filename
    with open(args.filename, 'w') as f:
        np.save(args.filename, similarity_matrix)
    print 'done'
