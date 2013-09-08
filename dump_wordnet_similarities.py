from wordnet_similarity import WunschPaths, safe_similarity_wrapper, lch_similarity, make_similarity_matrix
from ngrams import NgramReader
from nltk.corpus import wordnet as wn
import numpy as np


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='filename to dump the similarity matrix to')
    parser.add_argument('--vocab_size', type=int, default=50000, help='size of the vocabulary')
    parser.add_argument('--ngram_file', default='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5')
    parser.add_argument('--reduction_function', default='max', help='"max" or "mean"')
    args = parser.parse_args()

    try:
        if args.reduction_function == 'max':
            reduction_fn = np.max
        elif args.reduction_function == 'mean':
            reduction_fn = np.mean
        else:
            print 'unknown function %s, using np.max' % args.reduction_function
            reduction_fn = np.max
        reader = NgramReader(args.ngram_file)
        paths = WunschPaths(wn.all_synsets())
        safe_sim = safe_similarity_wrapper(lambda x, y: lch_similarity(paths, x, y))
        # safe_sim = safe_similarity_wrapper(lambda x, y: x.lch_similarity(y))
        similarity_matrix = make_similarity_matrix(reader.word_array[:args.vocab_size], similarity_fn=safe_sim, reduction_fn=reduction_fn)
        print 'writing to file %s' % args.filename
        with open(args.filename, 'w') as f:
            np.save(args.filename, similarity_matrix)
        print 'done'
    finally:
        pass
    #     try:
    #         import IPython
    #         IPython.embed()
    #     except:
    #         import code
    #         code.interact(local=locals())
