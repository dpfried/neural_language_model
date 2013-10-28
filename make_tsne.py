from calc_tsne import calc_tsne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from relational.relational_admm import *
import gzip
import cPickle
from config import DEFAULT_NGRAM_FILENAME

import fix_imports

def do_plot(embedding_layer, words, start=0, end=100):
    # plot 3d
    X = calc_tsne(embedding_layer[start:end], 3)
    words = words[start:end]
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2])
    for i in xrange(end-start):
        try:
            text = words[i].encode('ascii', 'ignore')
            ax.text(X[i,0], X[i,1], X[i,2], text)
        except:
            pass

    # plot 2d
    X = calc_tsne(embedding_layer[start:end], 2)
    ax = fig.add_subplot(122)
    ax.scatter(X[:,0], X[:,1])
    for i in xrange(end-start):
        try:
            text = words[i].encode('ascii', 'ignore')
            ax.text(X[i,0], X[i,1], text)
        except:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    args = parser.parse_args()

    with gzip.open(args.model, 'rb') as f:
        model = cPickle.load(f)

    E = model.get_embeddings()
    try:
        vocabulary = model.vocabulary
    except:
        ngram_filename = DEFAULT_NGRAM_FILENAME
        from ngrams import NgramReader
        reader = NgramReader(ngram_filename, vocab_size=model.vocab_size)
        vocabulary = reader.word_array

    do_plot(E, vocabulary, start=args.start, end=args.end)
    plt.title(args.model)
    plt.show()
