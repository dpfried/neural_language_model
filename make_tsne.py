import ngrams
from calc_tsne import calc_tsne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from query import load_zipped_pickle
import gzip
import cPickle

def data_and_dicts(classifier_path, ngram_file='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5'):
    classifier = load_zipped_pickle(classifier_path)
    ngram_reader = ngrams.NgramReader(ngram_file, vocab_size=classifier.vocab_size)
    return classifier.embedding_layer.embedding, [unicode(w, 'utf-8') for w in ngram_reader.word_array]

def do_plot(embedding_layer, words, start=0, end=100):
    X = calc_tsne(embedding_layer[start:end], 3)
    words = words[start:end]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2])
    for i in xrange(end-start):
        ax.text(X[i,0], X[i,1], X[i,2], words[i])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    args = parser.parse_args()

    with gzip.open(args.model, 'rb') as f:
        model = cPickle.load(f)

    E = model.embedding_layer.embedding
    try:
        vocabulary = model.vocabulary
    except:
        ngram_filename = model.other_params['ngram_filename']
        from ngrams import NgramReader
        reader = NgramReader(ngram_filename, vocab_size=model.vocab_size)
        vocabulary = reader.word_array

    do_plot(E, vocabulary, start=args.start, end=args.end)
    plt.title(args.model)
    plt.show()
