import ngrams
from calc_tsne import calc_tsne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from query import load_zipped_pickle

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
    plt.show()
