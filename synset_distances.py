from collections import defaultdict
from nltk.corpus import wordnet as wn
import gzip, cPickle
from scipy.spatial.distance import pdist
import numpy as np

def synsets_to_words(words):
    synsets_to_words = defaultdict(list)
    for word in words:
        for synset in wn.synsets(word):
            synsets_to_words[synset].append(word)
    return synsets_to_words

def get_related_words(word, synsets_to_words_lookup):
    return set(w for synset in wn.synsets(word)
               for w in synsets_to_words_lookup[synset])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to model containing embeddigns')
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

    s2w = synsets_to_words(vocabulary)
    id_map = dict((word, index) for index, word in enumerate(vocabulary))

    euclidean = []
    cosine = []
    for words in s2w.values():
        indices = [id_map[word] for word in words]
        dist_mat = pdist(E[indices], 'cosine')
        cosine.append(dist_mat.mean())
        dist_mat = pdist(E[indices], 'euclidean')
        euclidean.append(dist_mat.mean())

    euclidean = np.asarray(euclidean)
    euclidean = euclidean[np.isnan(euclidean) == False]

    cosine = np.asarray(cosine)
    cosine = cosine[np.isnan(cosine) == False]

    print 'within synset distance, cosine: %0.4f +/- %0.4f' % (cosine.mean(), cosine.std())
    print 'within synset distance, euclidean: %0.4f +/- %0.4f' % (euclidean.mean(), euclidean.std())

    all_dists = pdist(E, 'cosine')
    adf = np.asarray(all_dists).flatten()

    print 'all distance, cosine: %0.4f +/- %0.4f' % (adf.mean(), adf.std())

    all_dists = pdist(E, 'euclidean')
    adf = np.asarray(all_dists).flatten()
    print 'all distance, euclidean: %0.4f +/- %0.4f' % (adf.mean(), adf.std())
