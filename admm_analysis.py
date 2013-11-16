import utils
import gzip
import cPickle
import ngrams
import numpy as np

from relational.translational_admm import *
from relational.wordnet_rels import Relationships
from relational.synset_to_word import SynsetToWord

import matplotlib.pyplot as plt
from matplotlib import cm

def load_model(path):
    with gzip.open(path) as f:
        return cPickle.load(f)

def load_models(base_path='/cl/work/dfried/models/relational/translational_admm_0.01/', indices=range(0, 3500, 500)):
    model_paths = utils.models_in_folder(base_path)
    return { index: load_model(path) for index, path in model_paths.iteritems()
            if index in indices }

def norms(E):
    return (E * E).sum(1)

def residuals(model):
    return model.syntactic_embedding - model.semantic_embedding

def y(model):
    return model.y.get_value()

def syn_embedding(model):
    return model.syntactic_embedding

def sem_embedding(model):
    return model.semantic_embedding

def mag(E):
    return np.sqrt((E * E).sum(1))

def cosine_similarity(E, F):
    return (E * F).sum(1) / (mag(E) * mag(F))

def plot(arrs, indices=None, **kwargs):
    sample = next(arrs.itervalues())
    r, c = sample.shape
    vmin = reduce(min, map(np.min, arrs.values()))
    vmax = reduce(max, map(np.max, arrs.values()))
    print vmin, vmax
    plt_args = {
        'cmap': cm.hot,
        'interpolation': 'nearest',
        'vmin': vmin,
        'vmax': vmax,
        'extent': [0, c, 0, r],
        'aspect':'auto',
    }
    plt_args.update(kwargs)
    if indices is None:
        indices = sorted(arrs.keys())
    for i, num in enumerate(indices):
        plt.subplot(1, len(indices), i + 1)
        try:
            plt.imshow(arrs[num], **plt_args)
        except:
            pass
        plt.title(num)

if __name__ == "__main__":
    rels = Relationships()
    import ngrams
    vocab_size = 50000
    reader = ngrams.NgramReader('/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', vocab_size=vocab_size)
    s2w = SynsetToWord(reader.word_array)
    indices_in_relationships = s2w.all_words_in_relations(rels)
    print '%d words in vocabulary covered by relationships (out of %d)' % (len(indices_in_relationships) , len(reader.word_array))

    indices = range(0, 1000, 100) + range(1000, 3500, 500)

    intersection_words = sorted(indices_in_relationships)
    not_intersection_words = sorted(set(xrange(vocab_size)).difference(indices_in_relationships))

    translational_models = load_models('/cl/work/dfried/models/relational/translational_admm_0.01/', indices)
    syntactic_models = load_models('/cl/work/dfried/models/factorial/only_syntactic/', indices)
