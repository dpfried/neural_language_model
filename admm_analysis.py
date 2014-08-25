import utils
import gzip
import cPickle
import ngrams
import numpy as np
import pandas
import matplotlib.pyplot as plt

from evaluation.evaluation import DEFAULT_MODELS
from utils import map_dict

# from relational.translational_admm import *
from relational.wordnet_rels import Relationships
# from relational.synset_to_word import SynsetToWord

import matplotlib.pyplot as plt
from matplotlib import cm

def load_model(path):
    with gzip.open(path) as f:
        return cPickle.load(f)

def load_models(base_path, indices=range(0, 1050, 50)):
    model_paths = utils.models_in_folder(base_path)
    return { index: load_model(path) for index, path in model_paths.iteritems()
            if index in indices }

# def residuals(model):
#     return model.syntactic_embedding - model.semantic_embedding

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

def norms(E):
    return np.sqrt(np.sum(E**2, axis=1))

def residual_norms(model):
    return norms(model.w_trainer.embeddings[model.indices_in_intersection] - model.v_trainer.embeddings[model.indices_in_intersection])

def w_norms(model):
    return norms(model.w_trainer.embeddings[model.indices_in_intersection])

def v_norms(model):
    return norms(model.v_trainer.embeddings[model.indices_in_intersection])

def scaled_residuals(model):
    return (residual_norms(model) / np.sqrt((w_norms(model) * v_norms(model)))).mean()

def intersection_norms(model):
    w = w_norms(model).mean()
    v = v_norms(model).mean()
    return {'w':w, 'v':v}

def agg(model):
    stats = intersection_norms(model)
    stats['scaled_residuals'] = scaled_residuals(model)
    return stats

if __name__ == "__main__":
    # rels = Relationships()
    # import ngrams
    # vocab_size = 50000
    # reader = ngrams.NgramReader('/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', vocab_size=vocab_size)
    # s2w = SynsetToWord(reader.word_array)
    # indices_in_relationships = s2w.all_words_in_relations(rels)
    # print '%d words in vocabulary covered by relationships (out of %d)' % (len(indices_in_relationships) , len(reader.word_array))

    import config
    config.DYNAMIC['compile_on_load'] = False

    dump = False

    indices=range(0, 1050, 20)

    # names = ['NTN + NLM (0.5)', 'GD + NLM (0.5)', 'TransE + NLM (0.5)']
    # names = ['GD + NLM (\\rho=0.1)', 'GD + NLM (\\rho=0.05)', 'GD + NLM (\\rho=0.005)']
    names = ['GD + NLM ($\\rho=0.1$)', 'GD + NLM ($\\rho=0.05$)', 'GD + NLM ($\\rho=0.005$)']

    stats_frames = {}

    for pretty_name in names:
        name = pretty_name.replace('$', '')
        fname = name.replace(' ', '_') + '.pkl'
        if dump:
            models = load_models(DEFAULT_MODELS[name], indices=indices)
            stats = pandas.DataFrame(map_dict(agg, models))
            stats.to_pickle(fname)
        else:
            stats = pandas.read_pickle(fname)
            stats_frames[pretty_name] = stats

    def filter_(df):
        cols = df.columns.tolist()
        return df[df.index <= 1000][list(reversed(cols))]
    if not dump:
        w_df = filter_(pandas.DataFrame(map_dict(lambda d: d.transpose().w, stats_frames)))
        v_df = filter_(pandas.DataFrame(map_dict(lambda d: d.transpose().v, stats_frames)))
        sr_df = filter_(pandas.DataFrame(map_dict(lambda d: d.transpose().scaled_residuals, stats_frames)))

        for df, title in [(w_df, "mean $||\mathbf{w}||_2$"), (v_df, "mean $||\mathbf{v}||_2$"), (sr_df, "mean $||\mathbf{w} - \mathbf{v}||_2 / (\sqrt{||\mathbf{w}||_2 ||\mathbf{v}||_2})$")]:
            ax = df.plot(legend=False)
            plt.title(title)
            plt.xlabel("training iterations")
            plt.ylabel("mean norm")
            plt.subplots_adjust(bottom=0.2)
            plt.figure().legend(*ax.get_legend_handles_labels())
        plt.show()

#     translational_models = load_models('/cl/work/dfried/models/relational/translational_admm_0.01/', indices)
#     syntactic_models = load_models('/cl/work/dfried/models/factorial/only_syntactic/', indices)
