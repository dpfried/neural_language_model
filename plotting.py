import numpy as np
import matplotlib.pyplot as plt
import evaluation.semeval
from query import get_model, get_rels, get_vocab
import evaluation.kb_ranking
from utils import map_dict
from evaluation.evaluation import DEFAULT_MODELS
import pandas

def barplot_series(multiple_series, names, ylabels=None, title=None, xlabel=None, xlim=None):
    N = len(multiple_series[0])
    if not(all(len(xs) == N for xs in multiple_series)):
        raise ValueError("not all series are same length")
    ind = np.arange(N)

    height = 0.35

    fig, ax = plt.subplots()
    rectss = []
    for xs in multiple_series:
        rects = ax.barh(ind, xs, height)
        rectss.append(rects)

    ax.legend(rectss, names)

    if ylabels:
        ax.set_yticks(ind+height)
        ax.set_yticklabels(ylabels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)

def plot_accuracies(model, rels):
    breakdown, overall = test_socher(model, rels)
    named_breakdown = {rels.relations[ix]:acc for ix, acc in breakdown.items()}
    barplot_series([named_breakdown.values()], ['stuff'], ylabels=named_breakdown.keys())

def semeval(model, vocab):
    rho, accuracy, breakdown = evaluation.semeval.run(model.averaged_embeddings(), vocab)
    print scores.groupby('category')[['rho', 'accuracy']].mean()

def compare_kb():
    names = {
        'NTN',
        'TransE',
        'NTN + NLM (0.5)',
        'NTN + NLM (0.25)',
        'NTN + NLM (0.75)',
        'TransE + NLM (0.5)',
        'TransE + NLM (0.25)',
        'TransE + NLM (0.75)',
    }
    paths = {name:DEFAULT_MODELS[name] for name in names}
    models = map_dict(get_model, paths)
    vocabs = map_dict(get_vocab, paths)
    rels = get_rels(iter(paths.values()).next())
    def kb_accuracy(model):
        return evaluation.kb_ranking.test_socher(model, rels)
    return map_dict(kb_accuracy, models)

def compare_semeval():
    names = {
        'NTN',
        'TransE',
        'GD',
        'GD + NLM (0.5)',
        'GD + NLM (0.25)',
        'GD + NLM (0.75)',
        'NLM',
        'NTN + NLM (0.5)',
        'NTN + NLM (0.25)',
        'NTN + NLM (0.75)',
        'TransE + NLM (0.5)',
        'TransE + NLM (0.25)',
        'TransE + NLM (0.75)',
    }
    paths = {name:DEFAULT_MODELS[name] for name in names}
    models = map_dict(get_model, paths)
    vocabs = map_dict(get_vocab, paths)
    def semeval_score(name):
        model = models[name]
        vocab = vocabs[name]
        rho, accuracy, scores = evaluation.semeval.run(model.averaged_embeddings(), vocab)
        return scores.groupby('category')[['rho', 'accuracy']].mean()
    return {name:semeval_score(name) for name in names}

def plot_semeval(scores=None):
    if not scores:
        scores = compare_semeval()
    rhos = pandas.DataFrame({name:d['rho'] for name, d in scores.items()})
    acc = pandas.DataFrame({name:d['accuracy'] for name, d in scores.items()})
    for names in [['GD', 'NLM', 'NTN', 'TransE'], ['GD', 'NLM', 'NTN', 'TransE', 'GD + NLM (0.5)', 'NTN + NLM (0.5)', 'TransE + NLM (0.5)']]:
        ax = rhos[names].plot(kind='barh', legend=False)
        plt.subplots_adjust(left=0.12, bottom=0.05, right=0.63, top=0.97, wspace=0.2, hspace=0.2)
        plt.xlabel('correlation ($\\rho$)')
        plt.figure().legend(*ax.get_legend_handles_labels())
        ax = acc[names].plot(kind='barh', legend=False, xlim=(0.3, 0.55))
        plt.subplots_adjust(left=0.12, bottom=0.05, right=0.63, top=0.97, wspace=0.2, hspace=0.2)
        patches, labels = ax.get_legend_handles_labels()
        plt.xlabel('MaxDiff accuracy')
        plt.figure().legend(*ax.get_legend_handles_labels())
    plt.show()
