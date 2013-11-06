import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from utils import line_styles

def plot_stats_function(function, statss, paramss, labels=None, output_file=None, limit=0):
    """
    function should take a L{pandas.DataFrame} and a dictionary and return a L{pandas.DataFrame}
    which will be plotted.
    statss and paramss are lists of L{pandas.DataFrame} and L{dict} respectively which will be passed to this function
    kwargss is a list of dictionaries, each of which will be passed to the plot function for the respective result of
    the function
    """
    styles = line_styles(len(statss))
    for i, (stats, params, style) in enumerate(zip(statss, paramss, styles)):
        if labels:
            label = labels[i]
        try:
            to_plot = function(stats, params)
            if limit:
                to_plot = to_plot[:limit]
            to_plot.plot(style=style, label=label)
        except Exception as e:
            print e
    plt.legend(loc='best').get_frame().set_alpha(0.6)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')

def plot_loss_curves(stats_frame, limit=0):
    """take a single stats frame and plot the unaugmented syntactic, semantic, and combined loss"""
    if limit:
        stats_frame = stats_frame[:limit]
    stats_frame['joint_loss'] = stats_frame.semantic_mean + stats_frame.syntactic_mean
    stats_frame['syntactic_loss'] = stats_frame.syntactic_mean
    stats_frame['semantic_loss'] = stats_frame.semantic_mean
    stats_frame[['semantic_loss', 'syntactic_loss', 'joint_loss']].plot()
    plt.xlabel('ADMM Iterations')
    plt.ylabel('Loss')

def plot_figs(stats_frame, output_base=None, description=""):
    stats_frame[['semantic_mean', 'syntactic_mean']].plot()
    plt.title('loss %s' % description)
    if output_base:
        plt.savefig(output_base+'_loss.pdf', bbox_inches='tight')

    stats_frame[['semantic_mean', 'syntactic_mean']].plot(ylim=(0,1))
    plt.xlabel('block')
    plt.ylabel('mean loss')
    plt.title('loss %s' % description)
    if output_base:
        plt.savefig(output_base+'_loss.pdf', bbox_inches='tight')

    np.log(stats_frame[['y_norm', 'res_norm']]).plot()
    plt.xlabel('block')
    plt.ylabel('log of norm')
    plt.title('residual norms %s' % description)
    if output_base:
        plt.savefig(output_base+'_norms.pdf', bbox_inches='tight')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directories', nargs='+')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    stats = []
    params = []
    for model_dir in args.model_directories:
        try:
            _stats = pandas.read_pickle(os.path.join(model_dir, 'stats.pkl'))
            with open(os.path.join(model_dir, 'params.json')) as f:
                _params = json.load(f)

        except Exception as e:
            print e
            continue
        stats.append(_stats)
        params.append(_params)

    def total_loss(frame, params):
        if 'syntactic_weight' in params:
            syn_w = params['syntactic_weight']
            sem_w = 1 - syn_w
        else:
            syn_w = .5
            sem_w = .5
        return syn_w * frame.syntactic_mean + sem_w * frame.semantic_mean
    for plot_tile, fn in [
        ("Total loss (semantic + syntactic)", total_loss),
        ("Y-norm (pseudo avg)", lambda frame, _: frame.y_norm / (np.sqrt(50000.))),
        ("residual norm (pseudo avg)", lambda frame, _: frame.res_norm / (np.sqrt(50000.))),
        ("total loss (augmented)", lambda frame, _: frame.semantic_mean_augmented + frame.syntactic_mean_augmented),
        ("syntactic loss", lambda frame, _: frame.syntactic_mean),
        ("semantic loss", lambda frame, _: frame.semantic_mean),
        ("syntactic validation mean score", lambda frame, _: frame.syntactic_validation_mean_score),
        ("semantic validation mean jaccard", lambda frame, _: frame.semantic_validation_mean_jaccard),
    ]:
        try:
            plt.figure()
            plt.title(plot_tile)
            # for printing the names of models
            common_prefix = os.path.commonprefix(args.model_directories)
            def suffix(model_directory):
                return os.path.relpath(model_directory, common_prefix)

            plot_stats_function(fn, stats, params, [suffix(d) for d in args.model_directories], limit=args.limit)
        except Exception as e:
            print e
    plt.show()
