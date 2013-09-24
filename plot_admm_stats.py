import pandas
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, split
import json

def plot_stats_function(function, statss, paramss, stats_labels=None, output_file=None):
    """
    function should take a L{pandas.DataFrame} and a dictionary and return a L{pandas.DataFrame}
    which will be plotted.
    statss and paramss are lists of L{pandas.DataFrame} and L{dict} respectively
    which will be passed to this function
    """
    stats_label = ""
    for i, (stats, params) in enumerate(zip(statss, paramss)):
        if stats_labels:
            stats_label = stats_labels[i]
        function(stats, params).plot(label=stats_label)
    plt.legend(loc='best').get_frame().set_alpha(0.6)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')

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
    args = parser.parse_args()
    stats = [pandas.read_pickle(join(model_dir, 'stats.pkl'))
             for model_dir in args.model_directories]
    params = []
    for model_dir in args.model_directories:
        with open(join(model_dir, 'params.json')) as f:
            params.append(json.load(f))
    print params
    def total_loss(frame, params):
        if 'syntactic_weight' in params:
            syn_w = params['syntactic_weight']
            sem_w = 1 - syn_w
        else:
            syn_w = 1
            sem_w = 1
        return syn_w * frame.syntactic_mean + sem_w * frame.semantic_mean
    for plot_tile, fn in [
        ("Total loss (semantic + syntactic)", total_loss),
        ("Y-norm (pseudo avg)", lambda frame, _: frame.y_norm / (np.sqrt(50000.))),
        ("residual norm (pseudo avg)", lambda frame, _: frame.res_norm / (np.sqrt(50000.))),
        ("total loss (augmented)", lambda frame, _: frame.semantic_mean_augmented + frame.syntactic_mean_augmented),
    ]:
        plt.figure()
        plt.title(plot_tile)
        plot_stats_function(fn, stats, params, [split(d)[1] for d in args.model_directories])
    plt.show()
