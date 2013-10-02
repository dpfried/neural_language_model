import grefenstette, semeval, wordsim
import gzip, cPickle
import pandas
from semantic_network import *
from model import _default_word
from admm import ADMMModel
from joint import JointModel
from utils import models_in_folder
from os.path import split, join

def make_series(model_root_folder,
                include_synsets=None,
                normalize_components=False,
                plot_interval=100,
                limit=None,
                **run_model_args):
    store_fname = join(model_root_folder, 'eval-%s-%s.pkl' % (include_synsets, normalize_components))
    try:
        stats = pandas.read_pickle(store_fname)
    except:
        stats = pandas.DataFrame()

    models = models_in_folder(model_root_folder)
    model_nums = sorted(models.keys())

    to_plot = [n for n in model_nums if n % plot_interval == 0]
    if 1 in model_nums:
        to_plot = [1] + to_plot
    if limit is not None:
        to_plot = [n for n in to_plot if n <= limit]
    print model_root_folder
    for n in to_plot:
        if n in stats.index:
            print 'already has %i' % n
            continue
        with gzip.open(models[n]) as f:
            model = cPickle.load(f)
        this_stats = run_model(model, include_synsets, normalize_components, **run_model_args)
        stats = pandas.concat([stats, pandas.DataFrame([this_stats], index=[n])]).sort()
    stats.to_pickle(store_fname)
    return stats

def run_model(model, include_synsets=None, normalize_components=False,
              grefenstette_verb_file='/home/dfried/code/verb_disambiguation',
              semeval_root="/home/dfried/code/semeval",
              wordsim_root="/home/dfried/data/wordsim/combined.csv",
              **kwargs):
    stats = {}
    stats['grefenstette_rho'], _ = grefenstette.run(model,
                                                    include_synsets,
                                                    normalize_components,
                                                    grefenstette_verb_file)

    stats['semeval_correlation'], stats['semeval_accuracy'] \
            = semeval.run(model,
                          include_synsets,
                          normalize_components,
                          semeval_root)

    stats['wordsim_rho'], _ = wordsim.run(model,
                                          include_synsets,
                                          normalize_components,
                                          wordsim_root)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directories', nargs='+')
    parser.add_argument('--plot_interval', type=int, default=100)
    parser.add_argument('--all_synsets', action='store_true',)
    parser.add_argument('--top_synset', action='store_true',)
    parser.add_argument('--normalize_components', action='store_true')

    parser.add_argument('--grefenstette_verb_file', default='/home/dfried/code/verb_disambiguation')

    # parser.add_argument('--semeval_output_folder', help="folder to write results to", default="/home/dfried/code/nlm/semeval/junk")
    parser.add_argument('--semeval_root', help="folder containing semeval data", default="/home/dfried/code/semeval")

    parser.add_argument('--wordsim_root', help="folder containing wordsim353 csv file", default="/home/dfried/data/wordsim/combined.csv")

    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--save_graphs_base')
    args = parser.parse_args()

    if args.all_synsets:
        include_synsets='all'
    elif args.top_synset:
        include_synsets='top'
    else:
        include_synsets=None

    all_stats = dict((model_directory, make_series(model_directory,
                                                   include_synsets=include_synsets,
                                                   **vars(args)))
                     for model_directory in args.model_directories)

    stat_example = all_stats.values()[0]
    import matplotlib.pyplot as plt
    for stat_name in stat_example:
        plt.figure()
        plt.title(stat_name)
        for model_directory, data in all_stats.items():
            try:
                to_plot = data[stat_name]
                if args.limit:
                    to_plot = to_plot[to_plot.index <= args.limit]
                to_plot.plot(label=split(model_directory)[1])
            except Exception as e:
                print stat_name, model_directory
                print e
        plt.legend(loc='lower right').get_frame().set_alpha(0.6)
        if args.save_graphs_base:
            plt.savefig('%s_%s.pdf' % (args.save_graphs_base, stat_name), bbox_inches='tight')
    plt.show()
