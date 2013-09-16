import grefenstette, semeval, wordsim
import glob
import os
import re
import gzip, cPickle
import pandas
from semantic_network import *
from semantic_network import _default_word

digit_extractor = re.compile(r'.*/model-(\d+).pkl.gz')
def make_series(model_root_folder, include_synsets, normalize_components, args):
    store_fname = os.path.join(model_root_folder, 'eval-%s-%s.pkl' % (include_synsets, normalize_components))
    try:
        stats = pandas.read_pickle(store_fname)
    except:
        stats = pandas.DataFrame()

    get_model_fname = lambda model_num: os.path.join(model_root_folder, 'model-%s.pkl.gz' % model_num)
    model_fnames = glob.glob(get_model_fname('*'))
    model_nums = sorted([int(digit_extractor.match(fname).groups()[0])
                         for fname in model_fnames])
    to_plot = [n for n in model_nums if n % args.plot_interval == 0]
    for n in to_plot:
        if n in stats.index:
            continue
        with gzip.open(get_model_fname(n)) as f:
            model = cPickle.load(f)
        this_stats = run_model(model, include_synsets, normalize_components, args)
        stats = pandas.concat([stats, pandas.DataFrame([this_stats], index=[n])]).sort()
    stats.to_pickle(store_fname)
    return stats

def run_model(model, include_synsets, normalize_components, args):
    stats = {}
    stats['grefenstette_rho'], _ = grefenstette.run(model,
                                                    include_synsets,
                                                    normalize_components,
                                                    args.grefenstette_verb_file)

    stats['semeval_correlation'], stats['semeval_accuracy'] \
            = semeval.run(model,
                          include_synsets,
                          normalize_components,
                          args.semeval_root)

    stats['wordsim_rho'], _ = wordsim.run(model,
                                          include_synsets,
                                          normalize_components,
                                          args.wordsim_root)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directories', nargs='+')
    parser.add_argument('--plot_interval', type=int, default=100)
    parser.add_argument('--all_synsets', action='store_true',)
    parser.add_argument('--top_synset', action='store_true',)
    parser.add_argument('--normalize_components', action='store_true',)

    parser.add_argument('--grefenstette_verb_file', default='/home/dfried/code/verb_disambiguation')

    # parser.add_argument('--semeval_output_folder', help="folder to write results to", default="/home/dfried/code/nlm/semeval/junk")
    parser.add_argument('--semeval_root', help="folder containing semeval data", default="/home/dfried/code/semeval")

    parser.add_argument('--wordsim_root', help="folder containing wordsim353 csv file", default="/home/dfried/data/wordsim/combined.csv")

    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    if args.all_synsets:
        include_synsets='all'
    elif args.top_synset:
        include_synsets='top'
    else:
        include_synsets=None

    all_stats = dict((model_directory, make_series(model_directory, include_synsets, args.normalize_components, args))
                     for model_directory in args.model_directories)

    stat_example = all_stats.values()[0]
    import matplotlib.pyplot as plt
    for stat_name in stat_example:
        plt.figure()
        plt.title(stat_name)
        for model_directory, data in all_stats.items():
            to_plot = data[stat_name]
            if args.limit:
                to_plot = to_plot[to_plot.index <= args.limit]
            to_plot.plot(label=model_directory)
        plt.legend(loc='lower right')
    plt.show()
