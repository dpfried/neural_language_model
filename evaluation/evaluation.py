import theano
import grefenstette, semeval, wordsim
import gzip, cPickle
import pandas
from models import *
from utils import models_in_folder, line_styles
from os.path import join
import os
from query import get_vocab_container

# for backward compatability with unpickling models pickled with theano 0.5
# when unpickling with 0.6
theano.tensor.basic.Subtensor = theano.tensor.Subtensor
theano.config.on_unused_input = 'warn'

def make_series(model_root_folder,
                plot_interval=100,
                limit=None,
                no_new=False,
                average_embeddings=False,
                **run_model_args):
    if average_embeddings:
        suffix = 'eval-averaged.pkl'
    else:
        suffix = 'eval-None-False.pkl' # holdout from when we had include_synsets and normalize_components
    store_fname = join(model_root_folder, suffix)
    print store_fname
    try:
        stats = pandas.read_pickle(store_fname)
        print "read pickle from %s" % store_fname
    except:
        stats = pandas.DataFrame()
        print 'created new frame'
    if no_new:
        return stats

    models = models_in_folder(model_root_folder)
    model_nums = sorted(models.keys())

    latest_num = model_nums[-1] if model_nums else -1
    latest_num = -1

    print 'plotting every %i' % plot_interval

    to_plot = [n for n in model_nums if n % plot_interval == 0 and n != latest_num]
    if 1 in model_nums:
        to_plot = [1] + to_plot
    if limit is not None:
        to_plot = [n for n in to_plot if n <= limit]
    vocab_container = None
    print model_root_folder
    for n in to_plot:
        if n in stats.index:
            print 'already has %i' % n
            continue
        try:
            print 'loading %i' % n
            with gzip.open(models[n]) as f:
                model = cPickle.load(f)
        except Exception as e:
            print e
            continue
        # load the vocabulary if not already cached
        if not vocab_container:
            vocab_container = get_vocab_container(model)
        embeddings = model.embeddings if not average_embeddings else model.averaged_embeddings()
        this_stats = run_model(embeddings, vocab_container,  **run_model_args)
        stats = pandas.concat([stats, pandas.DataFrame([this_stats], index=[n])]).sort()
        stats.to_pickle(store_fname)
    return stats

def run_model(embeddings, vocab_container,
              grefenstette_verb_file='/home/dfried/code/verb_disambiguation',
              semeval_root="/home/dfried/code/semeval",
              wordsim_root="/home/dfried/data/wordsim/combined.csv",
              **kwargs):
    stats = {}
    stats['grefenstette_rho'], _ = grefenstette.run(embeddings,
                                                    vocab_container,
                                                    grefenstette_verb_file)

    stats['semeval_correlation'], stats['semeval_accuracy'], semeval_stats \
            = semeval.run(embeddings,
                          vocab_container,
                          semeval_root)

    stats['wordsim_rho'], _ = wordsim.run(embeddings,
                                          vocab_container,
                                          wordsim_root)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directories', nargs='+')
    parser.add_argument('--plot_interval', type=int, default=100)
    parser.add_argument('--grefenstette_verb_file', default='/home/dfried/code/verb_disambiguation')

    # parser.add_argument('--semeval_output_folder', help="folder to write results to", default="/home/dfried/code/nlm/semeval/junk")
    parser.add_argument('--semeval_root', help="folder containing semeval data", default="/home/dfried/code/semeval")

    parser.add_argument('--wordsim_root', help="folder containing wordsim353 csv file", default="/home/dfried/data/wordsim/combined.csv")

    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no_new', action='store_true')
    parser.add_argument('--save_graphs_base')
    parser.add_argument('--average_embeddings', action='store_true')
    args = parser.parse_args()

    import config
    # we only read the values from the models, we don't need to have any
    # compiled theano functions
    config.DYNAMIC['compile_on_load'] = False

    all_stats = dict((model_directory, make_series(model_directory,
                                                   **vars(args)))
                     for model_directory in args.model_directories)

    for name, s in all_stats.items():
        print name
        print s

    stat_params = {
        'grefenstette_rho': {
            'title': "Grefenstette",
        },
        'semeval_accuracy': {
            'title': "SemEval Accuracy",
        },
        'semeval_correlation': {
            'title': "SemEval Correlation",
        },
        'wordsim_rho': {
            'title': "WordSim",
        },
    }

    ys = {
        'grefenstette_rho': '$\\rho$',
        'semeval_accuracy': 'accuracy',
        'semeval_correlation': '$\\rho$',
        'wordsim_rho': '$\\rho$',
    }

    # for printing the names of models
    common_prefix = os.path.commonprefix(args.model_directories)
    def suffix(model_directory):
        if common_prefix != model_directory:
            return os.path.relpath(model_directory, common_prefix)
        else:
            return model_directory

    stat_example = [s for s in all_stats.values() if len(s) != 0][0]
    import matplotlib.pyplot as plt
    for stat_name in stat_example:
        plt.figure()
        plt.title(stat_name)
        print len(all_stats)
        styles = line_styles(len(all_stats))
        for model_directory, style in zip(args.model_directories, styles):
            data = all_stats[model_directory]
            try:
                to_plot = data[stat_name]
                if args.limit:
                    to_plot = to_plot[to_plot.index <= args.limit]
                to_plot.plot(label=suffix(model_directory), style=style, **stat_params[stat_name])
                plt.xlabel('training iterations')
                plt.ylabel(ys[stat_name])
                plt.subplots_adjust(bottom=0.15)
            except Exception as e:
                print 'exception'
                print stat_name, model_directory
                print e
        try:
            plt.legend(loc='lower right').get_frame().set_alpha(0.6)
        except Exception as e:
            print e
        if args.save_graphs_base:
            plt.savefig('%s_%s.pdf' % (args.save_graphs_base, stat_name), bbox_inches='tight')
    plt.show()
    max_index = max(reduce(lambda p, q: set(p).intersection(set(q)),
                           map(lambda s: s.index, all_stats.values())))
    if args.limit and max_index > args.limit:
        max_index = args.limit
    data = pandas.DataFrame({
        path.split('/')[-1]: frame.ix[max_index]
        for path, frame in all_stats.iteritems()
    })
    print 'evaluated at iteration %d:' % max_index
    print data.transpose().to_latex(float_format=lambda f: "%0.2f" % f)
    # import IPython
    # IPython.embed()
