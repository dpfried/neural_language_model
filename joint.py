import pandas
import theano
import theano.tensor as T
from model import NLM
from model import _default_word
from ngrams import NgramReader
import numpy as np
from utils import sample_cumulative_discrete_distribution
import semantic_module
import gzip, cPickle
import sys
import os
from utils import models_in_folder
import random

class JointModel(object):
    def __init__(self, syntactic_model, vocab_size, other_params, semantic_gd_rate=0.1, syntactic_gd_rate=0.1):
        self.syntactic_model = syntactic_model
        self.vocab_size = vocab_size
        self.other_params = other_params

        self.k = 0

        # self.y = theano.shared(value=y_init, name='y')
        self.semantic_gd_rate = theano.shared(value=semantic_gd_rate, name='semantic_gd_rate')
        self.syntactic_gd_rate = theano.shared(value=syntactic_gd_rate, name='syntactic_gd_rate')

        self._build_functions()

    def _build_functions(self):
        self.semantic_update_function = self.make_theano_semantic_update()

    @property
    def word_to_symbol(self):
        return self.syntactic_model.word_to_symbol

    @property
    def symbol_to_word(self):
        return self.syntactic_model.symbol_to_word

    @property
    def syntactic_embedding(self):
        return self.syntactic_model.embedding_layer.embedding

    def update_syntactic(self, correct_symbols, error_symbols):
        return self.syntactic_model.train(correct_symbols, error_symbols, weighted_learning_rate=self.syntactic_gd_rate.get_value())

    def make_theano_semantic_update(self):
        w1, w2 = [T.vector(name='%s_embedding' % name) for name in ['w1', 'w2']]

        actual_sim = T.scalar(name='semantic_similarity')

        cost = ((T.dot(w1, w2) / T.sqrt(T.dot(w1, w1) * T.dot(w2, w2))) - actual_sim) ** 2

        dw1 = T.grad(cost, w1)
        dw2 = T.grad(cost, w2)

        return theano.function(inputs=[w1, w2, actual_sim],
                               outputs=[dw1, dw2, cost])

    def update_semantic(self, index1, index2, actual_similarity):
        w1 = self.syntactic_embedding[index1]
        w2 = self.syntactic_embedding[index2]

        dw1, dw2, cost = self.semantic_update_function(w1, w2, actual_similarity)

        weight = self.semantic_gd_rate.get_value()

        self.syntactic_model.embedding_layer.update_embeddings(index1, - weight * dw1)
        self.syntactic_model.embedding_layer.update_embeddings(index2, - weight * dw2)

        return cost, dw1, dw2

    def increase_k(self):
        self.k += 1

    def get_embedding(self, *args, **kwargs):
        return self.syntactic_model.get_embedding(*args, **kwargs)

    def get_embeddings(self):
        return self.syntactic_model.get_embeddings()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="file to dump model and stats in")
    parser.add_argument('--sampling', default='semantic_nearest', help='semantic_nearest | embedding_nearest | random')
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--train_proportion', type=float, default=0.95)
    parser.add_argument('--dimensions', type=int, default=50)
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--semantic_gd_rate', type=float, default=0.1)
    parser.add_argument('--syntactic_gd_rate', type=float, default=0.1)
    parser.add_argument('--k_nearest', type=int, default=5)
    parser.add_argument('--ngram_filename', default='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5')
    parser.add_argument('--word_similarity_file', default='/cl/nldata/books_google_ngrams_eng/wordnet_similarities_max.npy')
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--save_model_frequency', type=int, default=1)
    parser.add_argument('--dont_save_model', action='store_true')
    parser.add_argument('--dont_save_stats', action='store_true')
    parser.add_argument('--syntactic_blocks_to_run', type=int, default=1)
    parser.add_argument('--existing_syntactic_model', help='use this existing trained model as the syntactic model')
    args = vars(parser.parse_args())

    # see if this model's already been run. If it has, load it and get the
    # params
    base_dir = args['base_dir']
    models = models_in_folder(base_dir)
    if models:
        model_num = max(models.keys())
        print 'loading existing model %s' % models[model_num]
        with gzip.open(models[model_num]) as f:
            model = cPickle.load(f)

        model_loaded = True
        args = model.other_params
        # rewrite in case we've copied the model file into this folder
        args['base_dir'] = base_dir
    else:
        model_loaded = False

    replacement_column_index = args['sequence_length'] / 2

    ngram_reader = NgramReader(args['ngram_filename'], vocab_size=args['vocab_size'], train_proportion=args['train_proportion'], test_proportion=None)
    vocabulary = ngram_reader.word_array
    print 'corpus contains %i ngrams' % (ngram_reader.number_of_ngrams)
    rng = np.random.RandomState(args['random_seed'])
    data_rng = np.random.RandomState(args['random_seed'])
    random.seed(args['random_seed'])
    if not model_loaded:
        print 'constructing model...'
        if args['existing_syntactic_model']:
            with gzip.open(args['existing_syntactic_model'], 'rb') as f:
                _syntactic_model = cPickle.load(f)
        else:
            _syntactic_model = NLM(rng=rng,
                                vocabulary=vocabulary,
                                dimensions=args['dimensions'],
                                sequence_length=args['sequence_length'],
                                n_hidden=args['n_hidden'],
                                L1_reg=0,
                                L2_reg=0)

        model = JointModel(syntactic_model=_syntactic_model,
                           vocab_size=args['vocab_size'],
                           other_params=args,
                           semantic_gd_rate=args['semantic_gd_rate'],
                           syntactic_gd_rate=args['syntactic_gd_rate'])

    print 'loading semantic similarities'
    word_similarity = semantic_module.WordSimilarity(vocabulary, args['word_similarity_file'])

    print 'training...'

    print_freq = 100

    stats_fname = os.path.join(args['base_dir'], 'stats.pkl')

    try:
        all_stats = pandas.load(stats_fname)
    except:
        all_stats = pandas.DataFrame()

    blocks_to_run = args.get('syntactic_blocks_to_run', 1)

    vocab_size = args['vocab_size']
    k_nearest = args['k_nearest']
    sampling = args['sampling']

    while True:
        model.increase_k()
        stats_for_k = {}
        # syntactic update step
        costs = []
        for block_num in xrange(blocks_to_run):
            training_block = ngram_reader.training_block(data_rng.random_sample())
            block_size = training_block.shape[0]
            for count in xrange(block_size):
                if count % print_freq == 0:
                    sys.stdout.write('\rk %i b%i: ngram %d of %d' % (model.k, block_num, count, block_size))
                    sys.stdout.flush()
                train_index = sample_cumulative_discrete_distribution(training_block[:,-1], rng=data_rng)
                correct_symbols, error_symbols, ngram_frequency = ngram_reader.contrastive_symbols_from_row(training_block[train_index], rng=data_rng)
                cost, correct_updates, error_updates = model.update_syntactic(correct_symbols, error_symbols)
                costs.append(cost)
            if blocks_to_run > 1:
                print
                print  '%i intermediate mean %f' % (block_num, np.mean(costs[-block_size:]))

        print
        stats_for_k['syntactic_mean'] = np.mean(costs)
        stats_for_k['syntactic_std'] = np.std(costs)
        print 'syntactic mean cost \t%f' % stats_for_k['syntactic_mean']
        print 'syntactic std cost \t%f' % stats_for_k['syntactic_std']

        # semantic update step
        this_count = 0
        costs = []
        for i in data_rng.permutation(vocab_size):
            this_count += 1
            if i == 0:
                continue # skip rare word w/ undef similarities
            if sampling == 'semantic_nearest':
                for j, sim in word_similarity.most_similar_indices(i, top_n=k_nearest):
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = model.update_semantic(i, j, sim)
            elif sampling == 'embedding_nearest':
                for j, embedding_dist in model.semantic_model.embedding_layer.most_similar_embeddings(i, top_n=k_nearest):
                    sim = word_similarity.word_pairwise_sims[i, j]
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = model.update_semantic(i, j, sim)
            elif sampling == 'random':
                for j in random.sample(xrange(vocab_size), k_nearest):
                    sim = word_similarity.word_pairwise_sims[i, j]
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = model.update_semantic(i, j, sim)
            costs.append(cost)

            if this_count % print_freq == 0:
                sys.stdout.write('\r k %i: pair : %d / %d' % (model.k, this_count, vocab_size))
                sys.stdout.flush()

        print
        stats_for_k['semantic_mean'] = np.mean(costs)
        stats_for_k['semantic_std'] = np.std(costs)
        print 'semantic mean cost \t%f' % stats_for_k['semantic_mean']
        print 'semantic std cost \t%f' % stats_for_k['semantic_std']

        # append the stats for this update to all stats
        all_stats = pandas.concat([all_stats, pandas.DataFrame(stats_for_k, index=[model.k])])

        # dump it
        if not args['dont_save_model'] and model.k % args['save_model_frequency'] == 0:
            fname = os.path.join(args['base_dir'], 'model-%d.pkl.gz' % model.k)
            sys.stdout.write('dumping model to %s' % fname)
            sys.stdout.flush()
            with gzip.open(fname, 'wb') as f:
                cPickle.dump(model, f)
            sys.stdout.write('\r')
            sys.stdout.flush()

        # dump stats
        if not args['dont_save_stats']:
            all_stats.to_pickle(stats_fname)
