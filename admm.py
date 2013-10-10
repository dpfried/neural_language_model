import json
import pandas
import theano
import theano.tensor as T
from model import NLM
from semantic_network import SemanticDistance
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

from model import EZPickle

class ADMMModel(EZPickle):
    SHARED = ['y', 'syntactic_weight']
    OTHERS = ['syntactic_model',
              'semantic_model',
              'vocab_size',
              'rho',
              'other_params',
              'y_init',
              'normalize_y',
              ('mode', 'FAST_RUN'),
              'k']
    def __init__(self, syntactic_model, semantic_model, vocab_size, rho, other_params, y_init=0.0, normalize_y=False, syntactic_weight=0.5, mode='FAST_RUN'):

        # the lagrangian
        y = (np.ones((vocab_size,syntactic_model.dimensions)) * y_init).astype(theano.config.floatX)

        # self.y = theano.shared(value=y_init, name='y')
        syntactic_weight = np.cast[theano.config.floatX](syntactic_weight)

        self.init_params(syntactic_model=syntactic_model,
                         semantic_model=semantic_model,
                         vocab_size=vocab_size,
                         rho=rho,
                         other_params=other_params,
                         y_init=y_init,
                         normalize_y=normalize_y,
                         syntactic_weight=syntactic_weight,
                         mode=mode,
                         y=y,
                         k=0)


    def init_params(self, **kwargs):
        super(ADMMModel, self).init_params(**kwargs)
        self._build_functions()

    def admm_penalty(self, w, v, y):
        if self.normalize_y:
            norm = 1.0 / self.vocab_size
        else:
            norm = 1.0
        return norm * T.dot(y, (w - v)) + self.rho / 2.0 * T.dot((w - v).T, w - v)

    def _build_functions(self):
        self.update_syntactic = self.make_theano_syntactic_update()
        self.update_semantic = self.make_theano_semantic_update()

    @property
    def word_to_symbol(self):
        return self.syntactic_model.word_to_symbol

    @property
    def symbol_to_word(self):
        return self.syntactic_model.symbol_to_word

    @property
    def syntactic_embedding(self):
        return self.syntactic_model.get_embeddings()

    @property
    def semantic_embedding(self):
        return self.semantic_model.get_embeddings()

    def embeddings_and_y_symbolic(self, correct_indices, error_indices):
        all_indices = T.concatenate(correct_indices, error_indices)

        w_correct_embedding = self.syntactic_model.embed_indices_symbolic(correct_indices)
        w_error_embedding = self.syntactic_model.embed_indices_symbolic(error_indices)

        v_correct_embedding = self.semantic_model.embed_indices_symbolic(correct_indices)
        v_error_embedding = self.semantic_model.embed_indices_symbolic(error_indices)

        y = self.y[all_indices]

        return w_correct_embedding, w_error_embedding, v_correct_embedding, v_error_embedding, y

    def make_theano_syntactic_update(self):
        # build the update functions for w, the embeddings of the syntactic
        # model
        correct_indices = self.syntactic_model.symbolic_indices('correct_index')
        error_indices = self.syntactic_model.symbolic_indices('error_index')

        w_correct_embedding, w_error_embedding, v_correct_embedding, v_error_embedding, y = self.embeddings_and_y_symbolic(correct_indices, error_indices)

        w = T.concatenate([w_correct_embedding, w_error_embedding])
        v = T.concatenate([v_correct_embedding, v_error_embedding])

        cost = self.syntactic_model.cost_from_embeddings_symbolic(w_correct_embedding, w_error_embedding)

        augmented_cost = self.syntactic_weight * cost + self.admm_penalty(w, v, y)

        updates = self.syntactic_model.updates_symbolic(augmented_cost, correct_indices, error_indices,
                                                        w_correct_embedding, w_error_embedding)

        return theano.function(inputs=[correct_indices, error_indices],
                               outputs=[cost, augmented_cost],
                               updates=updates,
                               mode=self.mode)

    def make_theano_semantic_update(self):
        index1 = T.scalar(dtype='int32', name='index1')
        index2 = T.scalar(dtype='int32', name='index2')

        w1, w2, v1, v2, y = self.embeddings_and_y_symbolic(T.stack(index1), T.stack(index2))

        w = T.concatenate([w1, w2])
        v = T.concatenate([v1, v2])

        actual_sim = T.scalar(name='semantic_similarity')

        cost = self.semantic_model.cost_from_embeddings_symbolic(v1, v2, actual_sim)
        augmented_cost = (1 - self.syntactic_weight) * cost + self.admm_penalty(w, v, y)

        updates = self.semantic_model.updates_symbolic(augmented_cost, index1, index2, v1, v2)

        return theano.function(inputs=[index1, index2, actual_sim],
                               outputs=[cost, augmented_cost],
                               updates=updates,
                               mode=self.mode)

    def update_y(self):
        return
        w = self.syntactic_model.embedding
        v = self.syntactic_model.embedding
        residual = w - v
        delta_y = self.rho * residual
        updates = (self.y, self.y + self.rho * residual)
        self.y += delta_y

        res = np.ravel(residual)
        y = np.ravel(self.y)
        res_norm = np.sqrt(np.dot(res, res))
        y_norm = np.sqrt(np.dot(y, y))
        return res_norm, y_norm

    def increase_k(self):
        self.k += 1

    def get_embedding(self, *args, **kwargs):
        # return self.syntactic_model.get_embedding(*args, **kwargs)
        return np.concatenate([self.syntactic_model.get_embedding(*args, **kwargs), self.semantic_model.get_embedding(*args, **kwargs)])
        # return 0.5 * (self.syntactic_model.get_embedding(*args, **kwargs) + self.semantic_model.get_embedding(*args, **kwargs))

    def get_embeddings(self):
        # return self.syntactic_model.get_embeddings()
        return np.concatenate([self.syntactic_model.get_embeddings(), self.semantic_model.get_embeddings()], axis=1)
        # return 0.5 * (self.syntactic_model.get_embeddings() + self.semantic_model.get_embeddings())

    def dump_embeddings(self, filename, precision=8):
        format_str = '%%0.%if' % precision
        float_to_str = lambda f: format_str % f
        with open(filename, 'w') as f:
            for index, embedding in enumerate(self.get_embeddings()):
                # skip RARE
                if index == 0:
                    continue
                vector_string_rep = ' '.join(map(float_to_str, embedding))
                f.write('%s %s\n' % (self.symbol_to_word[index], vector_string_rep))

class AnnealingADMMModel(ADMMModel):
    SHARED = ADMMModel.SHARED

    OTHERS = ADMMModel.OTHERS + ['semantic_gd_initial_rate', 'syntactic_gd_initial_rate', 'semantic_annealing_T', 'syntactic_annealing_T']

    def __init__(self, syntactic_model, semantic_model, vocab_size, rho, other_params, semantic_annealing_T, semantic_gd_initial_rate, syntactic_annealing_T, syntactic_gd_initial_rate, normalize_y=False, y_init=0.0, syntactic_weight=0.5, mode='FAST_RUN'):
        super(AnnealingADMMModel, self).__init__(syntactic_model,
                                                 semantic_model,
                                                 vocab_size,
                                                 rho,
                                                 other_params,
                                                 y_init=y_init,
                                                 semantic_gd_rate=semantic_gd_initial_rate,
                                                 syntactic_gd_rate=syntactic_gd_initial_rate,
                                                 normalize_y=normalize_y,
                                                 syntactic_weight=syntactic_weight,
                                                 mode=mode)
        self.semantic_gd_initial_rate = semantic_gd_initial_rate
        self.syntactic_gd_initial_rate = syntactic_gd_initial_rate
        self.semantic_annealing_T = semantic_annealing_T
        self.syntactic_annealing_T = syntactic_annealing_T

    def increase_k(self):
        super(AnnealingADMMModel, self).increase_k()
        def decayed_rate(initial, T):
            # initial: the maximum weight (at beginning)
            # T: the param that controls how fast we descend (proportional to
            # current value of k)
            return float(initial) / (1 + float(self.k) / T)
        self.semantic_gd_rate.set_value(decayed_rate(self.semantic_gd_initial_rate, self.semantic_annealing_T))
        self.syntactic_gd_rate.set_value(decayed_rate(self.syntactic_gd_initial_rate, self.syntactic_annealing_T))

def validate_syntactic(model, testing_block, ngram_reader, rng=None):
    if rng is None:
        rng = np.random

    test_values = []
    test_frequencies = []
    n_test_instances = testing_block.shape[0]
    for test_index in xrange(n_test_instances):
        if test_index % print_freq == 0:
            sys.stdout.write('\rtesting instance %d of %d (%f %%)\r' % (test_index, n_test_instances, 100. * test_index / n_test_instances))
            sys.stdout.flush()
        correct_symbols, error_symbols, ngram_frequency = ngram_reader.contrastive_symbols_from_row(testing_block[test_index], replacement_column_index=replacement_column_index, rng=rng)
        test_values.append(model.syntactic_model.score(correct_symbols) - model.syntactic_model.score(error_symbols))
        test_frequencies.append(ngram_frequency)
    test_mean = np.mean(test_values)
    test_weighted_mean = np.mean(np.array(test_values) * np.array(test_frequencies))
    return test_mean, test_weighted_mean

def validate_semantic(model, num_words, num_nearest, word_similarity, rng=None):
    if rng is None:
        rng = np.random
    jaccards = []
    for i in xrange(num_words):
        index = rng.randint(model.vocab_size)
        wn_closest = set(j for (j, sim) in word_similarity.most_similar_indices(index, top_n=num_nearest))
        embedding_closest = set(j for (j, dist) in model.semantic_model.embedding_layer.most_similar_embeddings(i, top_n=num_nearest))
        c_intersection = len(wn_closest.intersection(embedding_closest))
        c_union = len(wn_closest.union(embedding_closest))
        jaccards.append(float(c_intersection) / c_union)
    return np.mean(jaccards)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="file to dump model and stats in")
    parser.add_argument('--sampling', default='random', help='semantic_nearest | embedding_nearest | random')
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--train_proportion', type=float, default=0.95)
    parser.add_argument('--test_proportion', type=float, default=0.0005)
    parser.add_argument('--dimensions', type=int, default=50)
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--y_init', type=float, default=0.0)
    parser.add_argument('--semantic_gd_rate', type=float, default=0.01)
    parser.add_argument('--syntactic_gd_rate', type=float, default=0.01)
    parser.add_argument('--k_nearest', type=int, default=10)
    parser.add_argument('--ngram_filename', default='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5')
    parser.add_argument('--word_similarity_file', default='/cl/nldata/books_google_ngrams_eng/wordnet_similarities_max.npy')
    parser.add_argument('--word_similarity_memmap', default='/tmp/wordnet_similarities_max.memmap', help='use this file as a shared memmap between processes')
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--save_model_frequency', type=int, default=10)
    parser.add_argument('--dont_save_model', action='store_true')
    parser.add_argument('--dont_save_stats', action='store_true')
    parser.add_argument('--syntactic_blocks_to_run', type=int, default=1)
    parser.add_argument('--normalize_y', action='store_true')
    parser.add_argument('--syntactic_weight', type=float, default=0.5)
    parser.add_argument('--existing_syntactic_model', help='use this existing trained model as the syntactic model')
    parser.add_argument('--existing_semantic_model', help='use this existing trained model as the semantic model')
    parser.add_argument('--annealing', action='store_true')
    parser.add_argument('--semantic_annealing_T', type=float, default=100, help='if annealing is passed, use this to control how quickly the semantic learning rate decays from its initial value of semantic_gd_rate')
    parser.add_argument('--syntactic_annealing_T', type=float, default=100, help='if annealing is passed, use this to control how quickly the syntactic learning rate decays from its initial value of syntactic_gd_rate')
    parser.add_argument('--sem_validation_num_nearest', type=int, default=50, help='when running semantic validation after each round, look at the intersection of top N words in wordnet and top N by embedding for a given test word')
    parser.add_argument('--sem_validation_num_to_test', type=int, default=500, help='in semantic validation after each round, the number of test words to sample')
    parser.add_argument('--dont_run_semantic', action='store_true')
    parser.add_argument('--dont_run_syntactic', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = vars(parser.parse_args())

    if args['profile']:
        from theano import ProfileMode
        theano_mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
    else:
        theano_mode = 'FAST_RUN'

    # if we're only running semantic or syntactic, rho and y init must be 0 to
    # isolate the loss function to the syntactic or semantic loss
    if args['dont_run_semantic'] or args['dont_run_syntactic']:
        print 'not running joint model, setting y and rho to 0'
        args['rho'] = 0
        args['y_init'] = 0

    base_dir = args['base_dir']

    # see if this model's already been run. If it has, load it and get the
    # params
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
        # dump the params
        with open(os.path.join(args['base_dir'], 'params.json'), 'w') as f:
            json.dump(args, f)


    replacement_column_index = args['sequence_length'] / 2

    ngram_reader = NgramReader(args['ngram_filename'], vocab_size=args['vocab_size'], train_proportion=args['train_proportion'], test_proportion=args['test_proportion'])
    testing_block = ngram_reader.testing_block()
    vocabulary = ngram_reader.word_array
    print 'corpus contains %i ngrams' % (ngram_reader.number_of_ngrams)
    rng = np.random.RandomState(args['random_seed'])
    data_rng = np.random.RandomState(args['random_seed'])
    validation_rng = np.random.RandomState(args['random_seed'] + 1)
    random.seed(args['random_seed'])

    # construct the admm, possibly using some existing semantic or syntactic
    # model, possibly annealing
    if not model_loaded:
        print 'constructing model...'
        if args['existing_syntactic_model']:
            # check to see if the model to load is itself an ADMM. if it is,
            # pull out the syntactic model, otherwise treat it as its own
            # syntactic model
            with gzip.open(args['existing_syntactic_model'], 'rb') as f:
                loaded_model = cPickle.load(f)
                if issubclass(type(loaded_model), ADMMModel):
                    print 'pulling syntactic from existing model'
                    _syntactic_model = loaded_model.syntactic_model
                else:
                    _syntactic_model = loaded_model
        else:
            _syntactic_model = NLM(rng=rng,
                                   vocabulary=vocabulary,
                                   dimensions=args['dimensions'],
                                   sequence_length=args['sequence_length'],
                                   n_hidden=args['n_hidden'],
                                   mode=theano_mode)

        if args['existing_semantic_model']:
            # check to see if the model to load is itself an ADMM. if it is,
            # pull out the semantic model, otherwise treat it as its own
            # semantic model
            with gzip.open(args['existing_semantic_model'], 'rb') as f:
                loaded_model = cPickle.load(f)
                if issubclass(type(loaded_model), ADMMModel):
                    print 'pulling semantic from existing model'
                    _semantic_model = loaded_model.semantic_model
                else:
                    _semantic_model = loaded_model
        else:
            _semantic_model = SemanticDistance(rng=rng,
                                               vocabulary=vocabulary,
                                               dimensions=args['dimensions'],
                                               mode=theano_mode)

        if args['annealing']:
            print 'annealing'
            model = AnnealingADMMModel(syntactic_model=_syntactic_model,
                                       semantic_model=_semantic_model,
                                       vocab_size=args['vocab_size'],
                                       rho=args['rho'],
                                       other_params=args,
                                       semantic_annealing_T=args['semantic_annealing_T'],
                                       semantic_gd_initial_rate=args['semantic_gd_rate'],
                                       syntactic_annealing_T=args['syntactic_annealing_T'],
                                       syntactic_gd_initial_rate=args['syntactic_gd_rate'],
                                       normalize_y=args['normalize_y'],
                                       y_init=args['y_init'],
                                       syntactic_weight=args['syntactic_weight'],
                                       mode=theano_mode)
        else:
            model = ADMMModel(syntactic_model=_syntactic_model,
                              semantic_model=_semantic_model,
                              vocab_size=args['vocab_size'],
                              rho=args['rho'],
                              other_params=args,
                              y_init=args['y_init'],
                              semantic_gd_rate=args['semantic_gd_rate'],
                              syntactic_gd_rate=args['syntactic_gd_rate'],
                              normalize_y=args['normalize_y'],
                              syntactic_weight=args['syntactic_weight'],
                              mode=theano_mode)

    print 'loading semantic similarities'
    word_similarity = semantic_module.WordSimilarity(vocabulary, args['word_similarity_file'], memmap_filename=args['word_similarity_memmap'])

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
        if args['annealing']:
            print 'current syntactic learning rate %f' % model.syntactic_gd_rate.get_value()

        stats_for_k['syntactic_gd_rate'] = model.syntactic_gd_rate.get_value()
        stats_for_k['semantic_gd_rate'] = model.semantic_gd_rate.get_value()

        if not args['dont_run_syntactic']:
            # syntactic update step
            costs = []
            augmented_costs = []
            for block_num in xrange(blocks_to_run):
                training_block = ngram_reader.training_block(data_rng.random_sample())
                block_size = training_block.shape[0]
                for count in xrange(block_size):
                    if count % print_freq == 0:
                        sys.stdout.write('\rk %i b%i: ngram %d of %d' % (model.k, block_num, count, block_size))
                        sys.stdout.flush()
                        if args['profile'] and count > 0:
                            print
                            theano_mode.print_summary()
                    train_index = sample_cumulative_discrete_distribution(training_block[:,-1], rng=data_rng)
                    correct_symbols, error_symbols, ngram_frequency = ngram_reader.contrastive_symbols_from_row(training_block[train_index], rng=data_rng)
                    cost, augmented_cost, = model.update_syntactic(correct_symbols, error_symbols)
                    costs.append(cost)
                    augmented_costs.append(augmented_cost)
                if blocks_to_run > 1:
                    print
                    print  '%i intermediate mean %f' % (block_num, np.mean(costs[-block_size:]))
                    print  '%i intermediate aug mean %f' % (block_num, np.mean(augmented_costs[-block_size:]))

            print
            stats_for_k['syntactic_mean'] = np.mean(costs)
            stats_for_k['syntactic_std'] = np.std(costs)
            print 'training:'
            print 'syntactic mean cost \t%f' % stats_for_k['syntactic_mean']
            print 'syntactic std cost \t%f' % stats_for_k['syntactic_std']
            stats_for_k['syntactic_mean_augmented'] = np.mean(augmented_costs)
            stats_for_k['syntactic_std_augmented'] = np.std(augmented_costs)
            print 'syntactic mean augmented cost \t%f' % stats_for_k['syntactic_mean_augmented']
            print 'syntactic std augmented cost \t%f' % stats_for_k['syntactic_std_augmented']

            # syntactic validation
            syn_validation_mean, syn_validation_weighted_mean = validate_syntactic(model, testing_block, ngram_reader, validation_rng)
            stats_for_k['syntactic_validation_mean_score'] = syn_validation_mean
            stats_for_k['syntactic_validation_weighted_mean_score'] = syn_validation_weighted_mean

            print 'validation:'
            print 'syntactic mean score \t%f' % syn_validation_mean
            print 'syntactic mean weighted score \t%f' % syn_validation_mean

        # semantic update step
        if args['annealing']:
            print 'current semantic learning rate %f' % model.semantic_gd_rate.get_value()
        if not args['dont_run_semantic']:
            this_count = 0
            costs = []
            augmented_costs = []
            for i in data_rng.permutation(vocab_size):
                this_count += 1
                if i == 0:
                    continue # skip rare word w/ undef similarities
                if sampling == 'semantic_nearest':
                    for j, sim in word_similarity.most_similar_indices(i, top_n=k_nearest):
                        if sim == -np.inf:
                            continue
                        cost, augmented_cost, w1_update, w2_update = model.update_semantic(i, j, sim)
                elif sampling == 'embedding_nearest':
                    for j, embedding_dist in model.semantic_model.embedding_layer.most_similar_embeddings(i, top_n=k_nearest):
                        sim = word_similarity.word_pairwise_sims[i, j]
                        if sim == -np.inf:
                            continue
                        cost, augmented_cost, w1_update, w2_update = model.update_semantic(i, j, sim)
                elif sampling == 'random':
                    for j in random.sample(xrange(vocab_size), k_nearest):
                        sim = word_similarity.word_pairwise_sims[i, j]
                        if sim == -np.inf:
                            continue
                        cost, augmented_cost, w1_update, w2_update = model.update_semantic(i, j, sim)
                costs.append(cost)
                augmented_costs.append(augmented_cost)

                if this_count % print_freq == 0:
                    sys.stdout.write('\r k %i: pair : %d / %d' % (model.k, this_count, vocab_size))
                    sys.stdout.flush()

            print
            stats_for_k['semantic_mean'] = np.mean(costs)
            stats_for_k['semantic_std'] = np.std(costs)
            print 'semantic mean cost \t%f' % stats_for_k['semantic_mean']
            print 'semantic std cost \t%f' % stats_for_k['semantic_std']
            stats_for_k['semantic_mean_augmented'] = np.mean(augmented_costs)
            stats_for_k['semantic_std_augmented'] = np.std(augmented_costs)
            print 'semantic mean augmented cost \t%f' % stats_for_k['semantic_mean_augmented']
            print 'semantic std augmented cost \t%f' % stats_for_k['semantic_std_augmented']

            # semantic validation
            semantic_mean_jaccard = validate_semantic(model, args['sem_validation_num_to_test'], args['sem_validation_num_nearest'], word_similarity, validation_rng)
            stats_for_k['semantic_validation_mean_jaccard'] = semantic_mean_jaccard

            print 'validation:'
            print 'semantic mean jaccard \t%f' % semantic_mean_jaccard

        if not args['dont_run_semantic'] and not args['dont_run_syntactic']:
            # lagrangian update
            print 'updating y'
            res_norm, y_norm = model.update_y()
            stats_for_k['res_norm'] = res_norm
            stats_for_k['y_norm'] = y_norm
            print 'k: %d\tnorm(w - v) %f \t norm(y) %f' % (model.k, res_norm, y_norm)

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
