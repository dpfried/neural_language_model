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

class ADMMModel(object):
    def __init__(self, syntactic_model, semantic_model, vocab_size, rho, other_params, y_init=0.0, semantic_gd_rate=0.1, syntactic_gd_rate=0.1, normalize_y=False, syntactic_weight=0.5):
        self.syntactic_model = syntactic_model
        self.semantic_model = semantic_model
        self.vocab_size = vocab_size
        self.rho = rho
        self.other_params = other_params
        self.y_init = y_init
        self.normalize_y = normalize_y
        self.syntactic_weight = syntactic_weight


        self.k = 0

        # the lagrangian
        self.y = np.ones((vocab_size,syntactic_model.dimensions)) * y_init

        # self.y = theano.shared(value=y_init, name='y')
        self.semantic_gd_rate = theano.shared(value=semantic_gd_rate, name='semantic_gd_rate')
        self.syntactic_gd_rate = theano.shared(value=syntactic_gd_rate, name='syntactic_gd_rate')

        self._build_functions()

    def admm_penalty(self, w, v, y):
        if self.normalize_y:
            norm = 1.0 / self.vocab_size
        else:
            norm = 1.0
        return norm * T.dot(y, (w - v)) + self.rho / 2.0 * T.dot((w - v).T, w - v)

    def _build_functions(self):
        self.syntactic_update_function = self.make_theano_syntactic_update()
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

    @property
    def semantic_embedding(self):
        return self.semantic_model.embedding_layer.embedding

    def make_theano_syntactic_update(self):
        # build the update functions for w, the embeddings of the syntactic
        # model
        # these represent the embeddings from the semantic model for the good
        # and bad ngrams
        seq_length = self.syntactic_model.sequence_length

        w_correct_embeddings = [T.vector(name='w_correct_embedding%i' % i) for i in range(seq_length)]
        w_error_embeddings = [T.vector(name='w_error_embedding%i' % i) for i in range(seq_length)]
        w_embeddings = w_correct_embeddings + w_error_embeddings

        # these represent the corresponding embeddings from the semantic model
        v_correct_embeddings = [T.vector(name='v_correct_embedding%i' % i) for i in range(seq_length)]
        v_error_embeddings = [T.vector(name='v_error_embedding%i' % i) for i in range(seq_length)]
        v_embeddings = v_correct_embeddings + v_error_embeddings

        w = T.concatenate(w_embeddings)
        v = T.concatenate(v_embeddings)

        y_weights= [T.vector(name='y_weight%i' % i) for i in range(2 * seq_length)]
        y = T.concatenate(y_weights)

        cost = self.syntactic_model.loss(w_correct_embeddings, w_error_embeddings)
        augmented_cost = self.syntactic_weight * cost + self.admm_penalty(w, v, y)

        try:
            syntactic_params = self.syntactic_model.params
        except:
            # since sometimes we're loading in an old model where params hadn't
            # been set
            syntactic_params = self.syntactic_model.get_params()
        updates = [(param, param - self.syntactic_gd_rate * T.grad(augmented_cost, param))
                   for param in syntactic_params]

        dcorrect_embeddings = T.grad(augmented_cost, w_correct_embeddings)
        derror_embeddings = T.grad(augmented_cost, w_error_embeddings)

        return theano.function(inputs=w_embeddings + v_embeddings + y_weights,
                               outputs=dcorrect_embeddings + derror_embeddings + [cost, augmented_cost],
                               updates=updates)

    def update_syntactic(self, correct_symbols, error_symbols):
        syntactic_correct = [self.syntactic_embedding[i] for i in correct_symbols]
        syntactic_error = [self.syntactic_embedding[i] for i in error_symbols]

        semantic_correct = [self.semantic_embedding[i] for i in correct_symbols]
        semantic_error = [self.semantic_embedding[i] for i in error_symbols]

        y_correct = [self.y[i] for i in correct_symbols]
        y_error = [self.y[i] for i in error_symbols]
        y_weights = y_correct + y_error

        outputs = self.syntactic_update_function(*(syntactic_correct + syntactic_error + semantic_correct + semantic_error + y_weights))

        correct_grads = outputs[:self.syntactic_model.sequence_length]
        error_grads = outputs[self.syntactic_model.sequence_length:-2]
        cost, augmented_cost = outputs[-2:]

        weight = self.syntactic_gd_rate.get_value()

        correct_updates = - weight * np.array(correct_grads)
        error_updates = - weight * np.array(error_grads)

        self.syntactic_model.embedding_layer.update_embeddings(correct_symbols, correct_updates)
        self.syntactic_model.embedding_layer.update_embeddings(error_symbols, error_updates)

        return cost, augmented_cost, correct_updates, error_updates


    def make_theano_semantic_update(self):
        w1, w2, v1, v2 = [T.vector(name='%s_embedding' % name) for name in ['w1', 'w2', 'v1', 'v2']]
        y1, y2 = T.vector('y1_weight'), T.vector('y2_weight')

        w = T.concatenate([w1, w2])
        v = T.concatenate([v1, v2])
        y = T.concatenate([y1, y2])

        actual_sim = T.scalar(name='semantic_similarity')

        cost = self.semantic_model.loss(v1, v2, actual_sim)
        augmented_cost = (1 - self.syntactic_weight) * cost + self.admm_penalty(w, v, y)

        updates = [(param, param - self.semantic_gd_rate * T.grad(augmented_cost, param))
                   for param in self.semantic_model.params]

        dv1 = T.grad(augmented_cost, v1)
        dv2 = T.grad(augmented_cost, v2)

        return theano.function(inputs=[w1, w2, v1, v2, y1, y2, actual_sim],
                               outputs=[dv1, dv2, cost, augmented_cost],
                               updates=updates)

    def update_semantic(self, index1, index2, actual_similarity):
        w1 = self.syntactic_embedding[index1]
        w2 = self.syntactic_embedding[index2]

        v1 = self.semantic_embedding[index1]
        v2 = self.semantic_embedding[index2]

        y1 = self.y[index1]
        y2 = self.y[index2]

        dv1, dv2, cost, augmented_cost = self.semantic_update_function(w1, w2, v1, v2, y1, y2, actual_similarity)

        weight = self.semantic_gd_rate.get_value()

        self.semantic_model.embedding_layer.update_embeddings(index1, - weight * dv1)
        self.semantic_model.embedding_layer.update_embeddings(index2, - weight * dv2)

        return cost, augmented_cost, dv1, dv2

    def update_y(self):
        w = self.syntactic_embedding
        v = self.semantic_embedding
        residual = w - v
        delta_y = self.rho * residual
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

class AnnealingADMMModel(ADMMModel):
    def __init__(self, syntactic_model, semantic_model, vocab_size, rho, other_params, semantic_annealing_T, semantic_gd_initial_rate, syntactic_annealing_T, syntactic_gd_initial_rate, normalize_y=False, y_init=0.0, syntactic_weight=0.5):
        super(AnnealingADMMModel, self).__init__(syntactic_model,
                                                 semantic_model,
                                                 vocab_size,
                                                 rho,
                                                 other_params,
                                                 y_init=y_init,
                                                 semantic_gd_rate=semantic_gd_initial_rate,
                                                 syntactic_gd_rate=syntactic_gd_initial_rate,
                                                 normalize_y=normalize_y,
                                                 syntactic_weight=syntactic_weight)
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
    parser.add_argument('--sampling', default='semantic_nearest', help='semantic_nearest | embedding_nearest | random')
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--train_proportion', type=float, default=0.95)
    parser.add_argument('--test_proportion', type=float, default=0.0005)
    parser.add_argument('--dimensions', type=int, default=50)
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--y_init', type=float, default=0.0)
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
    parser.add_argument('--normalize_y', action='store_true')
    parser.add_argument('--syntactic_weight', type=float, default=0.5)
    parser.add_argument('--existing_syntactic_model', help='use this existing trained model as the syntactic model')
    parser.add_argument('--existing_semantic_model', help='use this existing trained model as the semantic model')
    parser.add_argument('--annealing', action='store_true')
    parser.add_argument('--semantic_annealing_T', type=float, default=100, help='if annealing is passed, use this to control how quickly the semantic learning rate decays from its initial value of semantic_gd_rate')
    parser.add_argument('--syntactic_annealing_T', type=float, default=100, help='if annealing is passed, use this to control how quickly the syntactic learning rate decays from its initial value of syntactic_gd_rate')
    parser.add_argument('--sem_validation_num_nearest', type=int, default=50, help='when running semantic validation after each round, look at the intersection of top N words in wordnet and top N by embedding for a given test word')
    parser.add_argument('--sem_validation_num_to_test', type=int, default=500, help='in semantic validation after each round, the number of test words to sample')
    args = vars(parser.parse_args())

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

    replacement_column_index = args['sequence_length'] / 2

    ngram_reader = NgramReader(args['ngram_filename'], vocab_size=args['vocab_size'], train_proportion=args['train_proportion'], test_proportion=args['test_proportion'])
    testing_block = ngram_reader.testing_block()
    vocabulary = ngram_reader.word_array
    print 'corpus contains %i ngrams' % (ngram_reader.number_of_ngrams)
    rng = np.random.RandomState(args['random_seed'])
    data_rng = np.random.RandomState(args['random_seed'])
    validation_rng = np.random.RandomState(args['random_seed'] + 1)
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

        if args['existing_semantic_model']:
            with gzip.open(args['existing_semantic_model'], 'rb') as f:
                _semantic_model = cPickle.load(f)
        else:
            _semantic_model = SemanticDistance(rng=rng,
                                            vocabulary=vocabulary,
                                            dimensions=args['dimensions'])

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
                                       syntactic_weight=args['syntactic_weight'])
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
                            syntactic_weight=args['syntactic_weight'])

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
        if args['annealing']:
            print 'current syntactic learning rate %f' % model.syntactic_gd_rate.get_value()
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
                train_index = sample_cumulative_discrete_distribution(training_block[:,-1], rng=data_rng)
                correct_symbols, error_symbols, ngram_frequency = ngram_reader.contrastive_symbols_from_row(training_block[train_index], rng=data_rng)
                cost, augmented_cost, correct_updates, error_updates = model.update_syntactic(correct_symbols, error_symbols)
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

        stats_for_k['syntactic_gd_rate'] = model.syntactic_gd_rate.get_value()
        stats_for_k['semantic_gd_rate'] = model.semantic_gd_rate.get_value()

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

        # syntactic validation
        semantic_mean_jaccard = validate_semantic(model, args['sem_validation_num_to_test'], args['sem_validation_num_nearest'], word_similarity, validation_rng)
        stats_for_k['semantic_validation_mean_jaccard'] = semantic_mean_jaccard

        print 'validation:'
        print 'semantic mean jaccard \t%f' % semantic_mean_jaccard

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
