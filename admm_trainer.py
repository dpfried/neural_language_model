import time
import json
import pandas
from models import SequenceScoringNN, SimilarityNN, ADMM
from ngrams import NgramReader
import numpy as np
from utils import sample_cumulative_discrete_distribution
import semantic_module
import gzip, cPickle
import sys
import os
from utils import models_in_folder, compose
import random
import theano

theano.config.exception_verbosity = 'high'

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
        test_values.append(model.w_trainer.score(*list(correct_symbols)) - model.w_trainer.score(*list(error_symbols)))
        test_frequencies.append(ngram_frequency)
    test_mean = np.mean(test_values)
    test_weighted_mean = np.mean(np.array(test_values) * np.array(test_frequencies))
    return test_mean, test_weighted_mean

def validate_semantic(model, num_words, num_nearest, word_similarity, rng=None):
    if rng is None:
        rng = np.random
    jaccards = []
    for i in xrange(num_words):
        index = rng.randint(model.v_trainer.vocab_size)
        wn_closest = set(j for (j, sim) in word_similarity.most_similar_indices(index, top_n=num_nearest))
        embedding_closest = set(j for (j, dist) in model.v_trainer.embedding_layer.most_similar_embeddings(i, top_n=num_nearest))
        c_intersection = len(wn_closest.intersection(embedding_closest))
        c_union = len(wn_closest.union(embedding_closest))
        jaccards.append(float(c_intersection) / c_union)
    return np.mean(jaccards)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="file to dump model and stats in")
    # params for both
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--dimensions', type=int, default=50)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--save_model_frequency', type=int, default=10)
    parser.add_argument('--mode', default='FAST_RUN')
    parser.add_argument('--adagrad', action='store_true')

    # params for syntactic
    parser.add_argument('--dont_run_syntactic', action='store_true')
    parser.add_argument('--existing_syntactic_model', help='use this existing trained model as the syntactic model')
    parser.add_argument('--syntactic_learning_rate', type=float, default=0.01)
    parser.add_argument('--train_proportion', type=float, default=0.95)
    parser.add_argument('--test_proportion', type=float, default=0.0005)
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--ngram_filename', default='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5')
    parser.add_argument('--syntactic_blocks_to_run', type=int, default=1)

    # params for semantic
    parser.add_argument('--dont_run_semantic', action='store_true')
    parser.add_argument('--existing_semantic_model', help='use this existing trained model as the semantic model')
    parser.add_argument('--semantic_learning_rate', type=float, default=0.01)
    parser.add_argument('--semantic_block_size', type=int, default=100000)
    parser.add_argument('--k_nearest', type=int, default=5)
    parser.add_argument('--word_similarity_file', default='/cl/nldata/books_google_ngrams_eng/wordnet_similarities_max.npy')
    parser.add_argument('--word_similarity_memmap', default='/tmp/wordnet_similarities_max.memmap', help='use this file as a shared memmap between processes')
    parser.add_argument('--sem_validation_num_nearest', type=int, default=50, help='when running semantic validation after each round, look at the intersection of top N words in wordnet and top N by embedding for a given test word')
    parser.add_argument('--sem_validation_num_to_test', type=int, default=500, help='in semantic validation after each round, the number of test words to sample')
    parser.add_argument('--semantic_blocks_to_run', type=int, default=1)

    args = vars(parser.parse_args())

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

    if not args['dont_run_semantic']:
        print 'loading semantic similarities'
        word_similarity = semantic_module.WordSimilarity(vocabulary, args['word_similarity_file'], memmap_filename=args['word_similarity_memmap'])
        print 'computing terms with semantic distance'
        indices_in_intersection=set(i for i, v in enumerate(map(compose(np.any, np.isfinite),
                                                                word_similarity.word_pairwise_sims))
                                    if bool(v)
                                    and i != 0) # exclude the rare word
    else:
        indices_in_intersection = set()

    # construct the admm, possibly using some existing semantic or syntactic
    # model
    if not model_loaded:
        print 'constructing model...'
        if args['existing_syntactic_model']:
            # check to see if the model to load is itself an ADMM. if it is,
            # pull out the syntactic model, otherwise treat it as its own
            # syntactic model
            with gzip.open(args['existing_syntactic_model'], 'rb') as f:
                loaded_model = cPickle.load(f)
                if issubclass(type(loaded_model), ADMM):
                    print 'pulling syntactic from existing model'
                    _syntactic_model = loaded_model.w_trainer
                else:
                    _syntactic_model = loaded_model
        else:
            _syntactic_model = SequenceScoringNN(rng=rng,
                                                 vocab_size=args['vocab_size'],
                                                 dimensions=args['dimensions'],
                                                 sequence_length=args['sequence_length'],
                                                 n_hidden=args['n_hidden'],
                                                 learning_rate=args['syntactic_learning_rate'],
                                                 mode=args['mode'],
                                                 adagrad=args['adagrad'])

        if args['existing_semantic_model']:
            # check to see if the model to load is itself an ADMM. if it is,
            # pull out the semantic model, otherwise treat it as its own
            # semantic model
            with gzip.open(args['existing_semantic_model'], 'rb') as f:
                loaded_model = cPickle.load(f)
                if issubclass(type(loaded_model), ADMM):
                    print 'pulling semantic from existing model'
                    _semantic_model = loaded_model.v_trainer
                else:
                    _semantic_model = loaded_model
        else:
            _semantic_model = SimilarityNN(rng=rng,
                                           vocab_size=args['vocab_size'],
                                           dimensions=args['dimensions'],
                                           learning_rate=args['semantic_learning_rate'],
                                           mode=args['mode'],
                                           adagrad=args['adagrad'])



        model = ADMM(w_trainer=_syntactic_model,
                     v_trainer=_semantic_model,
                     vocab_size=args['vocab_size'],
                     indices_in_intersection=list(indices_in_intersection),
                     dimensions=args['dimensions'],
                     rho=args['rho'],
                     other_params=args,
                     mode=args['mode'])

    def save_model():
        fname = os.path.join(args['base_dir'], 'model-%d.pkl.gz' % model.k)
        sys.stdout.write('dumping model to %s' % fname)
        sys.stdout.flush()
        with gzip.open(fname, 'wb') as f:
            cPickle.dump(model, f)
        sys.stdout.write('\r')
        sys.stdout.flush()

    # save the initial state
    if not model_loaded:
        save_model()

    print 'training...'

    print_freq = 100

    stats_fname = os.path.join(args['base_dir'], 'stats.pkl')

    try:
        all_stats = pandas.load(stats_fname)
    except:
        all_stats = pandas.DataFrame()

    vocab_size = args['vocab_size']
    k_nearest = args['k_nearest']

    while True:
        last_time = time.clock()
        model.increase_k()
        stats_for_k = {}

        if not args['dont_run_syntactic']:
            # syntactic update step
            costs = []
            for block_num in xrange(args['syntactic_blocks_to_run']):
                training_block = ngram_reader.training_block(data_rng.random_sample())
                block_size = training_block.shape[0]
                for count in xrange(block_size):
                    if count % print_freq == 0:
                        sys.stdout.write('\rk %i b%i: ngram %d of %d' % (model.k, block_num, count, block_size))
                        sys.stdout.flush()
                    train_index = sample_cumulative_discrete_distribution(training_block[:,-1], rng=data_rng)
                    correct_symbols, error_symbols, ngram_frequency = ngram_reader.contrastive_symbols_from_row(training_block[train_index], rng=data_rng)
                    cost = model.update_w(*(list(correct_symbols) + list(error_symbols)))
                    costs.append(cost)
                if args['syntactic_blocks_to_run'] > 1:
                    print
                    print  '%i intermediate mean %f' % (block_num, np.mean(costs[-block_size:]))

            print
            stats_for_k['syntactic_mean'] = np.mean(costs)
            stats_for_k['syntactic_std'] = np.std(costs)
            print 'training:'
            print 'syntactic mean cost \t%f' % stats_for_k['syntactic_mean']
            print 'syntactic std cost \t%f' % stats_for_k['syntactic_std']

            # syntactic validation
            syn_validation_mean, syn_validation_weighted_mean = validate_syntactic(model, testing_block, ngram_reader, validation_rng)
            stats_for_k['syntactic_validation_mean_score'] = syn_validation_mean
            stats_for_k['syntactic_validation_weighted_mean_score'] = syn_validation_weighted_mean

            print 'validation:'
            print 'syntactic mean score \t%f' % syn_validation_mean
            print 'syntactic mean weighted score \t%f' % syn_validation_weighted_mean

        # print 'time since block init: %f' % (time.clock() - last_time)

        # semantic update step
        if not args['dont_run_semantic']:
            this_count = 0
            costs = []
            for block_num in xrange(args['semantic_blocks_to_run']):
                for i in xrange(args['semantic_block_size']):
                    train_i = -1
                    while train_i not in indices_in_intersection:
                        train_i = sample_cumulative_discrete_distribution(ngram_reader.cumulative_word_frequencies, rng=data_rng)
                    for j in xrange(args['k_nearest']):
                        train_j = -1
                        while train_j not in indices_in_intersection:
                            train_j = sample_cumulative_discrete_distribution(ngram_reader.cumulative_word_frequencies, rng=data_rng)
                        if word_similarity.word_pairwise_sims[train_i, train_j] == -np.inf:
                            continue
                        sim = word_similarity.word_pairwise_sims[train_i, train_j]
                        cost = model.update_v(train_i, train_j, sim)
                    costs.append(cost)

                    if i % print_freq == 0:
                        sys.stdout.write('\r k %i: pair : %d / %d' % (model.k, i, args['semantic_block_size']))
                        sys.stdout.flush()

                if args['semantic_blocks_to_run'] > 1:
                    print
                    print  '%i intermediate mean %f' % (block_num, np.mean(costs[-args['semantic_block_size']:]))
            print
            stats_for_k['semantic_mean'] = np.mean(costs)
            stats_for_k['semantic_std'] = np.std(costs)
            print 'semantic mean cost \t%f' % stats_for_k['semantic_mean']
            print 'semantic std cost \t%f' % stats_for_k['semantic_std']

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

        print 'time: %f' % (time.clock() - last_time)

        # append the stats for this update to all stats
        all_stats = pandas.concat([all_stats, pandas.DataFrame(stats_for_k, index=[model.k])])

        # dump it
        if model.k % args['save_model_frequency'] == 0:
            save_model()

        # dump stats
        all_stats.to_pickle(stats_fname)
