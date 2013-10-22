import json
import pandas
import theano
import theano.tensor as T
from model_new import NLM
from model import _default_word
from ngrams import NgramReader
import numpy as np
import gzip, cPickle
import sys
import os
from utils import models_in_folder, sample_cumulative_discrete_distribution
import random
from os.path import join
from ntn import NeuralTensorNetwork, TensorLayer
from wordnet_rels import Relationships
from admm_new import ADMMModel
import time
from nltk.corpus import wordnet as wn

class SynsetToWord(object):
    def __init__(self, vocabulary):
        vocab = dict((word, index)
                     for index, word in enumerate(vocabulary))
        self.words_by_synset = dict(
            (synset, [vocab[lemma.name] for lemma in synset.lemmas
                      if lemma.name in vocab])
            for synset in wn.all_synsets()
        )

class RelationalADMMModel(ADMMModel):
    def make_theano_semantic_update(self):
        # should return a function that takes a number of indices: a_good,
        # b_good, rel_good, a_bad, b_bad, rel_bad
        a_good = T.scalar(dtype='int32', name='a_good')
        b_good = T.scalar(dtype='int32', name='b_good')
        rel_good = T.scalar(dtype='int32', name='rel_good')
        a_bad = T.scalar(dtype='int32', name='a_bad')
        b_bad = T.scalar(dtype='int32', name='b_bad')
        rel_bad = T.scalar(dtype='int32', name='rel_bad')

         # don't need the correct and incorrect embeddings, we calculate those
         # using the apply method
        w, v, y = self.embeddings_and_y_symbolic(T.stack(a_good, b_good), T.stack(a_bad, b_bad))[-3:]

        # W and V have nothing to do with w and v: W and V are the tensor and
        # matrix parameters, respectively for the relational network. w and v
        #are the embeddings of the syntactic and semantic models, respectively.
        # this is terrible, clean it up if this actually works
        good_score, ea_good, eb_good, W_rel_good, V_rel_good = self.semantic_model.apply(a_good, b_good, rel_good)
        bad_score, ea_bad, eb_bad, W_rel_bad, V_rel_bad = self.semantic_model.apply(a_bad, b_bad, rel_bad)

        cost = T.clip(1 - good_score + bad_score, 0, np.inf)
        augmented_cost = (1 - self.syntactic_weight) * cost + self.admm_penalty(w, v, y)

        embedding_indices = T.stack(a_good, b_good, a_bad, b_bad)
        dembeddings = T.stack(*T.grad(augmented_cost, [ea_good, eb_good, ea_bad, eb_bad]))

        embedding_updates =  [(self.semantic_model.embedding_layer.embedding, T.inc_subtensor(self.semantic_model.embedding_layer.embedding[embedding_indices],
                                                                              -self.semantic_model.learning_rate * dembeddings))]

        # tensor gradient and updates
        dW =  T.stack(*T.grad(augmented_cost, [W_rel_good, W_rel_bad]))
        dV = T.stack(*T.grad(augmented_cost, [V_rel_good, V_rel_bad]))
        tensor_indices = T.stack(rel_good, rel_bad)
        tensor_updates = [
            (self.semantic_model.tensor_layer.W, T.inc_subtensor(self.semantic_model.tensor_layer.W[tensor_indices],
                                                  -self.semantic_model.learning_rate * dW)),
            (self.semantic_model.tensor_layer.V, T.inc_subtensor(self.semantic_model.tensor_layer.V[tensor_indices],
                                                  -self.semantic_model.learning_rate * dV))
        ]

        output_updates = self.semantic_model.output_layer.updates_symbolic(augmented_cost, self.semantic_model.learning_rate)

        updates = embedding_updates + tensor_updates + output_updates

        return theano.function(inputs=[a_good, b_good, rel_good, a_bad, b_bad, rel_bad],
                               outputs=[cost, augmented_cost],
                               updates=updates,
                               mode=self.mode)

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="file to dump model and stats in")
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--train_proportion', type=float, default=0.95)
    parser.add_argument('--test_proportion', type=float, default=0.0005)
    parser.add_argument('--dimensions', type=int, default=50)
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--n_hidden_semantic', type=int, default=50)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--y_init', type=float, default=0.0)
    parser.add_argument('--semantic_gd_rate', type=float, default=0.01)
    parser.add_argument('--syntactic_gd_rate', type=float, default=0.01)
    parser.add_argument('--ngram_filename', default='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5')
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--save_model_frequency', type=int, default=10)
    parser.add_argument('--dont_save_model', action='store_true')
    parser.add_argument('--dont_save_stats', action='store_true')
    parser.add_argument('--syntactic_blocks_to_run', type=int, default=1)
    parser.add_argument('--normalize_y', action='store_true')
    parser.add_argument('--syntactic_weight', type=float, default=0.5)
    parser.add_argument('--existing_syntactic_model', help='use this existing trained model as the syntactic model')
    parser.add_argument('--existing_semantic_model', help='use this existing trained model as the semantic model')
    # parser.add_argument('--sem_validation_num_nearest', type=int, default=50, help='when running semantic validation after each round, look at the intersection of top N words in wordnet and top N by embedding for a given test word')
    # parser.add_argument('--sem_validation_num_to_test', type=int, default=500, help='in semantic validation after each round, the number of test words to sample')
    parser.add_argument('--dont_run_semantic', action='store_true')
    parser.add_argument('--dont_run_syntactic', action='store_true')
    parser.add_argument('--mode', default='FAST_RUN')
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

    relationship_path = join(base_dir, 'relationships.pkl.gz')
    try:
        with gzip.open(relationship_path) as f:
            relationships = cPickle.load(f)
        print 'loaded relationships from %s' % relationship_path
    except:
        relationships = Relationships()
        print 'saving relationships to %s' % relationship_path
        with gzip.open(relationship_path, 'wb') as f:
            cPickle.dump(relationships, f)

    replacement_column_index = args['sequence_length'] / 2

    # set up syntactic
    ngram_reader = NgramReader(args['ngram_filename'], vocab_size=args['vocab_size'], train_proportion=args['train_proportion'], test_proportion=args['test_proportion'])
    testing_block = ngram_reader.testing_block()
    vocabulary = ngram_reader.word_array
    print 'corpus contains %i ngrams' % (ngram_reader.number_of_ngrams)

    # set up semantic
    num_semantic_training = int(relationships.N * 0.98)
    semantic_training = relationships.data[:num_semantic_training]
    semantic_testing = relationships.data[num_semantic_training:]

    rng = np.random.RandomState(args['random_seed'])
    data_rng = np.random.RandomState(args['random_seed'])
    validation_rng = np.random.RandomState(args['random_seed'] + 1)
    random.seed(args['random_seed'])

    print 'constructing synset to word'
    synset_to_words = SynsetToWord(vocabulary)
    print '%d of %d synsets have no words!' % (sum(not names for names in synset_to_words.words_by_synset.values()), len(synset_to_words.words_by_synset))

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
                if issubclass(type(loaded_model), ADMMModel):
                    print 'pulling syntactic from existing model'
                    _syntactic_model = loaded_model.syntactic_model
                else:
                    _syntactic_model = loaded_model
            syn_loaded = True
            initial_embeddings = _syntactic_model.embedding_layer.embedding
        else:
            syn_loaded = False
            initial_embeddings = None

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
            sem_loaded = True
            initial_embeddings = _semantic_model.embedding_layer.embedding
        else:
            sem_loaded = False

        if not sem_loaded:
            _semantic_model = NeuralTensorNetwork(
                rng=rng,
                vocab_size=len(vocabulary),
                n_rel=len(relationships.relationships),
                dimensions=args['dimensions'],
                n_hidden=args['n_hidden_semantic'],
                mode=args['mode'],
            )
        if not syn_loaded:
            print 'creating new syn layer'
            _syntactic_model = NLM(rng=rng,
                                   vocabulary=vocabulary,
                                   dimensions=args['dimensions'],
                                   sequence_length=args['sequence_length'],
                                   n_hidden=args['n_hidden'],
                                   initial_embeddings=initial_embeddings,
                                   mode=args['mode'])

        model = RelationalADMMModel(syntactic_model=_syntactic_model,
                            semantic_model=_semantic_model,
                            vocab_size=args['vocab_size'],
                            rho=args['rho'],
                            other_params=args,
                            y_init=args['y_init'],
                            normalize_y=args['normalize_y'],
                            syntactic_weight=args['syntactic_weight'],
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

    blocks_to_run = args.get('syntactic_blocks_to_run', 1)

    vocab_size = args['vocab_size']

    while True:
        last_time = time.clock()
        model.increase_k()
        stats_for_k = {}

#         stats_for_k['syntactic_gd_rate'] = model.syntactic_model.learning_rate.get_value()
#         stats_for_k['semantic_gd_rate'] = model.semantic_model.learning_rate.get_value()

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
                    train_index = sample_cumulative_discrete_distribution(training_block[:,-1], rng=data_rng)
                    correct_symbols, error_symbols, ngram_frequency = ngram_reader.contrastive_symbols_from_row(training_block[train_index], rng=data_rng)
                    cost, augmented_cost = model.update_syntactic(correct_symbols, error_symbols)
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
            print 'syntactic mean weighted score \t%f' % syn_validation_weighted_mean

        print 'time since block init: %f' % (time.clock() - last_time)

        # semantic update step
        if not args['dont_run_semantic']:
            costs = []
            augmented_costs = []
            skip_count = 0
            for i, row in enumerate(semantic_training):
                if i % print_freq == 0:
                    sys.stdout.write('\r k %i: pair : %d / %d' % (model.k, this_count, vocab_size))
                    sys.stdout.flush()

                # get a tuple of entity, entity, relation indices
                a_index, b_index, rel_index = row
                # get the synsets for each index
                synset_a, synset_b, rel = relationships.indices_to_symbolic(row)
                # for each synset, get indices of words in the vocabulary
                # associated with the synset
                words_a = synset_to_words.words_by_synset[synset_a]
                words_b = synset_to_words.words_by_synset[synset_b]
                # if there aren't any for either, on to the next training
                # example
                if not words_a or not words_b:
                    skip_count += 1
                    continue
                # otherwise, randomly choose one and train on it
                word_a = data_rng.choice(words_a)
                word_b = data_rng.choice(words_b)

                word_b_bad = sample_cumulative_discrete_distribution(ngram_reader.cumulative_word_frequencies)

                cost, augmented_cost = model.update_semantic(word_a, word_b, rel_index, word_a, word_b_bad, rel_index)

                costs.append(cost)
                augmented_costs.append(augmented_cost)


            print 'skipped %d of %d relations because of missing synsets' % (skip_count, semantic_training.shape[0])
            stats_for_k['semantic_mean'] = np.mean(costs)
            stats_for_k['semantic_std'] = np.std(costs)
            print 'semantic mean cost \t%f' % stats_for_k['semantic_mean']
            print 'semantic std cost \t%f' % stats_for_k['semantic_std']
            stats_for_k['semantic_mean_augmented'] = np.mean(augmented_costs)
            stats_for_k['semantic_std_augmented'] = np.std(augmented_costs)
            print 'semantic mean augmented cost \t%f' % stats_for_k['semantic_mean_augmented']
            print 'semantic std augmented cost \t%f' % stats_for_k['semantic_std_augmented']

            # semantic validation
            # semantic_mean_jaccard = validate_semantic(model, args['sem_validation_num_to_test'], args['sem_validation_num_nearest'], word_similarity, validation_rng)
            # stats_for_k['semantic_validation_mean_jaccard'] = semantic_mean_jaccard

            # print 'validation:'
            # print 'semantic mean jaccard \t%f' % semantic_mean_jaccard

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
        if not args['dont_save_model'] and model.k % args['save_model_frequency'] == 0:
            save_model()

        # dump stats
        if not args['dont_save_stats']:
            all_stats.to_pickle(stats_fname)
