import time
import json
import pandas
from models import TensorNN, TranslationalNN
import numpy as np
import gzip, cPickle
import sys
import os
from utils import models_in_folder
import random
import theano
from os.path import join
# from relational.synset_to_word import Relationships, SynsetToWord
from relational.wordnet_rels import RelationshipsNTNDataset

theano.config.exception_verbosity = 'high'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="file to dump model and stats in")
    # params for both
    parser.add_argument('--dimensions', type=int, default=50)
    parser.add_argument('--save_model_frequency', type=int, default=10)
    parser.add_argument('--mode', default='FAST_RUN')
    parser.add_argument('--random_seed', type=int, default=1234)

    # params for semantic
    parser.add_argument('--model_class', type=str, default='TensorNN')
    parser.add_argument('--existing_semantic_model', help='use this existing trained model as the semantic model')
    parser.add_argument('--semantic_learning_rate', type=float, default=0.01)
    parser.add_argument('--semantic_tensor_n_hidden', type=int, default=50)
    # parser.add_argument('--semantic_block_size', type=int, default=100000)
    # parser.add_argument('--sem_validation_num_nearest', type=int, default=50, help='when running semantic validation after each round, look at the intersection of top N words in wordnet and top N by embedding for a given test word')
    # parser.add_argument('--sem_validation_num_to_test', type=int, default=500, help='in semantic validation after each round, the number of test words to sample')
    parser.add_argument('--semantic_blocks_to_run', type=int, default=1)

    args = vars(parser.parse_args())
    print args

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

    rng = np.random.RandomState(args['random_seed'])
    data_rng = np.random.RandomState(args['random_seed'])
    validation_rng = np.random.RandomState(args['random_seed'] + 1)
    random.seed(args['random_seed'])


    relationship_path = join(base_dir, 'relationships.pkl.gz')
    try:
        with gzip.open(relationship_path) as f:
            relationships = cPickle.load(f)
        print 'loaded relationships from %s' % relationship_path
    except:
        # relationships = Relationships()
        relationships = RelationshipsNTNDataset(None, data_rng)
        print 'saving relationships to %s' % relationship_path
        with gzip.open(relationship_path, 'wb') as f:
            cPickle.dump(relationships, f)

    vocabulary = relationships.vocab

    # print 'constructing synset to word'
    # synset_to_words = SynsetToWord(vocabulary)
    # print '%d of %d synsets have no words!' % (sum(not names for names in synset_to_words.words_by_synset.values()), len(synset_to_words.words_by_synset))

    # construct the admm, possibly using some existing semantic or syntactic
    # model
    if not model_loaded:
        semantic_class = eval(args['model_class'])
        semantic_args = {
            'rng': rng,
            'vocab_size': len(vocabulary),
            'n_rel': relationships.N_relationships,
            'dimensions': args['dimensions'],
            'learning_rate': args['semantic_learning_rate'],
            'mode': args['mode'],
            'other_params': args,
        }
        if args['model_class'] == 'TensorNN':
            semantic_args['n_hidden'] = args['semantic_tensor_n_hidden']

        model = semantic_class(**semantic_args)

    def save_model(filename=None):
        if filename is None:
            filename = 'model-%d.pkl.gz' % model.k
        fname = os.path.join(args['base_dir'], filename)
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

    vocab_size = len(vocabulary)

    while True:
        last_time = time.clock()
        model.increase_k()
        stats_for_k = {}

        # semantic update step
        this_count = 0
        augmented_costs = []
        costs = []
        for block_num in xrange(args['semantic_blocks_to_run']):
            # block_size = semantic_training.shape[0]
            block_size = relationships.N_train
            for i, (word_a, rel_index, word_b) in enumerate(relationships.training_block()):
            # for i in xrange(block_size):
                if i % print_freq == 0:
                    sys.stdout.write('\r k %i: pair : %d / %d' % (model.k, i, block_size))
                    sys.stdout.flush()

                word_a_new, word_b_new, rel_index_new = word_a, word_b, rel_index

                # choose to corrupt one part of the triple
                to_mod = data_rng.choice(3)

                # corrupt with some other part
                if to_mod == 0:
                    while word_a_new == word_a:
                        word_a_new = data_rng.choice(vocab_size)
                elif to_mod == 1:
                    while word_b_new == word_b:
                        word_b_new = data_rng.choice(vocab_size)
                elif to_mod == 2:
                    while rel_index_new == rel_index:
                        # rel_index_new = data_rng.randint(N_relationships)
                        rel_index_new = data_rng.randint(relationships.N_relationships)

                cost, augmented_cost = model.train(word_a, word_b, rel_index, word_a_new, word_b_new, rel_index_new)
                if not np.isfinite(cost):
                    print 'nan detected'
                    save_model('nan_dump.pkl.gz')
                    import IPython
                    IPython.embed()
                costs.append(cost)

                if i % print_freq == 0:
                    sys.stdout.write('\r k %i: pair : %d / %d' % (model.k, i, block_size))
                    sys.stdout.flush()

            if args['semantic_blocks_to_run'] > 1:
                print
                print  '%i intermediate mean %f' % (block_num, np.mean(costs[-block_size:]))
        print
        if not np.isfinite(np.mean(costs)):
            print 'nan cost mean detected'
            save_model('nan_dump.pkl.gz')
            import IPython
            IPython.embed()
        stats_for_k['semantic_mean'] = np.mean(costs)
        stats_for_k['semantic_std'] = np.std(costs)
        print 'semantic mean cost \t%f' % stats_for_k['semantic_mean']
        print 'semantic std cost \t%f' % stats_for_k['semantic_std']

        print 'time: %f' % (time.clock() - last_time)

        # append the stats for this update to all stats
        all_stats = pandas.concat([all_stats, pandas.DataFrame(stats_for_k, index=[model.k])])

        # dump it
        if model.k % args['save_model_frequency'] == 0:
            save_model()

        # dump stats
        all_stats.to_pickle(stats_fname)
