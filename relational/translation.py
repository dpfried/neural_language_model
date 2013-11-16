import numpy as np
from model import EmbeddingLayer, EZPickle, EmbeddingTrainer
import theano.tensor as T
import theano
import os
import gzip, cPickle
from relational.wordnet_rels import Relationships
from relational.synset_to_word import SynsetToWord
from utils import models_in_folder
import pandas
import json
import sys
import time
from ngrams import NgramReader

class TranslationLayer(EZPickle):
    SHARED = ["R"]
    OTHERS = ["n_rel", "n_dim", "normalize"]

    def __init__(self, rng, n_rel, n_dim, normalize=False):
        # create the relation embedding matrix: a n_rel x n_dim matrix. Each
        # initial embedding sampled from a multivariate normal with mean 0 and
        # standard deviation 1/k
        # R_size = (n_rel, n_dim)
        R_values = rng.multivariate_normal(np.zeros(n_dim), 1./(n_dim**2) * np.eye(n_dim), n_rel)

        self.init_params(R=R_values, n_rel=n_rel, n_dim=n_dim, normalize=normalize)

    def embedding_for_relation(self, rel):
        return self.R[rel]

    def updates_symbolic(self, cost, rel_list, embedding_list, learning_rate):
        dR = T.stack(*T.grad(cost, embedding_list))
        update = -learning_rate * dR
        if not self.normalize:
            return [(self.R, T.inc_subtensor(self.R[rel_list], update))]
        else:
            R_rel = T.stack(*rel_list)
            unnormed_R = R_rel + update
            normed_R = unnormed_R / T.sqrt(T.batched_dot(unnormed_R, unnormed_R))
            return [(self.R, T.set_subtensor(self.R[rel_list], normed_R))]

class TranslationNetwork(EmbeddingTrainer, EZPickle):
    SHARED = ['learning_rate']
    OTHERS = ['epoch',
              'n_rel',
              'dimensions',
              'embedding_layer',
              'relation_layer',
              'mode',
              'vocab_size',
              'other_params']

    def __init__(self, rng, vocabulary, n_rel, dimensions, other_params=None, initial_embeddings=None, learning_rate=0.01, mode='FAST_RUN'):
        super(TranslationNetwork, self).__init__(rng, vocabulary, dimensions)
        vocab_size = len(vocabulary)

        if other_params is None:
            other_params = {}

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions,
                                         initial_embeddings=initial_embeddings,
                                         mode=mode)

        relation_layer = TranslationLayer(rng, n_rel, dimensions, False)

        # store attributes
        self.init_params(epoch=0,
                         learning_rate=learning_rate,
                         n_rel=n_rel,
                         dimensions=dimensions,
                         mode=mode,
                         vocab_size=vocab_size,
                         other_params=other_params,
                         embedding_layer=embedding_layer,
                         relation_layer=relation_layer,
                         )

    def get_embeddings(self):
        return self.embedding_layer.get_embeddings()

    def init_params(self, **kwargs):
        super(TranslationNetwork, self).init_params(**kwargs)
        # wire the network
        self.make_functions()

    def apply(self, e1_index, e2_index, rel_index):
        """
        given the indices of entity 1, entity 2, and the relation,
        compute the plausibility score and return the score
        and the respective embeddings (so that the embeddings
        can be used in symbolic differentiation and updates
        """
        e1 = self.embedding_layer.embeddings_for_indices(e1_index)
        e2 = self.embedding_layer.embeddings_for_indices(e2_index)
        R_rel = self.relation_layer.embedding_for_relation(rel_index)

        composite = e1 + R_rel - e2
        score = -1 * T.sqrt(T.dot(composite, composite))
        return score, e1, e2, R_rel

    def inc_epoch(self):
        self.epoch += 1

    def embed_indices_symbolic(self, indices):
        return self.embedding_layer.embed_indices_symbolic(indices)

    def make_functions(self):
        # training function: take an entity, rel, entity triple and return
        # the cost
        e1_index_good = T.lscalar('e1_index_good')
        rel_index_good = T.lscalar('rel_index_good')
        e2_index_good = T.lscalar('e2_index_good')
        good_score, e1_good, e2_good, R_rel_good = self.apply(e1_index_good, e2_index_good, rel_index_good)


        e1_index_bad = T.lscalar('e1_index_bad')
        rel_index_bad = T.lscalar('rel_index_bad')
        e2_index_bad = T.lscalar('e2_index_bad')
        bad_score, e1_bad, e2_bad, R_rel_bad = self.apply(e1_index_bad, e2_index_bad, rel_index_bad)

        cost = T.clip(1 - good_score + bad_score, 0, np.inf)

        embedding_indices = [e1_index_good, e2_index_good, e1_index_bad, e2_index_bad]
        embeddings = [e1_good, e2_good, e1_bad, e2_bad]

        # embedding gradient and updates
        embedding_updates = self.embedding_layer.updates_symbolic(cost, embedding_indices, embeddings, self.learning_rate)

        # relation gradient and updates
        relation_updates = self.relation_layer.updates_symbolic(cost,
                                                                [rel_index_good, rel_index_bad],
                                                                [R_rel_good, R_rel_bad],
                                                                self.learning_rate)

        updates = embedding_updates + relation_updates

        self.train = theano.function([e1_index_good, e2_index_good, rel_index_good,
                                      e1_index_bad, e2_index_bad, rel_index_bad],
                                     cost,
                                     updates=updates,
                                     mode=self.mode)

        self.test = theano.function([e1_index_good, e2_index_good, rel_index_good],
                                    good_score,
                                    mode=self.mode)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="file to dump model and stats in")
    parser.add_argument('--save_model_frequency', type=int, default=10)
    parser.add_argument('--train_words', action='store_true', help='vocabulary is words sampled from synsets, instead of synsets')
    parser.add_argument('--vocab_size', type=int, default=50000, help='only valid with train_words')
    parser.add_argument('--ngram_filename', default='/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', help='only valid with train_words')

    args = vars(parser.parse_args())

    base_dir = args['base_dir']

    relationships_path = os.path.join(base_dir, 'relationships.pkl.gz')
    # setup training data
    try:
        with gzip.open(relationships_path, 'rb') as f:
            relationships = cPickle.load(f)
    except:
        relationships = Relationships()
        with gzip.open(os.path.join(base_dir, 'relationships.pkl.gz'), 'wb') as f:
            cPickle.dump(relationships, f)

    num_training = int(relationships.N * 0.9)
    training = relationships.data[:num_training]
    testing = relationships.data[num_training:]
    block_size = training.shape[0]
    if args['train_words']:
        ngram_reader = NgramReader(args['ngram_filename'], vocab_size=args['vocab_size'])
        vocabulary = ngram_reader.word_array
        synset_to_words = SynsetToWord(vocabulary)
    else:
        vocabulary = relationships.synsets
    vocab_size = len(vocabulary)
    N_relationships = len(relationships.relationships)
    rng = np.random.RandomState(1234)
    data_rng = np.random.RandomState(1234)
    testing_corruptions = data_rng.randint(0, vocab_size, size=len(testing))

    stats_fname = os.path.join(base_dir, 'stats.pkl')


    # see if this model's already been run. If it has, load it and get the
    # params
    models = models_in_folder(base_dir)
    if models:
        model_num = max(models.keys())
        print 'loading existing model %s' % models[model_num]
        with gzip.open(models[model_num]) as f:
            translation_model = cPickle.load(f)

        model_loaded = True
        args = translation_model.other_params
        # rewrite in case we've copied the model file into this folder
        args['base_dir'] = base_dir
    else:
        model_loaded = False
        # dump the params
        with open(os.path.join(args['base_dir'], 'params.json'), 'w') as f:
            json.dump(args, f)


        print '... building the model'

        translation_model = TranslationNetwork(
            rng=rng,
            vocabulary=vocabulary,
            n_rel=N_relationships,
            dimensions=50,
            learning_rate=0.01,
            other_params=args,
        )

    all_stats = pandas.DataFrame()

    def save_model():
        fname = os.path.join(args['base_dir'], 'model-%d.pkl.gz' % translation_model.epoch)
        sys.stdout.write('dumping model to %s' % fname)
        sys.stdout.flush()
        with gzip.open(fname, 'wb') as f:
            cPickle.dump(translation_model, f)
        sys.stdout.write('\r')
        sys.stdout.flush()

    def save_stats():
        all_stats.to_pickle(stats_fname)

    if not model_loaded:
        save_model()

    print '... training'

    last_time = time.clock()
    block_test_frequency = 1
    print_freq = 100


    while True:
        translation_model.inc_epoch()

        costs = []

        stats_for_block = {}
        skip_count = 0
        for count in xrange(block_size):
            row = training[data_rng.choice(block_size)]
            if count % print_freq == 0:
                sys.stdout.write('\repoch %i: training instance %d of %d (%f %%)\r' % (translation_model.epoch, count, block_size, 100. * count / block_size))
                sys.stdout.flush()

            if args['train_words']:
                # a, b should represent word indices, so we have to go from
                # synsets to words
                # get the synsets for each index
                _, _, rel = row
                synset_a, synset_b, _ = relationships.indices_to_symbolic(row)
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
                a = data_rng.choice(words_a)
                b = data_rng.choice(words_b)
            else:
                # a, b should represent synset indices
                a, b, rel = row

            a_new, b_new, rel_new = a, b, rel

            # choose to corrupt one part of the triple
            to_mod = data_rng.choice(3)

            # corrupt with some other part
            if to_mod == 0:
                while a_new == a:
                    a_new = data_rng.randint(vocab_size)
            elif to_mod == 1:
                while b_new == b:
                    b_new = data_rng.randint(vocab_size)
            elif to_mod == 2:
                while rel_new == rel:
                    rel_new = data_rng.randint(N_relationships)

            # calculate the weight as a function of the correct symbols and error symbols
            cost = translation_model.train(a,b,rel,a_new,b_new,rel_new)
            costs.append(cost)

        this_training_cost = np.mean(costs)

        current_time = time.clock()
        stats_for_block['time'] = current_time - last_time
        stats_for_block['training_cost'] = this_training_cost

        sys.stdout.write('\033[k\r')
        sys.stdout.flush()
        print 'block %i \t training cost %f %% \t %f seconds' % (translation_model.epoch, this_training_cost, current_time - last_time)
        last_time = current_time

        all_stats = pandas.concat([all_stats, pandas.DataFrame(stats_for_block, index=[translation_model.epoch])])
        save_stats()

        # for count, row in enumerate(testing):


        if translation_model.epoch % args['save_model_frequency'] == 0:
            save_model()
