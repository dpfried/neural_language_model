from wordnet_rels import Relationships
from model import EmbeddingLayer, EZPickle, EmbeddingTrainer
import theano.tensor as T
import numpy as np
import theano
import cPickle, gzip
from utils import models_in_folder
import os
import json
import pandas
import sys
import time

class DirectTensorLayer(EZPickle):
    # compute f(
    SHARED = ['V']
    OTHERS = ['n_rel', 'n_dim']

    # n_rel: number of relations (number of 2d tensors)
    # n_dim: vector representation dimensionality
    # V: a 3d tensor containing a set of 2d hidden layer matrics. Dimensions: (n_rel, n_dim, n_dim)

    def __init__(self, rng, n_rel, n_dim):
        V_size = (n_rel, n_dim, n_dim)
        V_values = np.asarray(rng.uniform(
            low=-np.sqrt(3. / (n_dim)),
            high=np.sqrt(3. / (n_dim)),
            size=V_size), dtype=theano.config.floatX)

        self.init_params(V=V_values, n_rel=n_rel, n_dim=n_dim)

    def tensor_for_relation(self, rel):
        return self.V[rel]

    def apply(self, V_rel, e1):
        return T.dot(e1, V_rel)

    def updates_symbolic(self, cost, rel, V_rel, learning_rate):
        """
        cost: a symbolic cost variable to be differentiated
        rel: the index of the relation, used to choose correct matrix V
        V_rel: the submatrix of V corresponding to rel
        learning_rate: how much to update the params prop to the cost gradient
        """
        dV = T.grad(cost, V_rel)
        return [(self.V, T.inc_subtensor(self.V[rel], -learning_rate * dV))]

class DirectTensorNetwork(EmbeddingTrainer, EZPickle):
    SHARED = ['learning_rate']
    OTHERS = ['epoch',
              'n_rel',
              'dimensions',
              'embedding_layer',
              'direct_tensor_layer',
              'mode',
              'vocab_size',
              'other_params']

    def __init__(self, rng, vocabulary, n_rel, dimensions, other_params=None, initial_embeddings=None, learning_rate=0.01, mode='FAST_RUN'):
        super(DirectTensorNetwork, self).__init__(rng, vocabulary, dimensions)
        vocab_size = len(vocabulary)

        if other_params is None:
            other_params = {}

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions)

        direct_tensor_layer = DirectTensorLayer(rng, n_rel, dimensions)

        # store attributes
        self.init_params(epoch=0,
                         learning_rate=learning_rate,
                         n_rel=n_rel,
                         dimensions=dimensions,
                         mode=mode,
                         vocab_size=vocab_size,
                         other_params=other_params,
                         embedding_layer=embedding_layer,
                         direct_tensor_layer=direct_tensor_layer)

    def get_embeddings(self):
        return self.embedding_layer.get_embeddings()

    def init_params(self, **kwargs):
        super(DirectTensorNetwork, self).init_params(**kwargs)
        # wire the network
        self.make_functions()

    def inc_epoch(self):
        self.epoch += 1

    def apply(self, emb_index, rel_index):
        emb = self.embedding_layer.embeddings_for_indices(emb_index)
        V_rel = self.direct_tensor_layer.tensor_for_relation(rel_index)

        tensor_output = self.direct_tensor_layer.apply(V_rel, emb)

        return tensor_output, emb, V_rel

    def make_functions(self):
        # training function: take an entity, rel, entity triple and return
        # the cost
        e1_index = T.lscalar('e1_index')
        rel_index = T.lscalar('rel_index')
        e2_index = T.lscalar('e2_index')
        projection, e1, V_rel = self.apply(e1_index, rel_index)

        e2 = self.embedding_layer.embeddings_for_indices(e2_index)

        difference = e2 - projection

        cost = T.dot(difference, difference)

        # embedding gradient and updates
        embedding_indices = T.stack(e1_index, e2_index)
        dembeddings = T.stack(*T.grad(cost, [e1, e2]))

        embedding_updates =  [(self.embedding_layer.embedding, T.inc_subtensor(self.embedding_layer.embedding[embedding_indices],
                                                                              -self.learning_rate * dembeddings))]

        # tensor gradient and updates
        dV = T.grad(cost, V_rel)
        tensor_updates = [
            (self.direct_tensor_layer.V, T.inc_subtensor(self.direct_tensor_layer.V[rel_index],
                                                  -self.learning_rate * dV))
        ]

        updates = embedding_updates + tensor_updates


        self.train = theano.function([e1_index, e2_index, rel_index],
                                     cost,
                                     updates=updates,
                                     mode=self.mode)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="file to dump model and stats in")
    parser.add_argument('--save_model_frequency', type=int, default=10)

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
    N_synsets = len(relationships.synsets)
    rng = np.random.RandomState(1234)
    data_rng = np.random.RandomState(1234)
    testing_corruptions = data_rng.randint(0, N_synsets, size=len(testing))

    stats_fname = os.path.join(base_dir, 'stats.pkl')


    # see if this model's already been run. If it has, load it and get the
    # params
    models = models_in_folder(base_dir)
    if models:
        model_num = max(models.keys())
        print 'loading existing model %s' % models[model_num]
        with gzip.open(models[model_num]) as f:
            ntn_model = cPickle.load(f)

        model_loaded = True
        args = ntn_model.other_params
        # rewrite in case we've copied the model file into this folder
        args['base_dir'] = base_dir
    else:
        model_loaded = False
        # dump the params
        with open(os.path.join(args['base_dir'], 'params.json'), 'w') as f:
            json.dump(args, f)


        print '... building the model'

        ntn_model = DirectTensorNetwork(
            rng=rng,
            vocabulary=relationships.synsets,
            n_rel=len(relationships.relationships),
            dimensions=50,
            learning_rate=0.01,
            other_params=args,
        )

    all_stats = pandas.DataFrame()

    def save_model():
        fname = os.path.join(args['base_dir'], 'model-%d.pkl.gz' % ntn_model.epoch)
        sys.stdout.write('dumping model to %s' % fname)
        sys.stdout.flush()
        with gzip.open(fname, 'wb') as f:
            cPickle.dump(ntn_model, f)
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
        ntn_model.inc_epoch()

        costs = []

        stats_for_block = {}
        for count, row in enumerate(data_rng.permutation(training)):
            if count % print_freq == 0:
                sys.stdout.write('\repoch %i: training instance %d of %d (%f %%)\r' % (ntn_model.epoch, count, block_size, 100. * count / block_size))
                sys.stdout.flush()

            a, b, rel = row

            # calculate the weight as a function of the correct symbols and error symbols
            cost = ntn_model.train(a,b,rel)
            costs.append(cost)

        this_training_cost = np.mean(costs)

        current_time = time.clock()
        stats_for_block['time'] = current_time - last_time
        stats_for_block['training_cost'] = this_training_cost

        sys.stdout.write('\033[k\r')
        sys.stdout.flush()
        print 'block %i \t training cost %f %% \t %f seconds' % (ntn_model.epoch, this_training_cost, current_time - last_time)
        last_time = current_time

        all_stats = pandas.concat([all_stats, pandas.DataFrame(stats_for_block, index=[ntn_model.epoch])])
        save_stats()

        # for count, row in enumerate(testing):


        if ntn_model.epoch % args['save_model_frequency'] == 0:
            save_model()
