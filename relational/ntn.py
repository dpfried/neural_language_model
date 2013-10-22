from wordnet_rels import Relationships
from model_new import EmbeddingLayer, EZPickle, LinearScalarResponse
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

class TensorLayer(EZPickle):
    # compute f(
    SHARED = ['W', 'V', 'b']
    OTHERS = ['n_rel', 'n_in', 'n_out']

    # n_rel: number of relations (number of 3d tensors)
    # n_in: input dimensionality
    # n_out: output dimensionality
    # W: a 4d tensor containing a set of 3d relationship tensors. Dimensions: (n_rel, n_out, n_in, n_in)
    # V: a 3d tensor containing a set of 2d hidden layer matrics. Dimensions: (n_rel, n_out, 2*n_in)

    def __init__(self, rng, n_rel, n_in, n_out, activation=T.tanh):
        W_size = (n_rel, n_out, n_in, n_in)
        # what to change this heuristic to?
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=W_size), dtype=theano.config.floatX)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        V_size = (n_rel, n_out, 2*n_in)
        V_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=V_size), dtype=theano.config.floatX)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        if n_out > 1:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        else:
            b_values = np.cast[theano.config.floatX](0.0)

        self.init_params(W=W_values, V=V_values, b=b_values, n_rel=n_rel, n_in=n_in, n_out=n_out)

    # def apply(self, rels, e1, e2):
    #     # e1 * W
    #     intr = T.batched_dot(e1, self.W[rels])

    #     # e1 * W * e2
    #     W_term, _ = theano.scan(fn=lambda x_mat, y_mat: T.tensordot(x_mat, y_mat, 1), outputs_info=None,
    #                                   sequences=[intr, e2], non_sequences=None)

    #     V_term = T.batched_dot(T.concatenate([e1, e2], axis=1), self.V)

    #     return self.activation(W_term + V_term + self.b)

    def tensors_for_relation(self, rel):
        return self.W[rel], self.V[rel]

    def apply(self, W_rel, V_rel, e1, e2):
        return T.dot(T.dot(e1, W_rel), e2) + T.dot(V_rel, T.concatenate([e1, e2])) + self.b

    def updates_symbolic(self, cost, rel, W_rel, V_rel, learning_rate):
        # cost: a symbolic cost variable to be differentiated
        # rel: the index of the relation, used to choose correct tensor W
        # and matrix V
        # W_rel: the subtensor of W corresponding to rel
        # V_rel: the submatrix of V corresponding to rel
        # learning_rate: how much to update the params prop to the cost gradient
        dW, dV, db = T.grad(cost, [W_rel, V_rel, self.b])
        return [(self.b, self.b + -learning_rate * db),
                (self.W, T.inc_subtensor(self.W[rel], -learning_rate * dW)),
                (self.V, T.inc_subtensor(self.V[rel], -learning_rate * dV))]

class NeuralTensorNetwork(EZPickle):
    SHARED = ['learning_rate']
    OTHERS = ['epoch',
              'n_hidden',
              'n_rel',
              'dimensions',
              'embedding_layer',
              'tensor_layer',
              'output_layer',
              'mode',
              'vocab_size',
              'other_params']

    def __init__(self, rng, vocab_size, n_rel, dimensions, n_hidden, other_params=None, initial_embeddings=None, learning_rate=0.01, mode='FAST_RUN'):
        if other_params is None:
            other_params = {}

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions)

        tensor_layer = TensorLayer(rng, n_rel, dimensions, n_hidden)

        output_layer = LinearScalarResponse(n_hidden)

        # store attributes
        self.init_params(epoch=0,
                         learning_rate=learning_rate,
                         n_rel=n_rel,
                         dimensions=dimensions,
                         n_hidden=n_hidden,
                         mode=mode,
                         vocab_size=vocab_size,
                         other_params=other_params,
                         embedding_layer=embedding_layer,
                         tensor_layer=tensor_layer,
                         output_layer=output_layer)


    def init_params(self, **kwargs):
        super(NeuralTensorNetwork, self).init_params(**kwargs)
        # wire the network
        self.make_functions()

    def inc_epoch(self):
        self.epoch += 1

    def apply(self, e1_index, e2_index, rel_index):
        e1 = self.embedding_layer.embeddings_for_indices(e1_index)
        e2 = self.embedding_layer.embeddings_for_indices(e2_index)
        W_rel, V_rel = self.tensor_layer.tensors_for_relation(rel_index)

        tensor_output = self.tensor_layer.apply(W_rel, V_rel, e1, e2)

        output = self.output_layer.apply(tensor_output)
        return output, e1, e2, W_rel, V_rel

    def embed_indices_symbolic(self, indices):
        return self.embedding_layer.embed_indices_symbolic(indices)

    def make_functions(self):
        # training function: take an entity, rel, entity triple and return
        # the cost
        e1_index_good = T.lscalar('e1_index_good')
        rel_index_good = T.lscalar('rel_index_good')
        e2_index_good = T.lscalar('e2_index_good')
        good_score, e1_good, e2_good, W_rel_good, V_rel_good = self.apply(e1_index_good, e2_index_good, rel_index_good)


        e1_index_bad = T.lscalar('e1_index_bad')
        rel_index_bad = T.lscalar('rel_index_bad')
        e2_index_bad = T.lscalar('e2_index_bad')
        bad_score, e1_bad, e2_bad, W_rel_bad, V_rel_bad = self.apply(e1_index_bad, e2_index_bad, rel_index_bad)

        cost = T.clip(1 - good_score + bad_score, 0, np.inf)

        # embedding gradient and updates
        embedding_indices = T.stack(e1_index_good, e2_index_good, e1_index_bad, e2_index_bad)
        dembeddings = T.stack(*T.grad(cost, [e1_good, e2_good, e1_bad, e2_bad]))

        embedding_updates =  [(self.embedding_layer.embedding, T.inc_subtensor(self.embedding_layer.embedding[embedding_indices],
                                                                              -self.learning_rate * dembeddings))]

        # tensor gradient and updates
        dW =  T.stack(*T.grad(cost, [W_rel_good, W_rel_bad]))
        dV = T.stack(*T.grad(cost, [V_rel_good, V_rel_bad]))
        tensor_indices = T.stack(rel_index_good, rel_index_bad)
        tensor_updates = [
            (self.tensor_layer.W, T.inc_subtensor(self.tensor_layer.W[tensor_indices],
                                                  -self.learning_rate * dW)),
            (self.tensor_layer.V, T.inc_subtensor(self.tensor_layer.V[tensor_indices],
                                                  -self.learning_rate * dV))
        ]


        output_updates = self.output_layer.updates_symbolic(cost, self.learning_rate)

        updates = embedding_updates + tensor_updates + output_updates


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

        ntn_model = NeuralTensorNetwork(
            rng=rng,
            vocab_size=N_synsets,
            n_rel=len(relationships.relationships),
            dimensions=50,
            n_hidden=50,
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
            b_bad = data_rng.randint(N_synsets)

            # calculate the weight as a function of the correct symbols and error symbols
            cost = ntn_model.train(a,b,rel,a,b_bad,rel)
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
