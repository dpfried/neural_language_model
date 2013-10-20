from wordnet_rels import Relationships
from model_new import EmbeddingLayer, EZPickle, LinearScalarResponse
import theano.tensor as T
import numpy as np
import theano

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
    OTHERS = ['n_hidden',
              'n_rel',
              'dimensions',
              'embedding_layer',
              'tensor_layer',
              'output_layer',
              'mode',
              'vocabulary',
              'vocab_size',
              'other_params']

    def __init__(self, rng, vocabulary, n_rel, dimensions, n_hidden, other_params=None, initial_embeddings=None, learning_rate=0.01, mode='FAST_RUN'):
        if other_params is None:
            other_params = {}

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=len(vocabulary),
                                         dimensions=dimensions)

        tensor_layer = TensorLayer(rng, n_rel, dimensions, n_hidden)

        output_layer = LinearScalarResponse(n_hidden)

        # store attributes
        self.init_params(learning_rate=learning_rate,
                         n_rel=n_rel,
                         dimensions=dimensions,
                         n_hidden=n_hidden,
                         mode=mode,
                         vocabulary=vocabulary,
                         vocab_size=len(vocabulary),
                         other_params=other_params,
                         embedding_layer=embedding_layer,
                         tensor_layer=tensor_layer,
                         output_layer=output_layer)

        # wire the network
        self.make_functions()

    def apply(self, e1_index, e2_index, rel_index):
        e1 = self.embedding_layer.embeddings_for_indices(e1_index)
        e2 = self.embedding_layer.embeddings_for_indices(e2_index)
        W_rel, V_rel = self.tensor_layer.tensors_for_relation(rel_index)

        tensor_output = self.tensor_layer.apply(W_rel, V_rel, e1, e2)

        output = self.output_layer.apply(tensor_output)
        return output, e1, e2, W_rel, V_rel

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
        dW_1 = T.grad(cost, W_rel_good)
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
                                        updates=updates)

        self.test = theano.function([e1_index_good, e2_index_good, rel_index_good],
                                    good_score)


if __name__ == "__main__":
    relationships = Relationships()

    num_training = int(relationships.N * 0.9)
    training = relationships.data[:num_training]
    testing = relationships.data[num_training:]
