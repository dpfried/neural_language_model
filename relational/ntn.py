from wordnet_rels import Relationships
from model_new import EmbeddingLayer, EZPickle, LinearScalarResponse
import theano.tensor as T

class TensorLayer(EZPickle):
    # compute f(
    SHARED = ['W', 'V', 'b']
    OTHERS = ['n_rel', 'n_in', 'n_out']

    # n_rel: number of relations (number of 3d tensors)
    # n_in: input dimensionality
    # n_out: output dimensionality
    # W: a 4d tensor containing a set of 3d relationship tensors. Dimensions: (n_rel, n_out, n_in, n_in)
    # V: a 3d tensor containing a set of 2d hidden layer matrics. Dimensions: (n_rel, 2*n_in, n_out)

    def __init__(self, rng, n_rel, n_in, n_out, activation=T.tanh):
        W_size = (n_rel, n_out, n_in, n_in)
        # what to change this heuristic to?
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=W_size), dtype=theano.config.floatX)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        V_size = (n_rel, 2*n_in, n_out)
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
              'dimensions',
              'embedding_layer',
              'tensor_layer',
              'output_layer',
              'mode',
              'vocabulary',
              'vocab_size']

    def __init__(self, rng, vocabulary, dimensions, n_hidden, other_params=None, initial_embeddings=None, learning_rate=0.01, mode='FAST_RUN'):
        if other_params is None:
            other_params = {}

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions)

        tensor_layer = TensorLayer(rng, dimensions, dimensions, n_hidden)

        output_layer = LinearScalarResponse(n_hidden)


        self.init_params(learning_rate=learning_rate,
                         n_hidden=n_hidden,
                         dimensions=dimensions,
                         mode=mode,
                         vocabulary=vocabulary,
                         vocab_size=len(vocabulary)
                         embedding_layer=embedding_layer,
                         tensor_layer=tensor_layer,
                         output_layer=output_layer)

        def init_params(self, **kwargs):
            super(NeuralTensorNetwork, self).init_params(**kwargs)
            self.make_functions()

        def make_functions(self):
            # TODO here

if __name__ == "__main__":
    relationships = Relationships()

    num_training = int(relationships.N * 0.9)
    training = relationships.data[:num_training]
    testing = relationships.data[num_training:]
