import theano
import theano.tensor as T
import numpy as np
import scipy

# code adapted from DeepLearning tutorial:
# deeplearning.net/tutorial/mlp.html

class EmbeddingLayer(object):
    def __init__(self, rng, vocab_size, dimensions, sequence_length=5, initial_embedding_range=0.01):
        """ Initialize the parameters of the embedding layer

        :type rng: nympy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type vocab_size: int
        :param vocab_size: the number of discrete items to be embedded
        in the distributed representation

        :type dimensions: int
        :param dimensions: the number of dimensions in the distributed
        representation.

        :type sequence_length: int
        :param sequence_length: the number of words in each n-gram
        """

        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.sequence_length = sequence_length

        initial_embedding = np.asarray(rng.uniform(
            low=-initial_embedding_range / 2.,
            high=initial_embedding_range / 2.,
            size=(self.vocab_size, self.dimensions)),
            dtype=theano.config.floatX)

        self.embedding = theano.shared(value=initial_embedding, name='embedding')
        # self.embedding = theano.shared(value=np.eye(vocab_size, dimensions), name='embedding')

        # params are those that should be updated in gradient descent and
        # are also serialized
        self.params = [self.embedding]

        # hyperparams are those that we'd serialize but aren't optimized in
        # gradient descent
        self.hyper_params = [self.vocab_size, self.dimensions, self.sequence_length]

        # memoize to use in one_hot_from_batch for lookup of one-hot encodings
        self.vocab_id = scipy.sparse.eye(vocab_size, vocab_size).tocsr()

    def apply(self, one_hot_input):
        # output: a dimension x sequence_length matrix
        # each row corresponds to the concatenated vectors of the
        # representations of the words in that n-gram (where n =
        # sequence_length)
        return T.flatten(T.dot(one_hot_input, self.embedding))

    def one_hot_from_symbols(self, symbol_indices):
        """returns a matrix of dimensions sequence_length x vocabulary_size that
        has a 1 in r,c if word_r is symbol_c, and 0 otherwise"""
        return self.vocab_id[symbol_indices].todense()

class LinearScalarResponse(object):
    def __init__(self, n_in):
        self.n_in = n_in

        # init the weights W as a vector of zeros
        self.W = theano.shared(value=np.zeros((n_in,),
                                              dtype=theano.config.floatX), name='W')

        # init the basis as a scalar, 0
        self.b = theano.shared(value=0., name='b')

        self.params = [self.W, self.b]

        self.hyper_params = [self.n_in]

    def apply(self, input):
        return T.dot(input, self.W) + self.b

class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, activation=T.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: fully connected units with sigmoidal
        activation function. Weight matrix has shape (n_in, n_out)
        and bias vector b has shape (n_out,).

        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: nympy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: a theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non Linearity to be applied in the hidden layer
        """
        self.activation = activation

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform is converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note: optimal init of weights depends on the activation function
        # used (among other things). Results in Xavier10 suggest using
        # 4 times larger initial weights for sigmoid compared to tanh
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name='W')

        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')

        # parameters of the model
        self.params = [self.W, self.b]

    def apply(self, input):
        return self.activation(T.dot(input, self.W) + self.b)

class NLM(object):
    def __init__(self, rng, vocab_size, dimensions, sequence_length, n_hidden):
        self.rng = rng
        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden

        self.embedding_layer = EmbeddingLayer(rng, vocab_size=vocab_size,
                                              dimensions=dimensions,
                                              sequence_length=sequence_length)

        self.hidden_layer = HiddenLayer(rng=rng,
                                        n_in=dimensions * sequence_length,
                                        n_out=n_hidden,
                                        activation=T.tanh)

        self.output_layer = LinearScalarResponse(n_in=n_hidden)

        self.layer_stack = [self.embedding_layer, self.hidden_layer, self.output_layer]

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.output_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.output_layer.W ** 2).sum()

        self.params = self.embedding_layer.params + self.hidden_layer.params + self.output_layer.params

    def score(self, input):
        return reduce(lambda layer_input, layer: layer.apply(layer_input), self.layer_stack, input)

    def cost(self, correct_input, error_input):
        return T.clip(1 - self.score(correct_input) + self.score(error_input), 0, np.inf)

    def one_hot_from_symbols(self, symbols):
        return self.embedding_layer.one_hot_from_symbols(symbols)
