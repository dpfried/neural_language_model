import theano
import theano.tensor as T
import numpy as np

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

        self.embedding = initial_embedding
        # self.embedding = theano.shared(value=np.eye(vocab_size, dimensions), name='embedding')

        self.params = []

    def flatten_embeddings(self, embeddings):
        """take a list of embedding vectors and put them into the appropriate format to feed to the next layer"""
        return T.concatenate(embeddings)

    def embeddings_from_symbols(self, symbol_indices):
        return self.embedding[symbol_indices]

    def update_embeddings(self, symbol_indices, updates):
        self.embedding[symbol_indices] += np.array(updates)

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
    def __init__(self, rng, vocab_size, dimensions, sequence_length, n_hidden, L1_reg, L2_reg, other_params={}):
        # initialize parameters
        self.rng = rng
        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.other_params = other_params

        self._build_layers()
        self._build_functions()

    def _build_layers(self):
        self.embedding_layer = EmbeddingLayer(self.rng, vocab_size=self.vocab_size,
                                              dimensions=self.dimensions,
                                              sequence_length=self.sequence_length)

        self.hidden_layer = HiddenLayer(rng=self.rng,
                                        n_in=self.dimensions * self.sequence_length,
                                        n_out=self.n_hidden,
                                        activation=T.nnet.sigmoid)

        self.output_layer = LinearScalarResponse(n_in=self.n_hidden)

        self.layer_stack = [self.hidden_layer, self.output_layer]

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.output_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.output_layer.W ** 2).sum()

        # connect the computation graph together from input to regularized cost
        # function

    def score_symbolic(self, sequence_embedding):
        return reduce(lambda layer_input, layer: layer.apply(layer_input), self.layer_stack, sequence_embedding)

    def compare_symbolic(self, correct_sequence_embedding, error_sequence_embedding):
        return T.clip(1 - self.score_symbolic(correct_sequence_embedding) + self.score_symbolic(error_sequence_embedding), 0, np.inf)

    def _build_functions(self):
        # create symbolic variables for correct and error input
        correct_embeddings = [T.vector(name='correct_embedding%i' % i) for i in range(self.sequence_length)]
        error_embeddings = [T.vector(name='error_embedding%i' % i) for i in range(self.sequence_length)]

        correct_sequence_embedding = self.embedding_layer.flatten_embeddings(correct_embeddings)
        error_sequence_embedding = self.embedding_layer.flatten_embeddings(error_embeddings)

        cost = self.compare_symbolic(correct_sequence_embedding, error_sequence_embedding) + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        weighted_learning_rate = T.scalar(name='weighted_learning_rate')

        # update the params of the model using the gradients
        updates = [(param, param - weighted_learning_rate * T.grad(cost, param))
                   for param in self.hidden_layer.params + self.output_layer.params]

        dcorrect_embeddings = T.grad(cost, correct_embeddings)
        derror_embeddings = T.grad(cost, error_embeddings)

        inputs = correct_embeddings + error_embeddings + [weighted_learning_rate]

        self.training_function = theano.function(inputs=inputs,
                                                 outputs=dcorrect_embeddings + derror_embeddings + [cost],
                                                 updates=updates)

        embeddings = [T.vector(name='embedding%i' % i) for i in range(self.sequence_length)]

        self.score_ngram = theano.function(inputs=embeddings,
                                           outputs=self.score_symbolic(self.embedding_layer.flatten_embeddings(embeddings)))

    def train(self, correct_sequence, error_sequence, weighted_learning_rate=0.01):
        correct_embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in correct_sequence]
        error_embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in error_sequence]

        outputs = self.training_function(*(correct_embeddings + error_embeddings + [weighted_learning_rate]))

        correct_grads = outputs[:self.sequence_length]
        error_grads = outputs[self.sequence_length:-1]

        cost = outputs[-1]

        correct_updates = -weighted_learning_rate * np.array(correct_grads)
        error_updates = -weighted_learning_rate * np.array(error_grads)

        self.embedding_layer.update_embeddings(correct_sequence, correct_updates)
        self.embedding_layer.update_embeddings(error_sequence, error_updates)

        return cost

    def score(self, sequence):
        embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in sequence]
        return self.score_ngram(*embeddings)
