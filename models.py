import theano
import theano.tensor as T
import numpy as np
from scipy.spatial.distance import cdist

class Picklable(object):
    def _nonshared_attrs(self):
        # should be overridden by subclasses to return a list of strings, which
        # will be the names of object attributes that should be pickled
        pass

    def _shared_attrs(self):
        # should be overridden by subclasses to return a list of strings, which
        # will be the names of theano shared variable object attributes that
        # should be pickled
        pass

    def _initialize(self):
        # override this with any code that should be run after unpickling, e.g.
        # code that relies on the read parameters
        pass

    def _set_attrs(self, **kwargs):
        for param in self._shared_attrs():
            if type(param) is tuple:
                name, default = param
            else:
                name, default = param, None
            try:
                setattr(self, name, theano.shared(kwargs.get(name, default), name=name))
            except TypeError as e: # in case we stored the shared variable, get its current value
                print e
                setattr(self, name, theano.shared(kwargs.get(name, default).get_value(), name=name))
        for param in self._nonshared_attrs():
            if type(param) is tuple:
                name, default = param
            else:
                name, default = param, None
            setattr(self, name, kwargs.get(name, default))

    def __setstate__(self, state):
        self._set_attrs(**state)
        self._initialize()

    def __getstate__(self):
        state = {}
        for val in self._nonshared_attrs():
            name = val[0] if type(val) is tuple else val
            state[name] = getattr(self, name)
        for val in self._shared_attrs():
            name = val[0] if type(val) is tuple else val
            state[name] = getattr(self, name).get_value()
        return state

class VectorEmbeddings(object):
    @property
    def embeddings(self):
        pass

    def most_similar_to(self, embedding, metric='cosine', top_n=10, **kwargs):
        C = cdist(embedding[np.newaxis,:], self.embeddings, metric, **kwargs)
        sims = C[0]
        return [(i, sims[i]) for i in np.argsort(sims)[:top_n]]

    def most_similar_embeddings(self, index, metric='cosine', top_n=10, **kwargs):
        this_embedding = self.embeddings[index]
        return self.most_similar_to(this_embedding, metric=metric,top_n=top_n, **kwargs)

class EmbeddingLayer(Picklable, VectorEmbeddings):
    def _nonshared_attrs(self):
        return [('mode', 'FAST_RUN'),
                'vocab_size',
                'dimensions',
                ]

    def _shared_attrs(self):
        return ['embedding']

    def _initialize(self):
        self.params = []

    def __init__(self, rng, vocab_size, dimensions, initial_embedding_range=0.01, initial_embeddings=None, mode='FAST_RUN'):
        """ Initialize the parameters of the embedding layer

        :type rng: nympy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type vocab_size: int
        :param vocab_size: the number of discrete items to be embedded
        in the distributed representation

        :type dimensions: int
        :param dimensions: the number of dimensions in the distributed
        representation.

        """

        embedding_shape = (vocab_size, dimensions)
        if initial_embeddings is None:
            initial_embeddings = np.asarray(rng.uniform(
                    low=-initial_embedding_range / 2.,
                    high=initial_embedding_range / 2.,
                    size=embedding_shape),
                dtype=theano.config.floatX)
        else:
            assert initial_embeddings.shape == embedding_shape

        self._set_attrs(vocab_size=vocab_size,
                        dimensions=dimensions,
                        mode=mode,
                        embedding=initial_embeddings)
        self._initialize()

    def __call__(self, index):
        return self.embedding[index]

    def updates(self, cost, index_list, embedding_list, learning_rate):
        # cost: a symbolic cost function
        # index_list: a list of indices into the matrix
        # embedding_list: a list of symbolic embeddings, one for each index in
        # index_list, to be updated with gradient descent
        # learning_rate: the learning rate (multiplicative constant) for
        # gradient descent
        embedding_indices = T.stack(*index_list)
        dembeddings = T.stack(*T.grad(cost, embedding_list))
        return [(self.embedding, T.inc_subtensor(self.embedding[embedding_indices],
                                                 -learning_rate * dembeddings))]

    @property
    def embeddings(self):
        return self.embedding.get_value()

class LinearScalarResponse(Picklable):
    def _nonshared_attrs(self):
        return ['n_in']

    def _shared_attrs(self):
        return ['W', 'b']

    def _initialize(self):
        self.params = [self.W, self.b]

    def __init__(self, n_in):
        # init the weights W as a vector of zeros
        W_values = np.zeros((n_in,), dtype=theano.config.floatX)

        # init the basis as a scalar, 0
        b_values = np.cast[theano.config.floatX](0.0)

        self._set_attrs(W=W_values, b=b_values, n_in=n_in)
        self._initialize()

    def __call__(self, x):
        return T.dot(x, self.W) + self.b

    def updates(self, cost, learning_rate):
        dW, db = T.grad(cost, [self.W, self.b])
        return [(self.W, self.W - learning_rate * dW),
                (self.b, self.b - learning_rate * db)]

class HiddenLayer(Picklable):
    def _nonshared_attrs(self):
        return ['activation']

    def _shared_attrs(self):
        return ['W', 'b']

    def _initialize(self):
        self.params = [self.W, self.b]

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

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform is converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note: optimal init of weights depends on the activation function
        # used (among other things). Results in Xavier10 suggest using
        # 4 times larger initial weights for sigmoid compared to tanh
        if n_out > 1:
            size = (n_in, n_out)
        else:
            size = (n_in,)
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=size), dtype=theano.config.floatX)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        if n_out > 1:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        else:
            b_values = np.cast[theano.config.floatX](0.0)

        self._set_attrs(W=W_values, b=b_values, activation=activation)
        self._initialize()

    def __call__(self, x):
        return self.activation(T.dot(x, self.W) + self.b)

    def updates(self, cost, learning_rate):
        dW, db = T.grad(cost, [self.W, self.b])
        return [(self.W, self.W - learning_rate * dW),
                (self.b, self.b - learning_rate * db)]

class SequenceScoringNN(Picklable, VectorEmbeddings):
    def _nonshared_attrs(self):
        return ['n_hidden',
                'other_params',
                'blocks_trained',
                ('mode', 'FAST_RUN'),
                'embedding_layer',
                'hidden_layer',
                'output_layer',
                'dimensions',
                'sequence_length',
                'vocab_size']

    def _shared_attrs(self):
        return [('learning_rate', 0.01)]

    def _initialize(self):
        self.params =  self.hidden_layer.params + self.output_layer.params
        self.train = self._make_training()
        self.score = self._make_scoring()

    @property
    def embeddings(self):
        return self.embedding_layer.embeddings

    def __init__(self, rng, vocab_size,  dimensions,  sequence_length, n_hidden, other_params=None, initial_embeddings=None, mode='FAST_RUN', learning_rate=0.01):
        # initialize parameters
        if other_params is None:
            other_params = {}
        blocks_trained = 0

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions,
                                         initial_embeddings=initial_embeddings)

        hidden_layer = HiddenLayer(rng=rng,
                                   n_in=dimensions * sequence_length,
                                   n_out=n_hidden,
                                   activation=T.nnet.sigmoid)

        output_layer = LinearScalarResponse(n_in=n_hidden)

        self._set_attrs(n_hidden=n_hidden,
                         other_params=other_params,
                         blocks_trained=blocks_trained,
                         mode=mode,
                         embedding_layer=embedding_layer,
                         hidden_layer=hidden_layer,
                         output_layer=output_layer,
                         learning_rate=learning_rate,
                         dimensions=dimensions,
                         sequence_length=sequence_length,
                         vocab_size=vocab_size)
        self._initialize()


    def __call__(self, index_list):
        """
        embed the given sequence of indices, run the embeddings through the network,
        and return the score and the list of embeddings
        The list of embeddings is returned for use in differentiation
        """
        embeddings = [self.embedding_layer(index) for index in index_list]
        embedded_sequence = T.concatenate(embeddings)
        return self.output_layer(self.hidden_layer(embedded_sequence)), embeddings

    def updates(self, cost, index_list, embedding_list, learning_rate):
        return self.embedding_layer.updates(cost, index_list, embedding_list, learning_rate)\
                + self.hidden_layer.updates(cost, learning_rate)\
                + self.output_layer.updates(cost, learning_rate)

    def cost(self, correct_index_list, error_index_list):
        correct_score, correct_embeddings = self(correct_index_list)
        error_score, error_embeddings = self(error_index_list)
        cost = T.clip(1 - correct_score + error_score, 0, np.inf)
        return cost, correct_embeddings, error_embeddings

    def _index_variables(self, basename='index'):
        return T.lscalars(*['%s_%d' % (basename, i)
                            for i in xrange(self.sequence_length)])

    def _make_training(self):
        """
        compile and return symbolic theano function for training (including update of embedding and weights),
        The compiled function will take two vectors of ints, each of length
        sequence_length, which are the indices of the words in the good and bad
        ngrams
        """
        correct_indices = self._index_variables('correct')
        error_indices = self._index_variables('error')

        cost, correct_embeddings, error_embeddings = self.cost(correct_indices, error_indices)
        updates = self.updates(cost,
                               correct_indices + error_indices,
                               correct_embeddings + error_embeddings,
                               self.learning_rate)

        inputs = correct_indices + error_indices
        outputs = cost
        return theano.function(inputs=inputs, outputs=outputs, updates=updates, mode=self.mode)

    def _make_scoring(self):
        indices = self._index_variables('index')
        score, embeddings = self(indices)
        return theano.function(inputs=indices, outputs=score, mode=self.mode)
