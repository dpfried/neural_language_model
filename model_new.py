import theano
import theano.tensor as T
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

theano.config.on_unused_input = 'warn' # for unpickling old models

class EZPickle(object):
    SHARED = [] # should be overridden by subclasses
    OTHERS = [] # should be overridden by subclasses
    def init_params(self, **kwargs):
        for param in self.SHARED:
            if type(param) is tuple:
                name, default = param
            else:
                name, default = param, None
            setattr(self, name, theano.shared(kwargs.get(name, default), name=name))
        for param in self.OTHERS:
            if type(param) is tuple:
                name, default = param
            else:
                name, default = param, None
            # print 'setting %s to %s' % (name, kwargs.get(name, default))
            setattr(self, name, kwargs.get(name, default))

    def __setstate__(self, state):
        self.init_params(**state)

    def __getstate__(self):
        state = {}
        for val in self.SHARED:
            name = val[0] if type(val) is tuple else val
            state[name] = getattr(self, name).get_value()
        for val in self.OTHERS:
            name = val[0] if type(val) is tuple else val
            state[name] = getattr(self, name)
        return state

class EmbeddingTrainer(object):
    """contains a vocabulary with some associated embeddings, of given dimensions. Designed to be implemented by objects with train and update methods"""
    def __init__(self, rng, vocabulary, dimensions):
        self.rng = rng
        self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)

        try:
            self.symbol_to_word = defaultdict(default_word, dict(enumerate(self.vocabulary)))
            self.symbol_to_word[0] = default_word()
            self.word_to_symbol = defaultdict(int, dict((word, index) for index, word in enumerate(self.vocabulary)))
        except:
            print 'model has already defined symbol lookup tables'

        self.dimensions = dimensions

    def get_embeddings(self):
        pass

    def dump_embeddings(self, filename, normalize=True, precision=8):
        format_str = '%%0.%if' % precision
        float_to_str = lambda f: format_str % f
        with open(filename, 'w') as f:
            for index, embedding in enumerate(self.get_embeddings()):
                # skip RARE
                if index == 0:
                    continue
                if normalize:
                    vector = embedding / np.sqrt(np.dot(embedding, embedding))
                else:
                    vector = embedding
                vector_string_rep = ' '.join(map(float_to_str, vector))
                f.write('%s %s\n' % (self.symbol_to_word[index], vector_string_rep))

    def get_embedding(self, word, normalize_components=False, include_synsets=None):
        """include_synsets not used but need it for interface to evaluation scripts"""
        if word not in self.word_to_symbol:
            print 'warning: %s not in vocab' % word
        word_embedding = self.get_embeddings()[self.word_to_symbol[word]]
        if normalize_components:
            return word_embedding / np.linalg.norm(word_embedding, 2)
        else:
            return word_embedding

class EmbeddingContainer(EmbeddingTrainer):
    def __init__(self, vocabulary, embeddings):
        vocab_size, dimensions = embeddings.shape
        super(EmbeddingContainer, self).__init__(None, vocabulary, dimensions)
        self.embedding = embedding

    def get_embeddings(self):
        return self.embeddings.get_value()

class EmbeddingLayer(EZPickle):
    SHARED = ['embedding']

    OTHERS = [('mode', 'FAST_RUN'),
              'vocab_size',
              'dimensions',
              'sequence_length']

    def __init__(self, rng, vocab_size, dimensions, sequence_length=5, initial_embedding_range=0.01, initial_embeddings=None, mode='FAST_RUN'):
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

        embedding_shape = (vocab_size, dimensions)
        if initial_embeddings is None:
            initial_embeddings = np.asarray(rng.uniform(
                    low=-initial_embedding_range / 2.,
                    high=initial_embedding_range / 2.,
                    size=embedding_shape),
                dtype=theano.config.floatX)
        else:
            assert initial_embeddings.shape == embedding_shape

        self.init_params(vocab_size=vocab_size,
                         dimensions=dimensions,
                         sequence_length=sequence_length,
                         mode=mode,
                         embedding=initial_embeddings)

    def init_params(self, **kwargs):
        super(EmbeddingLayer, self).init_params(**kwargs)
        self._build_functions()

        self.params = []

    def _build_functions(self):
        symbol_index = T.scalar(dtype='int32')
        self.embedding_from_symbol = theano.function([symbol_index],
                                                     self.embedding[symbol_index],
                                                     mode=self.mode)

    def embeddings_for_indices(self, indices):
        return self.embedding[indices]

    def embed_indices_symbolic(self, indices):
        return T.flatten(self.embeddings_for_indices(indices))

    def updates_symbolic(self, cost, indices, embeddings, learning_rate):
        # cost: a symbolic cost function
        # indices: a vector of length N
        # embeddings: a matrix of dimensions N x dimensions
        # learning_rate: the learning rate (multiplicative constant) for
        # gradient descent
        dembeddings = T.grad(cost, embeddings)
        return [(self.embedding, T.inc_subtensor(self.embedding[indices],
                                                 -learning_rate * dembeddings))]

    def most_similar_embeddings(self, index, metric='cosine', top_n=10, **kwargs):
        embeddings = self.embedding.get_value()
        this_embedding = embeddings[index]
        C = cdist(this_embedding[np.newaxis,:], embeddings, metric, **kwargs)
        sims = C[0]
        return [(i, sims[i]) for i in np.argsort(sims)[:top_n]]

class LinearScalarResponse(EZPickle):
    SHARED = ['W', 'b']

    OTHERS = ['n_in']

    def __init__(self, n_in):

        # init the weights W as a vector of zeros
        W_values = np.zeros((n_in,), dtype=theano.config.floatX)

        # init the basis as a scalar, 0
        b_values = np.cast[theano.config.floatX](0.0)

        self.init_params(W=W_values, b=b_values, n_in=n_in)

    def init_params(self, **kwargs):
        super(LinearScalarResponse, self).init_params(**kwargs)

        self.params = [self.W, self.b]

    def apply(self, input):
        return T.dot(input, self.W) + self.b

    def updates_symbolic(self, cost, learning_rate):
        dW, db = T.grad(cost, [self.W, self.b])
        return [(self.W, self.W - learning_rate * dW),
                (self.b, self.b - learning_rate * db)]

class HiddenLayer(EZPickle):
    SHARED = ['W', 'b']

    OTHERS = ['activation']

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

        self.init_params(W=W_values, b=b_values, activation=activation)

    def init_params(self, **kwargs):
        super(HiddenLayer, self).init_params(**kwargs)

        # params should only be those which need to be updated in gradient
        # descent
        self.params = [self.W, self.b]

    def apply(self, input):
        return self.activation(T.dot(input, self.W) + self.b)

    def updates_symbolic(self, cost, learning_rate):
        dW, db = T.grad(cost, [self.W, self.b])
        return [(self.W, self.W - learning_rate * dW),
                (self.b, self.b - learning_rate * db)]

class NLM(EmbeddingTrainer, EZPickle):
    SHARED = [('learning_rate', 0.01)]

    OTHERS = ['n_hidden',
              'other_params',
              'blocks_trained',
              ('mode', 'FAST_RUN'),
              'embedding_layer',
              'hidden_layer',
              'output_layer',
              'dimensions',
              'sequence_length',
              'vocab_size']

    def __init__(self, rng, vocabulary,  dimensions,  sequence_length, n_hidden, other_params=None, initial_embeddings=None, mode='FAST_RUN', learning_rate=0.01):
        super(NLM, self).__init__(rng, vocabulary, dimensions)
        # initialize parameters
        if other_params is None:
            other_params = {}
        blocks_trained = 0

        vocab_size = len(vocabulary)

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        embedding_layer = EmbeddingLayer(rng, vocab_size=vocab_size,
                                         dimensions=dimensions,
                                         sequence_length=sequence_length)

        hidden_layer = HiddenLayer(rng=rng,
                                   n_in=dimensions * sequence_length,
                                   n_out=n_hidden,
                                   activation=T.nnet.sigmoid)

        output_layer = LinearScalarResponse(n_in=n_hidden)

        self.init_params(n_hidden=n_hidden,
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


    def init_params(self, **kwargs):
        super(NLM, self).init_params(**kwargs)
        self.make_functions()

    # this is ugly, use it because we didn't always save the vocab
    def _get_vocabulary(self):
        try:
            return self.vocabulary
        except:
            import ngrams
            ngram_reader = ngrams.NgramReader(self.other_params['ngram_filename'], vocab_size=self.vocab_size)
            self.vocabulary = ngram_reader.word_array
            return self.vocabulary

    @property
    def word_to_symbol(self):
        try:
            return self._word_to_symbol
        except AttributeError:
            self._word_to_symbol = defaultdict(int, dict((word, index)
                                                         for index, word in enumerate(self._get_vocabulary())))
            return self._word_to_symbol

    @property
    def symbol_to_word(self):
        try:
            return self._symbol_to_word
        except AttributeError:
            self._symbol_to_word = defaultdict(lambda : '*UNKNOWN*', dict(enumerate(self._get_vocabulary())))
            self._symbol_to_word[0] = '*UNKNOWN*'
            return self._symbol_to_word

    def get_params(self):
        return self.hidden_layer.params + self.output_layer.params

    def score_embedding_symbolic(self, embedding):
        return reduce(lambda layer_input, layer: layer.apply(layer_input), self.layer_stack, embedding)

    def score_indices_symbolic(self, sequence_indices):
        return self.score_embedding_symbolic(self.embedding_layer.embed_indices_symbolic(sequence_indices))

    def symbolic_indices(self, basename):
        return T.vector(name = basename, dtype='int32')
        # return [T.scalar(name='%s%i' % (basename, i), dtype='int32') for i in range(self.sequence_length)]

    def embed_indices_symbolic(self, indices):
        return self.embedding_layer.embed_indices_symbolic(indices)

    def cost_from_embeddings_symbolic(self, correct_sequence_embedding, error_sequence_embedding):
        return T.clip(1 - self.score_embedding_symbolic(correct_sequence_embedding) + self.score_embedding_symbolic(error_sequence_embedding), 0, np.inf)

    def updates_symbolic(self, cost, correct_indices, error_indices, correct_sequence_embedding, error_sequence_embedding):
        param_updates = self.hidden_layer.updates_symbolic(cost, self.learning_rate) + self.output_layer.updates_symbolic(cost, self.learning_rate)

        embeddings = T.reshape(T.concatenate([correct_sequence_embedding, error_sequence_embedding]),
                               (self.sequence_length * 2, self.dimensions))
        indices = T.concatenate([correct_indices, error_indices])

        embedding_updates = self.embedding_layer.updates_symbolic(cost, indices, embeddings, self.learning_rate)

        return param_updates + embedding_updates

    def make_theano_training(self):
        """
        compile and return symbolic theano function for training (including update of embedding and weights),
        The compiled function will take two vectors of ints, each of length
        sequence_length, which are the indices of the words in the good and bad
        ngrams
        """
        correct_indices = self.symbolic_indices('correct_index')
        error_indices = self.symbolic_indices('error_index')

        correct_sequence_embedding = self.embed_indices_symbolic(correct_indices)
        error_sequence_embedding = self.embed_indices_symbolic(error_indices)

        cost = self.cost_from_embeddings_symbolic(correct_sequence_embedding, error_sequence_embedding)

        inputs = [correct_indices, error_indices]
        outputs = cost
        updates = self.updates_symbolic(cost, correct_indices, error_indices, correct_sequence_embedding, error_sequence_embedding)

        return theano.function(inputs=inputs, outputs=outputs, updates=updates, mode=self.mode)

    def make_theano_scoring(self):
        indices = self.symbolic_indices('index')
        return theano.function(inputs=[indices], outputs=self.score_indices_symbolic(indices), mode=self.mode)

    def make_functions(self):
        # create symbolic variables for correct and error input
        self.layer_stack = [self.hidden_layer, self.output_layer]

        self.params = self.get_params()

        self.train = self.make_theano_training()
        self.score = self.make_theano_scoring()

    def get_embeddings(self):
        return self.embedding_layer.get_embeddings()
