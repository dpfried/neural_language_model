import theano
import theano.tensor as T
import numpy as np
from utils import grouper
from collections import defaultdict
from scipy.spatial.distance import cdist

def _default_word():
    '''have to do this as a module level function b/c otherwise pickle won't
    let us save the defaultdict inside EmbeddingTrainer'''
    return '*UNKNOWN*'

default_word = _default_word

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
            for index, embedding in enumerate(self.embeddings()):
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
        self.embeddings = embeddings

    def get_embeddings(self):
        return self.embeddings.get_value()

class EmbeddingLayer(object):
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

        self.mode = mode

        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.sequence_length = sequence_length

        embedding_shape = (self.vocab_size, self.dimensions)
        if initial_embeddings is None:
            initial_embeddings = np.asarray(rng.uniform(
                    low=-initial_embedding_range / 2.,
                    high=initial_embedding_range / 2.,
                    size=embedding_shape),
                dtype=theano.config.floatX)
        else:
            assert initial_embeddings.shape == embedding_shape

        self.embeddings = theano.shared(value=initial_embeddings, name='embedding')

        self._build_functions()

        self.params = []

    def _build_functions(self):
        symbol_index = T.scalar(dtype='int32')
        embedding_delta = T.vector(dtype=theano.config.floatX)
        inc_embedding = T.inc_subtensor(self.embeddings[symbol_index], embedding_delta)
        self.update_embedding = theano.function([symbol_index, embedding_delta],
                                                updates=[(self.embeddings, inc_embedding)],
                                                mode=self.mode)

        self.embedding_from_symbol = theano.function([symbol_index],
                                                     self.embeddings[symbol_index],
                                                     mode=self.mode)

        indices = T.vector(name='indices', dtype='int32')
        delta = T.matrix(dtype=theano.config.floatX)
        update = (self.embeddings, T.inc_subtensor(self.embeddings[indices], delta))
        self.update_embeddings = theano.function([indices, delta], updates=[update], mode=self.mode)


    def embed_indices_symbolic(self, indices, num_indices=None):
        if num_indices is None:
            num_indices = self.sequence_length
        return T.reshape(self.embeddings[indices], (self.dimensions * num_indices,))

    def most_similar_embeddings(self, index, metric='cosine', top_n=10, **kwargs):
        embeddings = self.embeddings.get_value()
        embedding = embeddings[index]
        C = cdist(embedding[np.newaxis,:], embeddings, metric, **kwargs)
        sims = C[0]
        return [(i, sims[i]) for i in np.argsort(sims)[:top_n]]

class LinearScalarResponse(object):
    def __init__(self, n_in):
        self.n_in = n_in

        # init the weights W as a vector of zeros
        self.W = theano.shared(value=np.zeros((n_in,),
                                              dtype=theano.config.floatX), name='W')

        # init the basis as a scalar, 0
        self.b = theano.shared(value=np.cast[theano.config.floatX](0.0), name='b')

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

        self.W = theano.shared(value=W_values, name='W')

        if n_out > 1:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        else:
            b_values = np.cast[theano.config.floatX](0.0)
        self.b = theano.shared(value=b_values, name='b')

        # parameters of the model
        self.params = [self.W, self.b]

    def apply(self, input):
        return self.activation(T.dot(input, self.W) + self.b)

class NLM(EmbeddingTrainer):
    def __init__(self, rng, vocabulary,  dimensions,  sequence_length, n_hidden, other_params=None, initial_embeddings=None, mode='FAST_RUN', learning_rate=0.01):
        super(NLM, self).__init__(rng, vocabulary, dimensions)
        # initialize parameters
        if other_params is None:
            other_params = {}
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.other_params = other_params
        self.blocks_trained = 0

        self.learning_rate = theano.shared(value=np.cast[theano.config.floatX](learning_rate), name='learning_rate')

        self.mode = mode

        self._make_layers(initial_embeddings=initial_embeddings)
        self._make_functions()

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

    def _make_layers(self, initial_embeddings=None):
        self.embedding_layer = EmbeddingLayer(self.rng, vocab_size=self.vocab_size,
                                              dimensions=self.dimensions,
                                              sequence_length=self.sequence_length)

        self.hidden_layer = HiddenLayer(rng=self.rng,
                                        n_in=self.dimensions * self.sequence_length,
                                        n_out=self.n_hidden,
                                        activation=T.nnet.sigmoid)

        self.output_layer = LinearScalarResponse(n_in=self.n_hidden)

        self.params = self.get_params()

        self.layer_stack = [self.hidden_layer, self.output_layer]

    def get_params(self):
        return self.hidden_layer.params + self.output_layer.params

    def score_embedding_symbolic(self, embedding):
        return reduce(lambda layer_input, layer: layer.apply(layer_input), self.layer_stack, embedding)

    def score_indices_symbolic(self, sequence_indices):
        return self.score_embedding_symbolic(self.embedding_layer.embed_indices_symbolic(sequence_indices))

    def symbolic_indices(self, basename):
        return T.vector(name = basename, dtype='int32')
        # return [T.scalar(name='%s%i' % (basename, i), dtype='int32') for i in range(self.sequence_length)]

    def make_theano_weight_training(self):
        """
        compile and return symbolic theano function for training (including update of embedding and weights),
        given two lists of symbolic vars, each of which is a list of symbolic indices representing
        . First list is the list of indices for the training ngram, second is
        the list of indices for the corruption.
        """
        correct_indices = self.symbolic_indices('correct_index')
        error_indices = self.symbolic_indices('error_index')

        correct_sequence_embedding = self.embedding_layer.embed_indices_symbolic(correct_indices)
        error_sequence_embedding = self.embedding_layer.embed_indices_symbolic(error_indices)

        cost = T.clip(1 - self.score_embedding_symbolic(correct_sequence_embedding) + self.score_embedding_symbolic(error_sequence_embedding), 0, np.inf)

        param_updates = [(param, param - self.learning_rate * T.grad(cost, param))
                          for param in self.params]

        dcorrect = T.reshape(T.grad(cost, correct_sequence_embedding), (self.sequence_length, self.dimensions))
        derror = T.reshape(T.grad(cost, error_sequence_embedding), (self.sequence_length, self.dimensions))

        correct_update = -self.learning_rate * dcorrect
        error_update = -self.learning_rate * derror

        inputs = [correct_indices, error_indices]
        outputs = [correct_update, error_update, cost]

        return theano.function(inputs=inputs, outputs=outputs, updates=param_updates, mode=self.mode)

    def make_theano_scoring(self):
        indices = self.symbolic_indices('index')
        return theano.function(inputs=[indices], outputs=self.score_indices_symbolic(indices), mode=self.mode)

    def _make_functions(self):
        # create symbolic variables for correct and error input
        self.train_weights = self.make_theano_weight_training()
        self.score = self.make_theano_scoring()

    def train(self, correct_indices, error_indices):
        correct_update, error_update, cost = self.train_weights(correct_indices, error_indices)
        self.embedding_layer.update_embeddings(correct_indices, correct_update)
        self.embedding_layer.update_embeddings(error_indices, error_update)
        return cost

    def get_embeddings(self):
        return self.embedding_layer.get_embeddings()
