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

    def get_embeddings():
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
        try:
            return self.embedding[symbol_indices]
        except IndexError as e:
            print symbol_indices
            raise e

    def update_embeddings(self, symbol_indices, updates):
        self.embedding[symbol_indices] += np.array(updates)

    def most_similar_embeddings(self, index, metric='cosine', top_n=10, **kwargs):
        embedding = self.embedding[index]
        C = cdist(embedding[np.newaxis,:], self.embedding, metric, **kwargs)
        sims = C[0]
        return [(i, sims[i]) for i in np.argsort(sims)[:top_n]]

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
            b_values= 0.0
        self.b = theano.shared(value=b_values, name='b')

        # parameters of the model
        self.params = [self.W, self.b]

    def apply(self, input):
        return self.activation(T.dot(input, self.W) + self.b)

class NLM(EmbeddingTrainer):
    def __init__(self, rng, vocabulary,  dimensions,  sequence_length, n_hidden, L1_reg, L2_reg, other_params=None):
        super(NLM, self).__init__(rng, vocabulary, dimensions)
        # initialize parameters
        if other_params is None:
            other_params = {}
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.other_params = other_params
        self.blocks_trained = 0

        self._make_layers()
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

    def _make_layers(self):
        self.embedding_layer = EmbeddingLayer(self.rng, vocab_size=self.vocab_size,
                                              dimensions=self.dimensions,
                                              sequence_length=self.sequence_length)

        self.hidden_layer = HiddenLayer(rng=self.rng,
                                        n_in=self.dimensions * self.sequence_length,
                                        n_out=self.n_hidden,
                                        activation=T.nnet.sigmoid)

        self.output_layer = LinearScalarResponse(n_in=self.n_hidden)

        self.params = self.hidden_layer.params + self.output_layer.params

        self.layer_stack = [self.hidden_layer, self.output_layer]

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.output_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.output_layer.W ** 2).sum()

    def score_symbolic(self, sequence_embedding):
        return reduce(lambda layer_input, layer: layer.apply(layer_input), self.layer_stack, sequence_embedding)

    def compare_symbolic(self, correct_sequence_embedding, error_sequence_embedding):
        return T.clip(1 - self.score_symbolic(correct_sequence_embedding) + self.score_symbolic(error_sequence_embedding), 0, np.inf)

    def loss(self, correct_embeddings, error_embeddings):
        correct_sequence_embedding = self.embedding_layer.flatten_embeddings(correct_embeddings)
        error_sequence_embedding = self.embedding_layer.flatten_embeddings(error_embeddings)

        return self.compare_symbolic(correct_sequence_embedding, error_sequence_embedding)

    # def compare_symbolic(self, correct_sequence_embedding, error_sequence_embedding, logistic_scaling_factor=1.0):
    #     score_difference = self.score_symbolic(correct_sequence_embedding) - self.score_symbolic(error_sequence_embedding)
    #     return T.log(1 + T.exp(logistic_scaling_factor * -1 * score_difference))

    def make_theano_training(self, correct_embeddings, error_embeddings, cost_addition=None):
        """
        compile and return symbolic theano function for training (including update of embedding and weights),
        given two lists of symbolic vars, each of which is a list of symbolic vectors representing
        word embeddings. First list is the list of embeddings for the training ngram, second is
        the list of embeddings for the corruption.
        Cost addition can be some constants or a function of theano vars (possibly shared)"""

        cost = self.loss(correct_embeddings, error_embeddings)

        if cost_addition is not None:
            cost += cost_addition

        weighted_learning_rate = T.scalar(name='weighted_learning_rate')

        # update the params of the model using the gradients
        updates = [(param, param - weighted_learning_rate * T.grad(cost, param))
                   for param in self.params]

        dcorrect_embeddings = T.grad(cost, correct_embeddings)
        derror_embeddings = T.grad(cost, error_embeddings)

        inputs = correct_embeddings + error_embeddings + [weighted_learning_rate]
        outputs = dcorrect_embeddings + derror_embeddings + [cost]

        return theano.function(inputs=inputs, outputs=outputs, updates=updates)

    def make_theano_scoring(self, embeddings):
        """
        compile and return symbolic theano function for scoring an ngram, given a list of symbolic vars. Each
        var in the list is a symbolic vector, representing a word embedding
        """
        return theano.function(inputs=embeddings, outputs=self.score_symbolic(self.embedding_layer.flatten_embeddings(embeddings)))

    def _make_functions(self):
        # create symbolic variables for correct and error input
        correct_embeddings = [T.vector(name='correct_embedding%i' % i) for i in range(self.sequence_length)]
        error_embeddings = [T.vector(name='error_embedding%i' % i) for i in range(self.sequence_length)]
        embeddings = [T.vector(name='embedding%i' % i) for i in range(self.sequence_length)]

        self.training_function = self.make_theano_training(correct_embeddings, error_embeddings, self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr)
        self.score_ngram = self.make_theano_scoring(embeddings)

    def train(self, correct_sequence, error_sequence, weighted_learning_rate=0.01):
        correct_embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in correct_sequence]
        error_embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in error_sequence]

        outputs = self.training_function(*(correct_embeddings + error_embeddings + [weighted_learning_rate]))

        correct_grads, error_grads = list(grouper(self.sequence_length, outputs))[:2]

        cost = outputs[-1]

        correct_updates = -weighted_learning_rate * np.array(correct_grads)
        error_updates = -weighted_learning_rate * np.array(error_grads)

        self.embedding_layer.update_embeddings(correct_sequence, correct_updates)
        self.embedding_layer.update_embeddings(error_sequence, error_updates)

        return cost, correct_updates, error_updates

    def score(self, sequence):
        embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in sequence]
        return self.score_ngram(*embeddings)

    def get_embeddings(self):
        return self.embedding_layer.embedding
