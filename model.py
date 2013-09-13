import theano
import theano.tensor as T
import numpy as np
from utils import grouper
from nltk.corpus import wordnet as wn
from collections import defaultdict
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
        try:
            return self.embedding[symbol_indices]
        except IndexError as e:
            print symbol_indices
            raise e

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
    def __init__(self, rng, vocabulary, synsets, dimensions, synset_dimensions, sequence_length, n_hidden, L1_reg, L2_reg, other_params=None):
        # initialize parameters
        if other_params is None:
            other_params = {}
        self.rng = rng
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.dimensions = dimensions
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.other_params = other_params
        self.blocks_trained = 0

        self.synsets = synsets
        self.synsets_size = len(synsets)
        self.synset_dimensions = synset_dimensions

        self._build_layers()
        self._build_functions()

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
    def synset_to_symbol(self):
        try:
            return self._synset_to_symbol
        except AttributeError:
            self._synset_to_symbol = defaultdict(int, dict((synset, index)
                                                           for index, synset in enumerate(self.synsets, 1)))
            self._synset_to_symbol['NONE'] = 0
            return self._synset_to_symbol

    @property
    def word_to_symbol(self):
        try:
            return self._word_to_symbol
        except AttributeError:
            self._word_to_symbol = defaultdict(int, dict((word, index)
                                                         for index, word in enumerate(self._get_vocabulary())))
            return self._word_to_symbol

    @property
    def symbol_to_synset(self):
        try:
            return self._symbol_to_synset
        except AttributeError:
            self._symbol_to_synset = dict(enumerate(self.synsets, 1))
            return self._symbol_to_synset

    @property
    def symbol_to_word(self):
        try:
            return self._symbol_to_word
        except AttributeError:
            self._symbol_to_word = defaultdict(lambda : '*UNKNOWN*', dict(enumerate(self._get_vocabulary())))
            self._symbol_to_word[0] = '*UNKNOWN*'
            return self._symbol_to_word

    def _build_layers(self):
        self.embedding_layer = EmbeddingLayer(self.rng, vocab_size=self.vocab_size,
                                              dimensions=self.dimensions,
                                              sequence_length=self.sequence_length)

        self.synset_embedding_layer = EmbeddingLayer(self.rng, vocab_size=self.synsets_size,
                                                      dimensions=self.synset_dimensions,
                                                      sequence_length=self.sequence_length)

        self.hidden_layer = HiddenLayer(rng=self.rng,
                                        n_in=(self.dimensions + self.synset_dimensions) * self.sequence_length,
                                        n_out=self.n_hidden,
                                        activation=T.nnet.sigmoid)

        self.output_layer = LinearScalarResponse(n_in=self.n_hidden)

        self.layer_stack = [self.hidden_layer, self.output_layer]

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.output_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.output_layer.W ** 2).sum()

    def score_symbolic(self, sequence_embedding):
        return reduce(lambda layer_input, layer: layer.apply(layer_input), self.layer_stack, sequence_embedding)

    def compare_symbolic(self, correct_sequence_embedding, error_sequence_embedding):
        return T.clip(1 - self.score_symbolic(correct_sequence_embedding) + self.score_symbolic(error_sequence_embedding), 0, np.inf)

    # def compare_symbolic(self, correct_sequence_embedding, error_sequence_embedding, logistic_scaling_factor=1.0):
    #     score_difference = self.score_symbolic(correct_sequence_embedding) - self.score_symbolic(error_sequence_embedding)
    #     return T.log(1 + T.exp(logistic_scaling_factor * -1 * score_difference))

    def _build_functions(self):
        # create symbolic variables for correct and error input
        correct_embeddings = [T.vector(name='correct_embedding%i' % i) for i in range(self.sequence_length)]
        error_embeddings = [T.vector(name='error_embedding%i' % i) for i in range(self.sequence_length)]

        correct_synset_embeddings = [T.vector(name='correct_synset_embedding%i' % i) for i in range(self.sequence_length)]
        error_synset_embeddings = [T.vector(name='error_synset_embedding%i' % i) for i in range(self.sequence_length)]

        correct_sequence_embedding = self.embedding_layer.flatten_embeddings(correct_embeddings + correct_synset_embeddings)
        error_sequence_embedding = self.embedding_layer.flatten_embeddings(error_embeddings + error_synset_embeddings)

        cost = self.compare_symbolic(correct_sequence_embedding, error_sequence_embedding) + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        weighted_learning_rate = T.scalar(name='weighted_learning_rate')

        # update the params of the model using the gradients
        updates = [(param, param - weighted_learning_rate * T.grad(cost, param))
                   for param in self.hidden_layer.params + self.output_layer.params]

        dcorrect_embeddings = T.grad(cost, correct_embeddings)
        derror_embeddings = T.grad(cost, error_embeddings)

        dcorrect_synset_embeddings= T.grad(cost, correct_synset_embeddings)
        derror_synset_embeddings = T.grad(cost, error_synset_embeddings)

        inputs = correct_embeddings + error_embeddings + correct_synset_embeddings + error_synset_embeddings + [weighted_learning_rate]
        outputs = dcorrect_embeddings + derror_embeddings + dcorrect_synset_embeddings + derror_synset_embeddings + [cost]

        self.training_function = theano.function(inputs=inputs,
                                                 outputs=outputs,
                                                 updates=updates)

        embeddings = [T.vector(name='embedding%i' % i) for i in range(self.sequence_length)]
        synset_embeddings = [T.vector(name='synset_embedding%i' % i) for i in range(self.sequence_length)]

        self.score_ngram = theano.function(inputs=embeddings + synset_embeddings,
                                           outputs=self.score_symbolic(self.embedding_layer.flatten_embeddings(embeddings + synset_embeddings)))

    def train(self, correct_sequence, error_sequence, correct_synset_sequence, error_synset_sequence, weighted_learning_rate=0.01):
        correct_embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in correct_sequence]
        error_embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in error_sequence]

        correct_synset_embeddings = [self.synset_embedding_layer.embeddings_from_symbols(i) for i in correct_synset_sequence]
        error_synset_embeddings = [self.synset_embedding_layer.embeddings_from_symbols(i) for i in error_synset_sequence]

        outputs = self.training_function(*(correct_embeddings + error_embeddings + correct_synset_embeddings + error_synset_embeddings + [weighted_learning_rate]))

        correct_grads, error_grads, correct_synset_grads, error_synset_grads = list(grouper(self.sequence_length, outputs))[:4]

        cost = outputs[-1]

        correct_updates = -weighted_learning_rate * np.array(correct_grads)
        error_updates = -weighted_learning_rate * np.array(error_grads)

        correct_synset_updates = -weighted_learning_rate * np.array(correct_synset_grads)
        error_synset_updates = -weighted_learning_rate * np.array(error_synset_grads)

        self.embedding_layer.update_embeddings(correct_sequence, correct_updates)
        self.embedding_layer.update_embeddings(error_sequence, error_updates)
        self.synset_embedding_layer.update_embeddings(correct_synset_sequence, correct_synset_updates)
        self.synset_embedding_layer.update_embeddings(error_synset_sequence, error_synset_updates)

        return cost, correct_updates, error_updates, correct_synset_updates, error_synset_updates

    def score(self, sequence, synset_sequence):
        embeddings = [self.embedding_layer.embeddings_from_symbols(i) for i in sequence]
        synset_embeddings = [self.synset_embedding_layer.embeddings_from_symbols(i) for i in sequence]
        return self.score_ngram(*(embeddings + synset_embeddings))

    def dump_embeddings(self, filename, index_to_word, normalize=True, precision=8):
        format_str = '%%0.%if' % precision
        float_to_str = lambda f: format_str % f
        print normalize
        with open(filename, 'w') as f:
            for index, embedding in enumerate(self.embedding_layer.embedding):
                # skip RARE
                if index == 0:
                    continue
                if normalize:
                    vector = embedding / np.sqrt(np.dot(embedding, embedding))
                else:
                    vector = embedding
                vector_string_rep = ' '.join(map(float_to_str, vector))
                f.write('%s %s\n' % (index_to_word[index], vector_string_rep))

    def get_embedding(self, word, include_synsets=None, normalize_components=False):
        """include_synsets: None, 'all' or 'top'"""
        if word not in self.word_to_symbol:
            print 'warning: %s not in vocab' % word
        word_embedding = self.embedding_layer.embedding[self.word_to_symbol[word]]
        if include_synsets is None:
            components = [word_embedding]
        else:
            word_synsets = wn.synsets(word)
            if not word_synsets:
                indices = [0]
            elif include_synsets == 'all':
                indices = [self.synset_to_symbol[synset] for synset in word_synsets]
            elif include_synsets == 'top':
                indices = [self.synset_to_symbol[word_synsets[0]]]
            synset_embedding = self.synset_embedding_layer.embedding[indices].mean(0)
            components = [word_embedding, synset_embedding]
        if normalize_components:
            components = [component / np.linalg.norm(component, 2)
                          for component in components]
        return np.concatenate(components)

    def get_synset_embedding(self, synset, normalize=False):
        if synset not in self.synset_to_symbol:
            print 'warning: %s not in known synsets' % synset
        synset_embedding = self.synset_embedding_layer.embedding[self.synset_to_symbol[synset]]
        if normalize:
            return synset_embedding / np.linalg.norm(synset_embedding, 2)
        else:
            return synset_embedding
