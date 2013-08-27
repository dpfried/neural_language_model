import theano
import theano.tensor as T
import numpy as np
import scipy

# code adapted from DeepLearning tutorial:
# deeplearning.net/tutorial/mlp.html

class EmbeddingLayer(object):
    def __init__(self, rng, input, vocab_size, dimensions, sequence_length=5, initial_embedding_range=0.01):
        """ Initialize the parameters of the embedding layer

        :type rng: nympy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: a theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, dimensions * sequence_length) (instantiated
        values can be produced by one_hot_from_batch)

        :type vocab_size: int
        :param vocab_size: the number of discrete items to be embedded
        in the distributed representation

        :type dimensions: int
        :param dimensions: the number of dimensions in the distributed
        representation.

        :type sequence_length: int
        :param sequence_length: the number of words in each n-gram
        """

        self.rng = rng
        self.vocab_size = vocab_size
        self.dimensions = dimensions

        initial_embedding = np.asarray(rng.uniform(
            low=-initial_embedding_range / 2.,
            high=initial_embedding_range / 2.,
            size=(self.vocab_size, self.dimensions)),
            dtype=theano.config.floatX)

        self.embedding = theano.shared(value=initial_embedding, name='embedding')
        self.sequence_length = sequence_length

        # params are those that should be updated in gradient descent and
        # are also serialized
        self.params = [self.embedding]

        # hyperparams are those that we'd serialize but aren't optimized in
        # gradient descent
        self.hyper_params = [self.vocab_size, self.dimensions, self.sequence_length]

        # memoize to use in one_hot_from_batch for lookup of one-hot encodings
        self.vocab_id = scipy.sparse.eye(vocab_size, vocab_size).tocsr()

        # symbolizes a matrix (seq length x vocabulary size) that is the one-hot encoding
        # of the input sequence
        self.one_hot_input = T.matrix(name='one_hot_input')

        # output: a batch_size x (dimension * sequence_length) matrix
        # each row corresponds to the concatenated vectors of the
        # representations of the words in that n-gram (where n =
        # sequence_length)
        self.output = T.flatten(T.dot(self.one_hot_input, self.embedding), outdim=2)

    def one_hot_from_batch(self, batch_of_symbol_sequences):
        return np.array(self.one_hot_from_symbols(sequence) for sequence in batch_of_symbol_sequences)

    def one_hot_from_symbols(self, symbol_indices):
        """returns a matrix of dimensions sequence_length x vocabulary_size that
        has a 1 in r,c if word_r is symbol_c, and 0 otherwise"""
        return self.vocab_id[symbol_indices].todense()

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (e.g., one minibatch of input images)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoint lies

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the target lies
        """

        # init the weights W as a matrix of zeros (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                              dtype=theano.config.floatX), name='W')

        # init the basis as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                              dtype=theano.config.floatX), name='b')

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as a class whose probability is maximal in the
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.hyper_params = [self.n_in, self.n_out]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
        """
        Typical hidden layer of a MLP: fully connected units with sigmoidal
        activation function. Weight matrix has shape (n_in, n_out)
        and bias vector b has shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

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
        self.input = input

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

        self.output = activation(T.dot(input, self.W) + self.b)

        # parameters of the model
        self.params = [self.W, self.b]

class NLM(object):
    """
    """

    def __init__(self, rng, vocab_size, dimensions, sequence_length, n_hidden, n_out):
        self.rng = rng
        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.embedding_layer = EmbeddingLayer(rng,
                                              vocab_size=vocab_size,
                                              dimensions=dimensions,
                                              sequence_length=sequence_length)

        self.hidden_layer = HiddenLayer(rng=rng,
                                        input=self.embedding_layer.output,
                                        n_in=dimensions * sequence_length,
                                        n_out=n_hidden,
                                        activation=T.tanh)

        self.log_regression_layer = LogisticRegression(input=self.hidden_layer.output,
                                                       n_in=n_hidden,
                                                       n_out=n_out)

        # L1 norm
        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.log_regression_layer.W).sum()

        # square of L2 norm
        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.log_regression_layer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative log
        # likelihood of the output of the model, computed in the logistic
        # regression (output) layer
        self.negative_log_likelihood = self.log_regression_layer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.log_regression_layer.errors

        # the params of the model are the parameters of the two layers it is
        # made of
        self.params = self.embedding_layer.params + self.hidden_layer.params + self.log_regression_layer.params

        self.one_hot_from_batch = self.embedding_layer.one_hot_from_batch

        self.one_hot_input = self.embedding_layer.one_hot_input