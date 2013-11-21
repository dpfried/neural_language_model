import theano
import theano.tensor as T
import numpy as np
from scipy.spatial.distance import cdist
import config

def column(vector):
    """
    turn a theano vector into a broadcastable column
    """
    return T.addbroadcast(T.stack(vector).T, 1)

ADAGRAD_EPSILON = 1e-6 # prevent numerical errors if gradient is 0

class Picklable(object):
    def _nonshared_attrs(self):
        # should be overridden by subclasses to return a list of strings, which
        # will be the names of object attributes that should be pickled
        return []

    def _shared_attrs(self):
        # should be overridden by subclasses to return a list of strings, which
        # will be the names of theano shared variable object attributes that
        # should be pickled
        return []

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
                if name not in kwargs:
                    print 'warning: %s not found, setting to default %s' % (name, default)
                setattr(self, name, theano.shared(kwargs.get(name, default), name=name))
            except TypeError as e: # in case we stored the shared variable, get its current value
                print e
                if name not in kwargs:
                    print 'warning: %s not found, setting to default %s' % (name, default)
                setattr(self, name, theano.shared(kwargs.get(name, default).get_value(), name=name))
        for param in self._nonshared_attrs():
            if type(param) is tuple:
                name, default = param
            else:
                name, default = param, None
            if name not in kwargs:
                print 'warning: %s not found, setting to default %s' % (name, default)
            setattr(self, name, kwargs.get(name, default))

    def __setstate__(self, state):
        self._set_attrs(**state)
        if config.DYNAMIC['compile_on_load']:
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
        raise NotImplementedError('embeddings not implemented')

    def most_similar_to(self, embedding, metric='cosine', top_n=10, **kwargs):
        C = cdist(embedding[np.newaxis,:], self.embeddings, metric, **kwargs)
        sims = C[0]
        return [(i, sims[i]) for i in np.argsort(sims)[:top_n]]

    def most_similar_embeddings(self, index, metric='cosine', top_n=10, **kwargs):
        this_embedding = self.embeddings[index]
        return self.most_similar_to(this_embedding, metric=metric,top_n=top_n, **kwargs)

class EmbeddingLayer(Picklable, VectorEmbeddings):
    def _nonshared_attrs(self):
        return [ 'vocab_size',
                'dimensions',
                'adagrad',
                ]

    def _shared_attrs(self):
        return ['embedding',
                'learning_rate',
                'gradient_norms_sums']

    @property
    def ada_weights(self):
        return {
            'embedding': self.gradient_norms_sums,
        }

    def __init__(self, rng, vocab_size, dimensions, initial_embedding_range=0.01, initial_embeddings=None, adagrad=False, learning_rate=0.01):
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

        gradient_norms_sums = np.ones(vocab_size, dtype=theano.config.floatX) * ADAGRAD_EPSILON

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        self._set_attrs(vocab_size=vocab_size,
                        dimensions=dimensions,
                        embedding=initial_embeddings,
                        gradient_norms_sums=gradient_norms_sums,
                        learning_rate=learning_rate,
                        adagrad=adagrad,
                        )
        self._initialize()

    def __call__(self, index):
        if type(index) is list:
            indices = T.stack(*index)
            return self.embedding[indices]
        else:
            return self.embedding[index]

    def updates(self, cost, index_list, embedding_list):
        # cost: a symbolic cost function
        # index_list: a list of indices into the matrix
        # embedding_list: a list of symbolic embeddings, one for each index in
        # index_list, to be updated with gradient descent
        # gradient descent
        embedding_indices = T.stack(*index_list)
        dembeddings = T.stack(*T.grad(cost, embedding_list))
        gradient_norms = T.sqrt(T.sum(dembeddings ** 2, axis=1))
        if self.adagrad:
            update_weight = self.learning_rate / column(self.gradient_norms_sums[embedding_indices] + gradient_norms)
        else:
            update_weight = self.learning_rate
        return [(self.embedding, T.inc_subtensor(self.embedding[embedding_indices],
                                                 - update_weight * dembeddings)),
                (self.gradient_norms_sums, T.inc_subtensor(self.gradient_norms_sums[embedding_indices],
                                                           gradient_norms))]

    @property
    def embeddings(self):
        return self.embedding.get_value()

class LinearScalarResponse(Picklable):
    def _nonshared_attrs(self):
        return ['n_in', 'adagrad']

    def _shared_attrs(self):
        return ['W', 'b', 'learning_rate', 'dW_sum_squares', 'db_sum_squares']

    @property
    def ada_weights(self):
        return {
            'W': self.dW_sum_squares,
            'b': self.db_sum_squares,
        }

    def __init__(self, n_in, learning_rate=0.01, adagrad=False):
        # init the weights W as a vector of zeros
        W_values = np.zeros((n_in,), dtype=theano.config.floatX)
        dW_ss = np.ones_like(W_values, dtype=theano.config.floatX) * ADAGRAD_EPSILON

        # init the basis as a scalar, 0
        b_values = np.cast[theano.config.floatX](0.0)
        db_ss = np.cast[theano.config.floatX](ADAGRAD_EPSILON)

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        self._set_attrs(W=W_values,
                        b=b_values,
                        n_in=n_in,
                        learning_rate=learning_rate,
                        dW_sum_squares=dW_ss,
                        db_sum_squares=db_ss,
                        adagrad=adagrad)
        self._initialize()

    def __call__(self, x):
        return T.dot(x, self.W) + self.b

    def updates(self, cost):
        dW, db = T.grad(cost, [self.W, self.b])
        dW_ss = self.dW_sum_squares + dW**2
        db_ss = self.db_sum_squares + db**2
        if self.adagrad:
            dW_weight = self.learning_rate / T.sqrt(dW_ss)
            db_weight = self.learning_rate / T.sqrt(db_ss)
        else:
            dW_weight = self.learning_rate
            db_weight = self.learning_rate
        return [
            (self.W, self.W - dW_weight * dW),
            (self.b, self.b - db_weight * db),
            (self.dW_sum_squares, dW_ss),
            (self.db_sum_squares, db_ss),
        ]

class HiddenLayer(Picklable):
    def _nonshared_attrs(self):
        return ['activation', 'n_in', 'n_out', 'adagrad']

    def _shared_attrs(self):
        return ['W', 'b', 'learning_rate', 'dW_sum_squares', 'db_sum_squares']

    @property
    def ada_weights(self):
        return {
            'W': self.dW_sum_squares,
            'b': self.db_sum_squares,
        }

    def __init__(self, rng, n_in, n_out, activation=T.nnet.sigmoid, adagrad=False, learning_rate=0.01):
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
        dW_ss = np.ones_like(W_values, dtype=theano.config.floatX) * ADAGRAD_EPSILON

        if activation == T.nnet.sigmoid:
            W_values *= 4

        if n_out > 1:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            db_ss = np.ones_like(b_values, dtype=theano.config.floatX) * ADAGRAD_EPSILON
        else:
            b_values = np.cast[theano.config.floatX](0.0)
            db_ss = np.cast[theano.config.floatX](ADAGRAD_EPSILON)

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        self._set_attrs(W=W_values,
                        b=b_values,
                        activation=activation,
                        n_in=n_in,
                        n_out=n_out,
                        adagrad=adagrad,
                        learning_rate=learning_rate,
                        dW_sum_squares=dW_ss,
                        db_sum_squares=db_ss)
        self._initialize()

    def __call__(self, x):
        return self.activation(T.dot(x, self.W) + self.b)

    def updates(self, cost):
        dW, db = T.grad(cost, [self.W, self.b])
        dW_ss = self.dW_sum_squares + dW**2
        db_ss = self.db_sum_squares + db**2
        if self.adagrad:
            dW_weight = self.learning_rate / T.sqrt(dW_ss)
            db_weight = self.learning_rate / T.sqrt(db_ss)
        else:
            dW_weight = self.learning_rate
            db_weight = self.learning_rate
        return [
            (self.W, self.W - dW_weight * dW),
            (self.b, self.b - db_weight * db),
            (self.dW_sum_squares, dW_ss),
            (self.db_sum_squares, db_ss),
        ]


class ScaledBilinear(Picklable):
    def _nonshared_attrs(self):
        return [
            'adagrad',
        ]

    def _shared_attrs(self):
        return [
            'learning_rate',
            'w',
            'b',
            'dw_sum_squares',
            'db_sum_squares',
        ]

    @property
    def ada_weights(self):
        return {
            'w': self.dw_sum_squares,
            'b': self.db_sum_squares,
        }

    def __init__(self, adagrad=False, learning_rate=0.01):
        w = np.cast[theano.config.floatX](0.0)
        dw_ss = np.cast[theano.config.floatX](ADAGRAD_EPSILON)
        b = np.cast[theano.config.floatX](0.0)
        db_ss = np.cast[theano.config.floatX](ADAGRAD_EPSILON)

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        self._set_attrs(w=w,
                        b=b,
                        adagrad=adagrad,
                        learning_rate=learning_rate,
                        dw_sum_squares=dw_ss,
                        db_sum_squares=db_ss)
        self._initialize()

    def __call__(self, x, y):
        pass

    def updates(self, cost):
        dw, db = T.grad(cost, [self.w, self.b])
        dw_ss = self.dw_sum_squares + dw**2
        db_ss = self.db_sum_squares + db**2
        if self.adagrad:
            dw_weight = self.learning_rate / T.sqrt(dw_ss)
            db_weight = self.learning_rate / T.sqrt(db_ss)
        else:
            dw_weight = self.learning_rate
            db_weight = self.learning_rate
        return [
            (self.w, self.w - dw_weight * dw),
            (self.b, self.b - db_weight * db),
            (self.dw_sum_squares, dw_ss),
            (self.db_sum_squares, db_ss),
        ]

class CosineSimilarity(ScaledBilinear):
    def __call__(self, x, y):
        cos_distance = T.sum(x * y, -1) / T.sqrt(T.sum(x * x, -1) * T.sum(y * y, -1))
        return cos_distance * self.w + self.b

class EuclideanDistance(ScaledBilinear):
    def __call__(self, x, y):
        res = x - y
        return T.sqrt(T.sum(res ** 2, -1)) * self.w + self.b

class SimilarityNN(Picklable, VectorEmbeddings):
    def _nonshared_attrs(self):
        return ['other_params',
                'mode',
                'embedding_layer',
                'similarity_layer',
                'dimensions',
                'vocab_size']

    def _initialize(self):
        self.train = self._make_training()
        self.score = self._make_scoring()

    @property
    def components(self):
        return ['embedding_layer', 'similarity_layer']

    @property
    def embeddings(self):
        return self.embedding_layer.embeddings

    def __init__(self, rng, vocab_size, dimensions, other_params=None, initial_embeddings=None, mode='FAST_RUN', learning_rate=0.01, adagrad=False, similarity_class=CosineSimilarity):
        # initialize parameters
        if other_params is None:
            other_params = {}

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions,
                                         initial_embeddings=initial_embeddings,
                                         learning_rate=learning_rate,
                                         adagrad=adagrad)

        similarity_layer = similarity_class(learning_rate=learning_rate,
                                      adagrad=adagrad)

        self._set_attrs(other_params=other_params,
                        mode=mode,
                        embedding_layer=embedding_layer,
                        similarity_layer=similarity_layer,
                        dimensions=dimensions,
                        vocab_size=vocab_size)
        self._initialize()


    def __call__(self, index_a, index_b):
        """
        embed the given indices, run the embeddings through the network,
        and return the score (distance) and the list of embeddings
        The list of embeddings is returned for use in differentiation
        """
        embedding_a = self.embedding_layer(index_a)
        embedding_b = self.embedding_layer(index_b)
        return self.similarity_layer(embedding_a, embedding_b), [embedding_a, embedding_b]

    def updates(self, cost, index_list, embedding_list):
        return self.embedding_layer.updates(cost, index_list, embedding_list)\
                + self.similarity_layer.updates(cost)

    def cost(self, index_a, index_b, true_similarity):
        score, embeddings = self(index_a, index_b)
        cost = (score - true_similarity) ** 2
        return cost, [index_a, index_b], embeddings

    def _index_variable(self, name='index'):
        return T.lscalar(name)

    def _make_training(self, cost_addition=None):
        index_a = self._index_variable('index_a')
        index_b = self._index_variable('index_b')
        similarity = T.scalar('true_similarity')

        cost, indices, embeddings = self.cost(index_a, index_b, similarity)
        if cost_addition:
            augmented_cost = cost + cost_addition(indices, embeddings)
        else:
            augmented_cost = cost
        updates = self.updates(augmented_cost,
                               indices,
                               embeddings)

        return theano.function(inputs=[index_a, index_b, similarity], outputs=[augmented_cost, cost], updates=updates, mode=self.mode)

    def _make_scoring(self):
        index_a = self._index_variable('index_a')
        index_b = self._index_variable('index_b')
        score, [embedding_a, embedding_b] = self(index_a, index_b)
        return theano.function(inputs=[index_a, index_b], outputs=score, mode=self.mode)

class SequenceScoringNN(Picklable, VectorEmbeddings):
    def _nonshared_attrs(self):
        return ['n_hidden',
                'other_params',
                'blocks_trained',
                'mode',
                'embedding_layer',
                'hidden_layer',
                'output_layer',
                'dimensions',
                'sequence_length',
                'vocab_size']

    @property
    def components(self):
        return ['embedding_layer', 'hidden_layer', 'output_layer']

    def _initialize(self):
        self.train = self._make_training()
        self.score = self._make_scoring()

    @property
    def embeddings(self):
        return self.embedding_layer.embeddings

    def __init__(self, rng, vocab_size,  dimensions,  sequence_length, n_hidden, other_params=None, initial_embeddings=None, mode='FAST_RUN', learning_rate=0.01, adagrad=False):
        # initialize parameters
        if other_params is None:
            other_params = {}
        blocks_trained = 0

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions,
                                         initial_embeddings=initial_embeddings,
                                         learning_rate=learning_rate,
                                         adagrad=adagrad)

        hidden_layer = HiddenLayer(rng=rng,
                                   n_in=dimensions * sequence_length,
                                   n_out=n_hidden,
                                   activation=T.nnet.sigmoid,
                                   learning_rate=learning_rate,
                                   adagrad=adagrad)

        output_layer = LinearScalarResponse(n_in=n_hidden,
                                            adagrad=adagrad,
                                            learning_rate=learning_rate)

        self._set_attrs(n_hidden=n_hidden,
                         other_params=other_params,
                         blocks_trained=blocks_trained,
                         mode=mode,
                         embedding_layer=embedding_layer,
                         hidden_layer=hidden_layer,
                         output_layer=output_layer,
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

    def updates(self, cost, index_list, embedding_list):
        return self.embedding_layer.updates(cost, index_list, embedding_list)\
                + self.hidden_layer.updates(cost)\
                + self.output_layer.updates(cost)

    def cost(self, correct_index_list, error_index_list):
        correct_score, correct_embeddings = self(correct_index_list)
        error_score, error_embeddings = self(error_index_list)
        cost = T.clip(1 - correct_score + error_score, 0, np.inf)
        return cost, correct_index_list + error_index_list, correct_embeddings + error_embeddings

    def _index_variables(self, basename='index'):
        return T.lscalars(*['%s_%d' % (basename, i)
                            for i in xrange(self.sequence_length)])

    def _make_training(self, cost_addition=None):
        """
        compile and return symbolic theano function for training (including update of embedding and weights),
        The compiled function will take two vectors of ints, each of length
        sequence_length, which are the indices of the words in the good and bad
        ngrams
        """
        correct_indices = self._index_variables('correct')
        error_indices = self._index_variables('error')

        cost, indices, embeddings = self.cost(correct_indices, error_indices)
        if cost_addition:
            augmented_cost = cost + cost_addition(indices, embeddings)
        else:
            augmented_cost = cost
        updates = self.updates(augmented_cost,
                               indices,
                               embeddings)

        inputs = correct_indices + error_indices
        outputs = [augmented_cost, cost]
        return theano.function(inputs=inputs, outputs=outputs, updates=updates, mode=self.mode)

    def _make_scoring(self):
        indices = self._index_variables('index')
        score, embeddings = self(indices)
        return theano.function(inputs=indices, outputs=score, mode=self.mode)

class TranslationalNN(Picklable, VectorEmbeddings):
    def _nonshared_attrs(self):
        return ['other_params',
                'mode',
                'embedding_layer',
                'translation_layer',
                'n_rel',
                'dimensions',
                'vocab_size']

    @property
    def components(self):
        return ['embedding_layer', 'translation_layer']

    def _initialize(self):
        self.train = self._make_training()
        self.score = self._make_scoring()

    @property
    def embeddings(self):
        return self.embedding_layer.embeddings

    def __init__(self, rng, vocab_size, n_rel, dimensions, other_params=None, initial_embeddings=None, mode='FAST_RUN', learning_rate=0.01, adagrad=False):
        # initialize parameters
        if other_params is None:
            other_params = {}

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions,
                                         initial_embeddings=initial_embeddings,
                                         learning_rate=learning_rate,
                                         adagrad=adagrad)

        # represent each of n_rel relationships as a vector embedding to be
        # added to the embedding of the left entity in the relationship
        translation_layer = EmbeddingLayer(rng,
                                           vocab_size=n_rel,
                                           dimensions=dimensions,
                                           learning_rate=learning_rate,
                                           adagrad=adagrad)

        self._set_attrs(other_params=other_params,
                        mode=mode,
                        embedding_layer=embedding_layer,
                        translation_layer=translation_layer,
                        dimensions=dimensions,
                        n_rel=n_rel,
                        vocab_size=vocab_size)
        self._initialize()


    def __call__(self, left_entity_index, right_entity_index, relationship_index):
        """
        score the given entity with the given relationship, and return the lists
        of embeddings for use in differentiation
        """
        left_embedding = self.embedding_layer(left_entity_index)
        right_embedding = self.embedding_layer(right_entity_index)
        rel_embedding = self.translation_layer(relationship_index)

        composite = left_embedding + rel_embedding - right_embedding
        score = -1 * T.sqrt(T.sum(composite **2, -1))

        return score, [left_embedding, right_embedding], [rel_embedding]

    def updates(self, cost, entity_index_list, entity_embedding_list, relationship_index_list, relationship_embedding_list):
        return self.embedding_layer.updates(cost, entity_index_list, entity_embedding_list)\
                + self.translation_layer.updates(cost, relationship_index_list, relationship_embedding_list)

    def cost(self,
             left_index_good, right_index_good, rel_index_good,
             left_index_bad, right_index_bad, rel_index_bad):

        good_entity_indices = [left_index_good, right_index_good]
        bad_entity_indices = [left_index_bad, right_index_bad]

        good_score, good_entity_embeddings, good_rel_embeddings = self(left_index_good, right_index_good, rel_index_good)
        bad_score, bad_entity_embeddings, bad_rel_embeddings = self(left_index_bad, right_index_bad, rel_index_bad)

        cost = T.clip(1 - good_score + bad_score, 0, np.inf)

        entity_indices = good_entity_indices + bad_entity_indices
        rel_indices = [rel_index_good, rel_index_bad]

        entity_embeddings = good_entity_embeddings + bad_entity_embeddings
        rel_embeddings = good_rel_embeddings + bad_rel_embeddings

        return cost, entity_indices, entity_embeddings, rel_indices, rel_embeddings


    def _index_variable(self, name='index'):
        return T.lscalar(name)

    def _make_training(self, cost_addition=None):
        left_good, left_bad, right_good, right_bad, rel_good, rel_bad = map(self._index_variable, ['left_good', 'left_bad', 'right_good', 'right_bad', 'rel_good', 'rel_bad'])

        cost, entity_indices, entity_embeddings, rel_indices, rel_embeddings = self.cost(left_good, right_good, rel_good,
                                                                                         left_bad, right_bad, rel_bad)

        if cost_addition:
            augmented_cost = cost + cost_addition(entity_indices, entity_embeddings)
        else:
            augmented_cost = cost
        updates = self.updates(cost,
                               entity_indices, entity_embeddings,
                               rel_indices, rel_embeddings)

        return theano.function(inputs=[left_good, right_good, rel_good, left_bad, right_bad, rel_bad],
                               outputs=[augmented_cost, cost],
                               updates=updates,
                               mode=self.mode)

    def _make_scoring(self):
        left, right, rel = map(self._index_variable, ['left_index', 'right_index', 'rel_index'])
        score, _, _ = self(left, right, rel)
        return theano.function(inputs=[left, right, rel], outputs=score, mode=self.mode)

class NeuralTensorLayer(Picklable):
    def _shared_attrs(self):
        return ['W', 'V', 'b', 'learning_rate']

    def _nonshared_attrs(self):
        return ['n_rel', 'n_in', 'n_out', 'activation']

    @property
    def ada_weights(self):
        return {
            'W': self.dW_sum_squares,
            'V': self.dV_sum_squares,
            'b': self.db_sum_squares,
        }

    # n_rel: number of relations (number of 3d tensors)
    # n_in: input dimensionality
    # n_out: output dimensionality
    # W: a 4d tensor containing a set of 3d relationship tensors. Dimensions: (n_rel, n_out, n_in, n_in)
    # V: a 3d tensor containing a set of 2d hidden layer matrics. Dimensions: (n_rel, n_out, 2*n_in)

    def __init__(self, rng, n_rel, n_in, n_out, activation=T.tanh, adagrad=False, learning_rate=0.01):
        if adagrad:
            print 'warning: adagrad unimplemented for TensorLayer'

        W_size = (n_rel, n_out, n_in, n_in)
        # what to change this heuristic to?
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=W_size), dtype=theano.config.floatX)

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        V_size = (n_rel, n_out, 2*n_in)
        V_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=V_size), dtype=theano.config.floatX)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        b_values = np.zeros((n_rel,n_out), dtype=theano.config.floatX)

        self._set_attrs(W=W_values,
                        V=V_values,
                        b=b_values,
                        n_rel=n_rel,
                        n_in=n_in,
                        n_out=n_out,
                        activation=activation,
                        learning_rate=learning_rate)

    def __call__(self, x1, x2, relation_index):
        """
        x1: an embedding
        x2: an embedding
        relation_index: the index of the relation to use
        returns the output, the W_embedding, the V_embedding, and the b embedding
        """
        # get the relationship embeddings
        W_rel, V_rel, b_rel = self.W[relation_index], self.V[relation_index], self.b[relation_index]
        return T.dot(T.dot(x1, W_rel), x2) + T.dot(V_rel, T.concatenate([x1, x2])) + b_rel, [W_rel], [V_rel], [b_rel]

    def updates(self, cost, relation_index_list, W_rel_list, V_rel_list, b_rel_list):
        """
        cost: a symbolic cost variable to be differentiated
        relation_index_list: the list of relation indices corresponding to
            the embeddings in W_rel_list and V_rel_list, used to choose correct
            tensor W and matrix V
        W_rel_list: the list of subtensors of W corresponding to indices in relationship_index_list
        V_rel_list: the list of subtensors of V corresponding to indices in relationship_index_list
        b_rel_list: the list of rows of b corresponding to indices in relationship_index_list
        """
        indices = T.stack(*relation_index_list)
        dW = T.stack(*T.grad(cost, W_rel_list))
        dV = T.stack(*T.grad(cost, V_rel_list))
        db = T.stack(*T.grad(cost, b_rel_list))
        return [(self.b, T.inc_subtensor(self.b[indices], - self.learning_rate * db)),
                (self.W, T.inc_subtensor(self.W[indices], - self.learning_rate * dW)),
                (self.V, T.inc_subtensor(self.V[indices], - self.learning_rate * dV))]


class TensorNN(Picklable, VectorEmbeddings):
    def _nonshared_attrs(self):
        return ['n_hidden',
                'n_rel',
                'dimensions',
                'embedding_layer',
                'tensor_layer',
                'output_layer',
                'mode',
                'vocab_size',
                'other_params']

    def _initialize(self):
        self.train = self._make_training()
        self.score = self._make_scoring()

    def __init__(self, rng, vocab_size, n_rel, dimensions, n_hidden=None, other_params=None, learning_rate=0.01, mode='FAST_RUN', initial_embeddings=None, adagrad=False):
        if other_params is None:
            other_params = {}

        learning_rate = np.cast[theano.config.floatX](learning_rate)

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions,
                                         learning_rate=learning_rate,
                                         adagrad=adagrad,
                                         initial_embeddings=initial_embeddings)

        tensor_layer = NeuralTensorLayer(rng, n_rel, dimensions, n_hidden, learning_rate=learning_rate, adagrad=adagrad)

        output_layer = LinearScalarResponse(n_hidden, learning_rate=learning_rate, adagrad=adagrad)

        # store attributes
        self._set_attrs(n_rel=n_rel,
                        dimensions=dimensions,
                        n_hidden=n_hidden,
                        mode=mode,
                        vocab_size=vocab_size,
                        other_params=other_params,
                        embedding_layer=embedding_layer,
                        tensor_layer=tensor_layer,
                        output_layer=output_layer)

    @property
    def embeddings(self):
        return self.embedding_layer.embeddings

    def __call__(self, left_entity_index, right_entity_index, relationship_index):
        """
        score the given entities with the given relationship, and return the score, list of entity embeddings,
        and lists of W and V embeddings for the relationship
        """
        left_embedding = self.embedding_layer(left_entity_index)
        right_embedding = self.embedding_layer(right_entity_index)
        tensor_layer_output, W_embeddings, V_embeddings, b_embeddings = self.tensor_layer(left_embedding, right_embedding, relationship_index)
        score = self.output_layer(tensor_layer_output)
        return score, [left_embedding, right_embedding], W_embeddings, V_embeddings, b_embeddings

    def updates(self, cost, entity_index_list, entity_embedding_list, relationship_index_list, W_list, V_list, b_list):
        return self.embedding_layer.updates(cost, entity_index_list, entity_embedding_list)\
                + self.tensor_layer.updates(cost, relationship_index_list, W_list, V_list, b_list)\
                + self.output_layer.updates(cost)

    def cost(self,
             left_index_good, right_index_good, rel_index_good,
             left_index_bad, right_index_bad, rel_index_bad):
        good_entity_indices = [left_index_good, right_index_good]
        bad_entity_indices = [left_index_bad, right_index_bad]

        good_score, good_entity_embeddings, good_W, good_V, good_b = self(left_index_good, right_index_good, rel_index_good)
        bad_score, bad_entity_embeddings, bad_W, bad_V, bad_b = self(left_index_bad, right_index_bad, rel_index_bad)

        cost = T.clip(1 - good_score + bad_score, 0, np.inf)

        entity_indices = good_entity_indices + bad_entity_indices
        rel_indices = [rel_index_good, rel_index_bad]

        entity_embeddings = good_entity_embeddings + bad_entity_embeddings
        W_embeddings = good_W + bad_W
        V_embeddings = good_V + bad_V
        b_embeddings = good_b + bad_b

        return cost, entity_indices, entity_embeddings, rel_indices, W_embeddings, V_embeddings, b_embeddings

    def _index_variable(self, name='index'):
        return T.lscalar(name)

    def _make_training(self, cost_addition=None):
        left_good, left_bad, right_good, right_bad, rel_good, rel_bad = map(self._index_variable, ['left_good', 'left_bad', 'right_good', 'right_bad', 'rel_good', 'rel_bad'])

        cost_return = self.cost(left_good, right_good, rel_good,
                                left_bad, right_bad, rel_bad)

        cost, entity_indices, entity_embeddings, rel_indices, W_embeddings, V_embeddings, b_embeddings = cost_return

        if cost_addition:
            augmented_cost = cost + cost_addition(entity_indices, entity_embeddings)
        else:
            augmented_cost = cost

        updates = self.updates(augmented_cost, entity_indices, entity_embeddings, rel_indices, W_embeddings, V_embeddings, b_embeddings)

        return theano.function(inputs=[left_good, right_good, rel_good, left_bad, right_bad, rel_bad],
                               outputs=[augmented_cost, cost],
                               updates=updates,
                               mode=self.mode)

    def _make_scoring(self):
        left, right, rel = map(self._index_variable, ['left_index', 'right_index', 'rel_index'])
        score, _, _, _, _ = self(left, right, rel)
        return theano.function(inputs=[left, right, rel], outputs=score, mode=self.mode)


class ADMM(Picklable):
    def _admm_cost(self, side='w'):
        if side == 'w':
            other_model = self.v_trainer
        elif side == 'v':
            other_model = self.w_trainer
        else:
            raise ValueError('bad side value %s' % side)
        def admm_cost(indices, this_model_embeddings):
            other_model_embeddings = other_model.embedding_layer(indices)
            index_vector = T.stack(*indices)

            indicator = column(self.intersection_indicator[index_vector])

            y = self.y[index_vector] * indicator

            if side == 'w':
                res = (this_model_embeddings - other_model_embeddings) * indicator
            else:
                res = (other_model_embeddings - this_model_embeddings) * indicator

            return T.sum(y * res) + self.rho / 2.0 * T.sum(res * res)
        return admm_cost

    def _shared_attrs(self):
        return ['y', 'intersection_indicator']

    def _nonshared_attrs(self):
        return ['w_trainer',
                'v_trainer',
                'indices_in_intersection',
                'other_params',
                'rho',
                'mode',
                'k']

    @property
    def components(self):
        return ['w_trainer', 'v_trainer']

    def __init__(self, w_trainer, v_trainer, vocab_size, indices_in_intersection, dimensions, rho, other_params=None, mode='FAST_RUN'):
        # the lagrangian
        y = np.zeros((vocab_size, dimensions), dtype=theano.config.floatX)

        intersection_indicator = np.zeros(vocab_size, dtype=np.int8)
        intersection_indicator[list(indices_in_intersection)] = 1

        if not other_params:
            other_params = {}

        self._set_attrs(w_trainer=w_trainer,
                        v_trainer=v_trainer,
                        rho=rho,
                        other_params=other_params,
                        intersection_indicator=intersection_indicator,
                        indices_in_intersection=list(indices_in_intersection),
                        mode=mode,
                        y=y,
                        k=0)
        self._initialize()

    def _initialize(self):
        self.update_w = self.w_trainer._make_training(cost_addition=self._admm_cost('w'))
        self.update_v = self.v_trainer._make_training(cost_addition=self._admm_cost('v'))
        self.update_y = self._make_y_update()

    def _make_y_update(self):
        w = self.w_trainer.embedding_layer.embedding
        v = self.v_trainer.embedding_layer.embedding
        if self.indices_in_intersection:
            res = (w - v)[self.indices_in_intersection]
            y = self.y[self.indices_in_intersection]
            updates = [(self.y, T.inc_subtensor(y, self.rho * res))]
        else:
            res = w - v
            y = self.y
            updates = [(self.y, self.y + self.rho * res)]

        res_norm_mean = T.mean(T.sqrt(T.sum(res**2, 1)))
        y_norm_mean = T.mean(T.sqrt(T.sum(y**2, 1)))

        return theano.function(inputs=[],
                               outputs=[res_norm_mean, y_norm_mean],
                               updates=updates,
                               mode=self.mode)

    @property
    def embeddings(self):
        include_syntactic = not ('dont_run_syntactic' in self.other_params and self.other_params['dont_run_syntactic'])
        include_semantic = not ('dont_run_semantic' in self.other_params and self.other_params['dont_run_semantic'])
        if include_syntactic and include_semantic:
            return np.concatenate((self.w_trainer.embeddings, self.v_trainer.embeddings), 1)
        elif include_syntactic:
            return self.w_trainer.embeddings
        else:
            return self.v_trainer.embeddings

    def increase_k(self):
        self.k += 1
