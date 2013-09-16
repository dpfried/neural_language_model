from ngrams import NgramReader
import theano
import theano.tensor as T
import numpy as np
import gzip, cPickle
import semantic_module
import sys

from model import EmbeddingLayer, HiddenLayer, EmbeddingTrainer

class SemanticDistance(EmbeddingTrainer):
    def __init__(self, rng, vocabulary, dimensions):
        super(SemanticDistance, self).__init__(rng, vocabulary, dimensions)
        self.embedding_layer = EmbeddingLayer(self.rng, vocab_size=self.vocab_size,
                                              dimensions=self.dimensions,
                                              sequence_length=1)
        self._build_functions()

    def similarity_symbolic(self, w1, w2):
        return T.dot(w1, w2) / T.sqrt(T.dot(w1, w1) * T.dot(w2, w2))

    def loss(self, w1, w2, actual_similarity):
        return (self.similarity_symbolic(w1, w2) - actual_similarity) ** 2

    # def compare_symbolic(self, correct_sequence_embedding, error_sequence_embedding, logistic_scaling_factor=1.0):
    #     score_difference = self.score_symbolic(correct_sequence_embedding) - self.score_symbolic(error_sequence_embedding)
    #     return T.log(1 + T.exp(logistic_scaling_factor * -1 * score_difference))

    def _build_functions(self):
        # create symbolic variables for correct and error input
        w1_embedding = T.vector(name='w1_embedding')
        w2_embedding = T.vector(name='w2_embedding')
        embeddings = [w1_embedding, w2_embedding]
        training_similarity = T.scalar(name='similarity')

        cost = self.loss(w1_embedding, w2_embedding, training_similarity)

        weighted_learning_rate = T.scalar(name='weighted_learning_rate')

        dembeddings = T.grad(cost, embeddings)

        inputs = embeddings + [training_similarity] + [weighted_learning_rate]
        outputs = dembeddings + [cost]

        self.training_function = theano.function(inputs=inputs,
                                                 outputs=outputs)

        self.similarity = theano.function(inputs=embeddings,
                                           outputs=self.similarity_symbolic(w1_embedding, w2_embedding))

        # no params to update (embeddings are handled separately)
        self.params = []

    def train(self, w1_index, w2_index, actual_similarity, weighted_learning_rate=0.01):
        w1_embedding = self.embedding_layer.embeddings_from_symbols(w1_index)
        w2_embedding = self.embedding_layer.embeddings_from_symbols(w2_index)

        dw1, dw2, cost = self.training_function(w1_embedding,
                                                w2_embedding,
                                                actual_similarity,
                                                weighted_learning_rate)

        w1_update = -weighted_learning_rate * dw1
        w2_update = -weighted_learning_rate * dw2

        self.embedding_layer.update_embeddings([w1_index, w2_index], [w1_update, w2_update])

        return cost, w1_update, w2_update

    def score(self, w1_index, w2_index):
        w1_embedding = self.embedding_layer.embeddings_from_symbols(w1_index)
        w2_embedding = self.embedding_layer.embeddings_from_symbols(w2_index)
        return self.similarity(w1_embedding, w2_embedding)

    def get_embeddings(self):
        return self.embedding_layer.embedding

class SemanticNet(EmbeddingTrainer):
    def __init__(self, rng, vocabulary, dimensions, n_hidden, L1_reg, L2_reg, other_params=None):
        super(SemanticNet, self).__init__(rng, vocabulary, dimensions)
        # initialize parameters
        if other_params is None:
            other_params = {}
        self.n_hidden = n_hidden
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.other_params = other_params
        self.blocks_trained = 0

        self._build_layers()
        self._build_functions()

    def _build_layers(self):
        self.embedding_layer = EmbeddingLayer(self.rng, vocab_size=self.vocab_size,
                                              dimensions=self.dimensions,
                                              sequence_length=2)

        self.hidden_layer = HiddenLayer(rng=self.rng,
                                        n_in=self.dimensions * 2,
                                        n_out=self.n_hidden,
                                        activation=T.tanh)

        self.output_layer = HiddenLayer(rng=self.rng,
                                        n_in=self.n_hidden,
                                        n_out=1,
                                        activation=T.tanh)

        self.layer_stack = [self.hidden_layer, self.output_layer]
        self.params = self.hidden_layer.params + self.output_layer.params

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.output_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.output_layer.W ** 2).sum()

    def similarity_symbolic(self, smashed_words_embedding):
        return reduce(lambda layer_input, layer: layer.apply(layer_input), self.layer_stack, smashed_words_embedding)

    def loss(self, smashed_words_embedding, actual_similarity):
        return (self.similarity_symbolic(smashed_words_embedding) - actual_similarity) ** 2

    # def compare_symbolic(self, correct_sequence_embedding, error_sequence_embedding, logistic_scaling_factor=1.0):
    #     score_difference = self.score_symbolic(correct_sequence_embedding) - self.score_symbolic(error_sequence_embedding)
    #     return T.log(1 + T.exp(logistic_scaling_factor * -1 * score_difference))

    def _build_functions(self):
        # create symbolic variables for correct and error input
        w1_embedding = T.vector(name='w1_embedding')
        w2_embedding = T.vector(name='w2_embedding')
        embeddings = [w1_embedding, w2_embedding]
        training_similarity = T.scalar(name='similarity')

        smashed_embedding = self.embedding_layer.flatten_embeddings(embeddings)

        cost = self.loss(smashed_embedding, training_similarity) + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        weighted_learning_rate = T.scalar(name='weighted_learning_rate')

        # update the params of the model using the gradients
        updates = [(param, param - weighted_learning_rate * T.grad(cost, param))
                   for param in self.params]

        dembeddings = T.grad(cost, embeddings)

        inputs = embeddings + [training_similarity] + [weighted_learning_rate]
        outputs = dembeddings + [cost]

        self.training_function = theano.function(inputs=inputs,
                                                 outputs=outputs,
                                                 updates=updates)

        self.similarity = theano.function(inputs=embeddings,
                                           outputs=self.similarity_symbolic(self.embedding_layer.flatten_embeddings(embeddings)))

    def train(self, w1_index, w2_index, actual_similarity, weighted_learning_rate=0.01):
        w1_embedding = self.embedding_layer.embeddings_from_symbols(w1_index)
        w2_embedding = self.embedding_layer.embeddings_from_symbols(w2_index)

        dw1, dw2, cost = self.training_function(w1_embedding,
                                                w2_embedding,
                                                actual_similarity,
                                                weighted_learning_rate)

        w1_update = -weighted_learning_rate * dw1
        w2_update = -weighted_learning_rate * dw2

        self.embedding_layer.update_embeddings([w1_index, w2_index], [w1_update, w2_update])

        return cost, w1_update, w2_update

    def score(self, w1_index, w2_index):
        w1_embedding = self.embedding_layer.embeddings_from_symbols(w1_index)
        w2_embedding = self.embedding_layer.embeddings_from_symbols(w2_index)
        return self.similarity(w1_embedding, w2_embedding)

    def get_embeddings(self):
        return self.embedding_layer.embedding

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='either "net" or "distance"')
    parser.add_argument('--model_basename', help='save to this file')
    parser.add_argument('--dimensions', type=int, default=50)
    parser.add_argument('--sampling', type=str, default='semantic_nearest or embedding_nearest or random')
    parser.add_argument('--k_nearest', type=int, default=20)
    parser.add_argument('--n_hidden', type=int, default=50, help="only valid if type is net")
    args = parser.parse_args()
    num_epochs = None
    N = 50000
    k_nearest = args.k_nearest
    print 'loading reader'
    reader = NgramReader('/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', vocab_size=N)
    print 'loading semantic module'
    word_similarity = semantic_module.WordSimilarity(reader.word_array, '/cl/nldata/books_google_ngrams_eng/wordnet_similarities_max.npy')
    rng = np.random.RandomState(1234)
    print 'initializing network'
    other_params = {
        'N': N,
        'k_nearest': k_nearest,
        'sampling': args.sampling,
    }
    if args.type == 'net':
        network = SemanticNet(rng, word_similarity.vocabulary, args.dimensions, args.n_hidden, 0, 0, other_params=other_params)
    elif args.type == 'distance':
        network = SemanticDistance(rng, word_similarity.vocabulary, args.dimensions)
    else:
        print 'bad type %s' % args.type
    epoch = 0
    SAVE_EVERY = 10
    while True:
        epoch += 1
        this_count = 0
        costs = []
        for i in rng.permutation(N):
            this_count += 1
            if i == 0:
                continue # skip rare word w/ undef similarities
            if args.sampling == 'semantic_nearest':
                for j, sim in word_similarity.most_similar_indices(i, top_n = k_nearest):
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = network.train(i, j, sim)
                    costs.append(cost)
            elif args.sampling == 'embedding_nearest':
                for j, embedding_dist in network.embedding_layer.most_similar_embeddings(i, top_n=k_nearest):
                    sim = word_similarity.word_pairwise_sims[i, j]
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = network.train(i, j, sim)
                    costs.append(cost)
            elif args.sampling == 'random':
                for j in rng.permutation(N)[:k_nearest]:
                    sim = word_similarity.word_pairwise_sims[i, j]
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = network.train(i, j, sim)
                    costs.append(cost)

            sys.stdout.write('\r epoch %d: %d / %d\r' % (epoch, this_count, N))
            sys.stdout.flush()
        print 'epoch %d complete\ttraining cost %f' % (epoch, np.mean(costs))
        if epoch % SAVE_EVERY == 0 and args.model_basename:
            with gzip.open('%s-%d.pkl.gz'% (args.model_basename, epoch), 'wb') as f:
                cPickle.dump(network, f)
