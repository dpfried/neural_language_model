from ngrams import NgramReader
import theano
import theano.tensor as T
import numpy as np
import gzip, cPickle
import semantic_module
import sys
import random
import time

from model import EmbeddingLayer, HiddenLayer, EmbeddingTrainer, EZPickle

class SemanticDistance(EmbeddingTrainer, EZPickle):
    SHARED = [('learning_rate', 0.01)]

    OTHERS = ['blocks_trained',
              ('mode', 'FAST_RUN'),
              'embedding_layer',
              'dimensions',
              'vocab_size']

    def init_params(self, **kwargs):
        super(SemanticDistance, self).init_params(**kwargs)
        self.make_stack()
        self.make_functions()

    def make_stack(self):
        # no params to update (overridden by SemanticNet)
        self.params = []
        self.layer_stack = []

    def __init__(self, rng, vocabulary, dimensions, initial_embeddings=None, mode='FAST_RUN', learning_rate=0.01):
        vocab_size = len(vocabulary)
        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=vocab_size,
                                         dimensions=dimensions,
                                         sequence_length=1,
                                         initial_embeddings=initial_embeddings,
                                         mode=mode)
        learning_rate = np.cast[theano.config.floatX](learning_rate)
        self.init_params(learning_rate=learning_rate,
                         blocks_trained=0,
                         mode=mode,
                         embedding_layer=embedding_layer,
                         dimensions=dimensions,
                         vocab_size=vocab_size)

    def embed_indices_symbolic(self, indices):
        return self.embedding_layer.embed_indices_symbolic(indices)

    def similarity_symbolic(self, w1, w2):
        return T.dot(w1, w2) / T.sqrt(T.dot(w1, w1) * T.dot(w2, w2))

    def cost_from_embeddings_symbolic(self, w1, w2, actual_similarity):
        return (self.similarity_symbolic(w1, w2) - actual_similarity) ** 2

    # def compare_symbolic(self, correct_sequence_embedding, error_sequence_embedding, logistic_scaling_factor=1.0):
    #     score_difference = self.score_symbolic(correct_sequence_embedding) - self.score_symbolic(error_sequence_embedding)
    #     return T.log(1 + T.exp(logistic_scaling_factor * -1 * score_difference))

    def updates_symbolic(self, cost, index1, index2, w1, w2):
        d1 = T.grad(cost, w1)
        d2 = T.grad(cost, w2)
        dembeddings = T.stack(d1, d2)

        indices = T.stack(index1, index2)

        return [(self.embedding_layer.embedding,
                 T.inc_subtensor(self.embedding_layer.embedding[indices], -self.learning_rate * dembeddings))]

    def make_functions(self):
        # create symbolic variables for correct and error input
        index1 = T.scalar(name='index1', dtype='int32')
        index2 = T.scalar(name='index2', dtype='int32')

        w1 = self.embed_indices_symbolic(index1)
        w2 = self.embed_indices_symbolic(index2)
        training_similarity = T.scalar(name='similarity')

        cost = self.cost_from_embeddings_symbolic(w1, w2, training_similarity)

        self.train = theano.function(inputs=[index1, index2, training_similarity],
                                     outputs=cost,
                                     updates=self.updates_symbolic(cost, index1, index2, w1, w2),
                                     mode=self.mode)

        self.similarity = theano.function(inputs=[index1, index2],
                                          outputs=self.similarity_symbolic(w1, w2),
                                          mode=self.mode)

    def get_embeddings(self):
        return self.embedding_layer.get_embeddings()

class SemanticNet(SemanticDistance):
    SHARED = SemanticDistance.SHARED

    OTHERS = SemanticDistance.OTHERS + ['hidden_layer', 'output_layer']

    def init_params(self, **kwargs):
        super(SemanticNet, self).init_params(**kwargs)

    def make_stack(self):
        self.params = self.hidden_layer.params + self.output_layer.params
        self.layer_stack = [self.hidden_layer, self.output_layer]

    def __init__(self, rng, vocabulary, dimensions, n_hidden, other_params=None, initial_embeddings=None, learning_rate=0.01, mode='FAST_RUN'):
        EmbeddingTrainer.__init__(self, rng, vocabulary, dimensions)
        # initialize parameters
        if other_params is None:
            other_params = {}

        embedding_layer = EmbeddingLayer(rng,
                                         vocab_size=len(vocabulary),
                                         dimensions=dimensions,
                                         sequence_length=2,
                                         initial_embeddings=initial_embeddings)

        hidden_layer = HiddenLayer(rng=rng,
                                   n_in=dimensions * 2,
                                   n_out=n_hidden,
                                   activation=T.tanh)

        output_layer = HiddenLayer(rng=rng,
                                   n_in=n_hidden,
                                   n_out=1,
                                   activation=T.tanh)

        self.init_params(learning_rate=0.01,
                         blocks_trained=0,
                         mode=mode,
                         dimensions=dimensions,
                         other_params=other_params,
                         vocab_size=len(vocabulary),
                         embededing_layer=embedding_layer,
                         hidden_layer=hidden_layer,
                         output_layer=output_layer,
                         )

    def similarity_symbolic(self, w1, w2):
        smashed_words_embedding = T.concatenate([w1, w2])
        return reduce(lambda layer_input, layer: layer.apply(layer_input), self.layer_stack, smashed_words_embedding)

    def updates_symbolic(self, cost, index1, index2, w1, w2):
        embedding_updates = super(SemanticNet, self).updates_symbolic(cost, index1, index2, w1, w2)
        other_updates = [(param, param - self.learning_rate * T.grad(cost, param))
                         for param in self.params]
        return other_updates + embedding_updates

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='either "net" or "distance"')
    parser.add_argument('--model_basename', help='save to this file')
    parser.add_argument('--word_similarity_memmap', default='/tmp/wordnet_similarities_max.memmap', help='use this file as a shared memmap between processes')
    parser.add_argument('--dimensions', type=int, default=50)
    parser.add_argument('--sampling', type=str, default='random', help='semantic_nearest or embedding_nearest or random')
    parser.add_argument('--k_nearest', type=int, default=20)
    parser.add_argument('--n_hidden', type=int, default=50, help="only valid if type is net")
    parser.add_argument('--mode', default='FAST_RUN')
    args = parser.parse_args()
    num_epochs = None
    N = 50000
    k_nearest = args.k_nearest
    print 'loading reader'
    reader = NgramReader('/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', vocab_size=N)
    print 'loading semantic module'
    word_similarity = semantic_module.WordSimilarity(reader.word_array, '/cl/nldata/books_google_ngrams_eng/wordnet_similarities_max.npy', args.word_similarity_memmap)
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
        network = SemanticDistance(rng, word_similarity.vocabulary, args.dimensions, mode=args.mode)
    else:
        print 'bad type %s' % args.type
    epoch = 0
    SAVE_EVERY = 10
    last_time = time.clock()
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
                    cost = network.train(i, j, sim)
                    costs.append(cost)
            elif args.sampling == 'embedding_nearest':
                for j, embedding_dist in network.embedding_layer.most_similar_embeddings(i, top_n=k_nearest):
                    sim = word_similarity.word_pairwise_sims[i, j]
                    if sim == -np.inf:
                        continue
                    cost = network.train(i, j, sim)
                    costs.append(cost)
            elif args.sampling == 'random':
                for j in random.sample(xrange(N), k_nearest):
                    sim = word_similarity.word_pairwise_sims[i, j]
                    if sim == -np.inf:
                        continue
                    cost = network.train(i, j, sim)
                    costs.append(cost)
            else:
                raise ValueError('bad argument %s for --sampling' % args.sampling)

            if this_count % 100 == 0:
                sys.stdout.write('\r epoch %d: %d / %d\r' % (epoch, this_count, N))
                sys.stdout.flush()
        current_time = time.clock()
        elapsed = current_time - last_time
        print 'epoch %d complete\ttraining cost %f\t%f' % (epoch, np.mean(costs), elapsed)
        last_time = current_time

        if epoch % SAVE_EVERY == 0 and args.model_basename:
            with gzip.open('%s-%d.pkl.gz'% (args.model_basename, epoch), 'wb') as f:
                cPickle.dump(network, f)
