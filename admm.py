import theano
import theano.tensor as T
from model import NLM
from semantic_network import SemanticDistance
from ngrams import NgramReader
import numpy as np
from utils import grouper, sample_cumulative_discrete_distribution
import semantic_module
import gzip, cPickle
import sys

class ADMMModel(object):
    def __init__(self, syntactic_model, semantic_model, rho, other_params, y_init=1.0, semantic_gd_rate=0.1, syntactic_gd_rate=0.1):
        self.syntactic_model = syntactic_model
        self.semantic_model = semantic_model
        self.rho = rho
        self.other_params = other_params
        self.y_init = y_init

        # the lagrangian
        self.y = np.ones((vocab_size,syntactic_model.dimensions)) * y_init

        # self.y = theano.shared(value=y_init, name='y')
        self.semantic_gd_rate = theano.shared(value=semantic_gd_rate, name='semantic_gd_rate')
        self.syntactic_gd_rate = theano.shared(value=syntactic_gd_rate, name='syntactic_gd_rate')

        self._build_functions()

    def admm_penalty(self, w, v, y):
        return T.dot(y, (w - v)) + self.rho / 2.0 * T.sqrt(T.dot((w - v).T, w - v))

    def _build_functions(self):
        self.syntactic_update_function = self.make_theano_syntactic_update()
        self.semantic_update_function = self.make_theano_semantic_update()

    @property
    def syntactic_embedding(self):
        return self.syntactic_model.embedding_layer.embedding

    @property
    def semantic_embedding(self):
        return self.semantic_model.embedding_layer.embedding

    def make_theano_syntactic_update(self):
        # build the update functions for w, the embeddings of the syntactic
        # model
        # these represent the embeddings from the semantic model for the good
        # and bad ngrams
        seq_length = self.syntactic_model.sequence_length

        w_correct_embeddings = [T.vector(name='w_correct_embedding%i' % i) for i in range(seq_length)]
        w_error_embeddings = [T.vector(name='w_error_embedding%i' % i) for i in range(seq_length)]
        w_embeddings = w_correct_embeddings + w_error_embeddings

        # these represent the corresponding embeddings from the semantic model
        v_correct_embeddings = [T.vector(name='v_correct_embedding%i' % i) for i in range(seq_length)]
        v_error_embeddings = [T.vector(name='v_error_embedding%i' % i) for i in range(seq_length)]
        v_embeddings = v_correct_embeddings + v_error_embeddings

        w = T.concatenate(w_embeddings)
        v = T.concatenate(v_embeddings)

        y_weights= [T.vector(name='y_weight%i' % i) for i in range(seq_length)]
        y = T.concatenate(y_weights)

        cost = self.syntactic_model.loss(w_correct_embeddings, w_error_embeddings) + self.admm_penalty(w, v, y)

        updates = [(param, param - self.syntactic_gd_rate * T.grad(cost, param))
                   for param in self.syntactic_model.params]

        dcorrect_embeddings = T.grad(cost, w_correct_embeddings)
        derror_embeddings = T.grad(cost, w_error_embeddings)

        return theano.function(inputs=w_embeddings + v_embeddings + y_weights,
                               outputs=dcorrect_embeddings + derror_embeddings + [cost],
                               updates=updates)

    def update_syntactic(self, correct_symbols, error_symbols):
        syntactic_correct = [self.syntactic_embedding[i] for i in correct_symbols]
        syntactic_error = [self.syntactic_embedding[i] for i in error_symbols]

        semantic_correct = [self.semantic_embedding[i] for i in correct_symbols]
        semantic_error = [self.semantic_embedding[i] for i in error_symbols]

        y_weights = [self.y[i] for i in correct_symbols + error_symbols]

        outputs = self.syntactic_update_function(*(syntactic_correct + syntactic_error + semantic_correct + semantic_error + y_weights))

        correct_grads, error_grads = list(grouper(self.syntactic_model.sequence_length, outputs))[:2]

        cost = outputs[-1]

        weight = self.syntactic_gd_rate.value

        correct_updates = - weight * np.array(correct_grads)
        error_updates = - weight * np.array(error_grads)

        self.syntactic_model.embedding_layer.update_embeddings(correct_symbols, correct_updates)
        self.syntactic_model.embedding_layer.update_embeddings(error_symbols, error_updates)

        return cost, correct_updates, error_updates


    def make_theano_semantic_update(self):
        w1, w2, v1, v2 = [T.vector(name='%s_embedding' % name) for name in ['w1', 'w2', 'v1', 'v2']]
        y1, y2 = T.vector('y1_weight'), T.vector('y2_weight')

        w = T.concatenate([w1, w2])
        v = T.concatenate([v1, v2])
        y = T.concatenate([y1, y2])

        actual_sim = T.scalar(name='semantic_similarity')

        cost = self.semantic_model.loss(v1, v2, actual_sim) + self.admm_penalty(w, v, y)

        updates = [(param, param - self.semantic_gd_rate * T.grad(cost, param))
                   for param in self.semantic_model.params]

        dv1 = T.grad(cost, v1)
        dv2 = T.grad(cost, v2)

        return theano.function(inputs=[w1, w2, v1, v2, y1, y2, actual_sim],
                               outputs=[dv1, dv2, cost],
                               updates=updates)

    def update_semantic(self, index1, index2, actual_similarity):
        w1 = self.syntactic_embedding[index1]
        w2 = self.syntactic_embedding[index2]

        v1 = self.semantic_embedding[index1]
        v2 = self.semantic_embedding[index2]

        y1 = self.y[index1]
        y2 = self.y[index2]

        dv1, dv2, cost = self.semantic_update_function(w1, w2, v1, v2, y1, y2, actual_similarity)

        weight = self.semantic_gd_rate.value

        self.semantic_model.embedding_layer.update_embeddings(index1, - weight * dv1)
        self.semantic_model.embedding_layer.update_embeddings(index2, - weight * dv2)

        return cost, dv1, dv2

    def update_y(self):
        w = self.syntactic_embedding
        v = self.semantic_embedding
        residual = w - v
        delta_y = self.rho * residual
        self.y += delta_y

        res = np.ravel(residual)
        y = np.ravel(self.y)
        res_norm = np.sqrt(np.dot(res, res))
        y_norm = np.sqrt(np.dot(y, y))
        return res_norm, y_norm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_basename')
    args = parser.parse_args()
    ngram_filename = '/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5'
    vocab_size = 50000
    train_proportion = 0.95
    test_proportion=0.0001
    dimensions = 75
    n_hidden = 200
    sequence_length = 5
    rho = 1.0
    y_init = 1.0
    semantic_gd_rate=0.1
    syntactic_gd_rate=0.1
    k_nearest=20
    word_similarity_file='/cl/nldata/books_google_ngrams_eng/wordnet_similarities_max.npy'

    sampling='semantic_nearest'

    other_params = {
        'ngram_filename' : ngram_filename,
        'vocab_size' : vocab_size,
        'train_proportion' : train_proportion,
        'test_proportion' : test_proportion,
        'dimensions' : dimensions,
        'n_hidden' : n_hidden,
        'sequence_length' : sequence_length,
        'rho' : rho,
        'y_init' : y_init,
        'semantic_gd_rate' : semantic_gd_rate,
        'syntactic_gd_rate' : syntactic_gd_rate,
        'k_nearest': k_nearest,
        'word_similarity_file' : word_similarity_file,
    }

    replacement_column_index = sequence_length / 2

    ngram_reader = NgramReader(ngram_filename, vocab_size=vocab_size, train_proportion=train_proportion, test_proportion=test_proportion)
    vocabulary = ngram_reader.word_array
    print 'corpus contains %i ngrams' % (ngram_reader.number_of_ngrams)
    print 'constructing model...'
    rng = np.random.RandomState(1234)
    syntactic_model = NLM(rng=rng,
                          vocabulary=vocabulary,
                          dimensions=dimensions,
                          sequence_length=sequence_length,
                          n_hidden=n_hidden,
                          L1_reg=0,
                          L2_reg=0)

    semantic_model = SemanticDistance(rng=rng,
                                      vocabulary=vocabulary,
                                      dimensions=dimensions)

    model = ADMMModel(syntactic_model,
                      semantic_model,
                      rho,
                      other_params,
                      y_init=y_init,
                      semantic_gd_rate=semantic_gd_rate,
                      syntactic_gd_rate=syntactic_gd_rate)

    print 'loading semantic similarities'
    word_similarity = semantic_module.WordSimilarity(vocabulary, word_similarity_file)

    print 'training...'

    k = 0
    print_freq = 100
    while True:
        k += 1
        # syntactic update step
        costs = []
        training_block = ngram_reader.training_block(rng.random_sample())
        block_size = training_block.shape[0]
        for count in xrange(block_size):
            if count % print_freq == 0:
                sys.stdout.write('\rk %i: ngram %d of %d (%f %%)\r' % (k, count, block_size, 100. * count / block_size))
                sys.stdout.flush()
            train_index = sample_cumulative_discrete_distribution(training_block[:,-1])
            correct_symbols, error_symbols, ngram_frequency = ngram_reader.contrastive_symbols_from_row(training_block[train_index], rng=rng)
            cost, correct_updates, error_updates = model.update_syntactic(correct_symbols, error_symbols)
            costs.append(cost)

        print 'syntactic mean cost %f\t\t' % np.mean(costs)

        # semantic update step
        this_count = 0
        costs = []
        for i in rng.permutation(vocab_size):
            this_count += 1
            if i == 0:
                continue # skip rare word w/ undef similarities
            if sampling == 'semantic_nearest':
                for j, sim in word_similarity.most_similar_indices(i, top_n = k_nearest):
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = model.update_semantic(i, j, sim)
                    costs.append(cost)
            elif sampling == 'embedding_nearest':
                for j, embedding_dist in model.semantic_model.embedding_layer.most_similar_embeddings(i, top_n=k_nearest):
                    sim = word_similarity.word_pairwise_sims[i, j]
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = model.update_semantic(i, j, sim)
                    costs.append(cost)
            elif sampling == 'random':
                for j in rng.permutation(vocab_size)[:k_nearest]:
                    sim = word_similarity.word_pairwise_sims[i, j]
                    if sim == -np.inf:
                        continue
                    cost, w1_update, w2_update = model.update_semantic(i, j, sim)
                    costs.append(cost)

            if this_count % print_freq == 0:
                sys.stdout.write('\r k %i: pair : %d / %d\r' % (k, this_count, vocab_size))
                sys.stdout.flush()

        print 'semantic mean cost %f\t\t' % np.mean(costs)

        # lagrangian update
        print 'updating y'
        res_norm, y_norm = model.update_y()
        print 'k: %d\tnorm(w - v) %f \t norm(y) %f' % (k, res_norm, y_norm)

        # dump it
        with gzip.open('%s-%d.pkl.gz'% (args.model_basename, k), 'wb') as f:
            cPickle.dump(model, f)
