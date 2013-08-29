import theano
import theano.tensor as T
import numpy as np
from model import NLM
import cPickle
import time
import sys

def test_nlm(vocab_size, dimensions, n_hidden, data, rng=None, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0000, n_epochs=1000, batch_size=20, save_model_basename=None, epochs=np.inf):
    print '... building the model'

    train_set_x, test_set_x = data

    n_train_instances, sequence_length = train_set_x.shape
    n_test_instances, _ = test_set_x.shape

    if rng is None:
        rng = np.random.RandomState(1234)

    classifier = NLM(rng, vocab_size, dimensions, sequence_length, n_hidden)

    # create symbolic variables for good and bad input
    correct_input = T.matrix(name='correct_one_hot_input')
    error_input = T.matrix(name='error_one_hot_input')

    cost = classifier.cost(correct_input, error_input) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # update the params of the model using the gradients
    updates = [(param, param - learning_rate * T.grad(cost, param))
               for param in classifier.params]

    train_model = theano.function(inputs=[correct_input, error_input],
                                  outputs=cost,
                                  updates=updates)

    score_ngram = theano.function(inputs=[correct_input],
                                  outputs=classifier.score(correct_input))

    def rank_ngram_symbols(symbols, replacement_column = sequence_length / 2):
        symbols = symbols.copy()
        rank = 1
        original_score = score_ngram(classifier.one_hot_from_symbols(symbols))
        for offset in range(vocab_size):
            symbols[replacement_column] += 1
            symbols = np.mod(symbols, vocab_size)
            if score_ngram(classifier.one_hot_from_symbols(symbols)) > original_score:
                rank += 1
        return rank

    print '... training'

    last_time = time.clock()
    epoch = 0
    epoch_test_frequency = 1
    while epoch < epochs:
        epoch += 1
        # for minibatch_index in xrange(n_train_batches):
        print_freq = 100
        costs = []
        # for stochastic gradient descent, feed each training example in random order
        for count, train_index in enumerate(rng.permutation(n_train_instances)):
            if count % print_freq == 0:
                sys.stdout.write('\rtraining instance %d of %d (%f %%)\r' % (count, n_train_instances, 100. * count / n_train_instances))
                sys.stdout.flush()
            correct_symbols = train_set_x[train_index]
            error_symbols = ngrams_reader.add_noise_to_symbols(correct_symbols, vocab_size, rng=rng)
            costs.append(train_model(classifier.one_hot_from_symbols(correct_symbols),
                                     classifier.one_hot_from_symbols(error_symbols)))

        this_training_cost = np.mean(costs)

        if epoch % epoch_test_frequency == 0:
            test_ranks = []
            for test_index in xrange(n_test_instances):
                if test_index % (print_freq / 100) == 0:
                    sys.stdout.write('\rtesting instance %d of %d (%f %%)\r' % (test_index, n_test_instances, 100. * test_index / n_test_instances))
                    sys.stdout.flush()
                test_ranks.append(rank_ngram_symbols(test_set_x[test_index]))
            this_average_rank = np.mean(test_ranks)
            this_average_rank_str = '%f' % this_average_rank
        else:
            this_average_rank_str = '--'

        current_time = time.clock()
        sys.stdout.write('\033[k\r')
        sys.stdout.flush()
        print 'epoch %i \t training cost %f %% \t test avg rank %s %% \t %f seconds' % (epoch, this_training_cost, this_average_rank_str, current_time - last_time)
        last_time = current_time

        if save_model_basename:
            sys.stdout.write('dumping to file..\r')
            sys.stdout.flush()
            with gzip.open('%s-%d.pkl.gz' % (save_model_basename, epoch), 'wb') as f:
                cPickle.dump(classifier, f)
            sys.stdout.write('\033[k\r')
            sys.stdout.flush()

    return classifier

if __name__ == '__main__':
    import ngrams_reader
    import gzip
    print 'loading n_grams...'
    # with gzip.open('data/n_grams.pkl.gz', 'rb') as f:
    #     n_grams = cPickle.load(f)
    n_grams = ngrams_reader.NGramsContainer(ngrams_reader.DATA_BY_SIZE[0], num_words=5000)
    print 'read %i sentences, with %i word types. vocabulary is %i types' % (n_grams.n_examples, len(n_grams.frequency_counts), n_grams.vocab_size)
    print 'dumping n_gram representation...'
    with gzip.open('data/n_grams_size0_data.pkl.gz', 'wb') as f:
        cPickle.dump(n_grams, f)

    print 'extracting data...'
    rng = np.random.RandomState(1234)
    data = n_grams.get_data(rng=rng, train_proportion=0.1, test_proportion=0.0002)
    print 'constructing model...'
    params = {'rng':rng,
              'vocab_size':n_grams.vocab_size,
              'dimensions':20,
              'n_hidden':30,
              'L1_reg':0.0000,
              'L2_reg':0.0000,
              'save_model_basename':'data/rank/size0-0.1_tinytrain_20_30',
              'epochs':np.inf}
    print params
    params['data'] = data
    classifier = test_nlm(**params)
