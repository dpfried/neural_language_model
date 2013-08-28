import theano
import theano.tensor as T
import numpy as np
from model import NLM
import cPickle
import time
import sys

def get_data():
    return # [(train_X, train_Y), (test_X, test_Y)]

def fake_data():
    summands = np.random.randint(10, size=(1000, 3))
    positive = summands * 2
    negative = positive + 1

    positive[:,1] = np.random.randint(10, size=(1000,))
    negative[:,1] = np.random.randint(10, size=(1000,))

    positive = np.mod(positive, 10)
    negative = np.mod(negative, 10)

    train = np.vstack([negative[:900], positive[:900]]) , np.hstack([np.array([0] * 900, dtype=np.int32), np.array([1] * 900, dtype=np.int32)])
    test = np.vstack([negative[900:], positive[900:]]) , np.hstack([np.array([0] * 100, dtype=np.int32), np.array([1] * 100, dtype=np.int32)])

    return train, test

def test_nlm(vocab_size, dimensions, n_hidden, data, rng=None, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0000, n_epochs=1000, batch_size=20, save_model_basename=None, epochs=np.inf):
    print '... building the model'

    ( train_set_x, train_set_y ), ( test_set_x, test_set_y ) = data

    n_train_instances, sequence_length = train_set_x.shape
    n_test_instances, _ = test_set_x.shape

    y = T.iscalar('y')
    if rng is None:
        rng = np.random.RandomState(1234)

    classifier = NLM(rng, vocab_size, dimensions, sequence_length, n_hidden, 2)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms; cost is expressed here
    # symbolically
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[classifier.one_hot_input, y],
                                 outputs=classifier.errors(y))

    # predict_model = theano.function(inputs=[classifier.one_hot_input],
    #                                 outputs=classifier.log_regression_layer.y_pred)

    # validate_model = theano.function(inputs=[classifier.one_hot_input],
    #                                  outputs=classifier.errors(y))

    # update the params of the model using the gradients
    updates = [(param, param - learning_rate * T.grad(cost, param))
               for param in classifier.params]

    train_model = theano.function(inputs=[classifier.one_hot_input, y],
                                  outputs=cost,
                                  updates=updates)

    # predict_probs = theano.function(inputs=[classifier.one_hot_input],
    #                                 outputs=classifier.log_regression_layer.p_y_given_x)

    # get_minibatch = lambda data, batch_index: classifier.one_hot_from_batch(data[batch_index * batch_size : (batch_index + 1) * batch_size])

    # print test_set_y[0:1000:50]
    # print [predict_probs(classifier.one_hot_from_symbols(test_set_x[test_index]))
    #        for test_index in xrange(n_test_instances)][0:1000:50]
    # print [int(predict_model(classifier.one_hot_from_symbols(test_set_x[test_index])))
    #        for test_index in xrange(n_test_instances)][0:1000:50]
    print '... training'

    last_time = time.clock()
    epoch = 0
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
            costs.append(train_model(classifier.one_hot_from_symbols(train_set_x[train_index]), train_set_y[train_index]))

        test_losses = []
        for test_index in xrange(n_test_instances):
            if test_index % print_freq == 0:
                sys.stdout.write('\rtesting instance %d of %d (%f %%)\r' % (test_index, n_test_instances, 100. * test_index / n_test_instances))
                sys.stdout.flush()
            test_losses.append(test_model(classifier.one_hot_from_symbols(test_set_x[test_index]),
                                          test_set_y[test_index]))
        this_training_cost = np.mean(costs)
        this_test_loss = np.mean(test_losses)

        # print test_set_y[0:1000:50]
        # print [predict_probs(classifier.one_hot_from_symbols(test_set_x[test_index]))
        #        for test_index in xrange(n_test_instances)][0:1000:50]
        # print [int(predict_model(classifier.one_hot_from_symbols(test_set_x[test_index])))
        #        for test_index in xrange(n_test_instances)][0:1000:50]
        # print 'losses', test_losses[0:1000:50]
        # print 'losses max', np.max(test_losses)

        current_time = time.clock()
        sys.stdout.write('\033[k\r')
        sys.stdout.flush()
        print 'epoch %i \t training cost %f %% \t test error %f %% \t %f seconds' % (epoch, this_training_cost, this_test_loss * 100., current_time - last_time)
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
    with gzip.open('data/n_grams.pkl.gz', 'rb') as f:
        n_grams = cPickle.load(f)
    # n_grams = ngrams_reader.NGramsContainer(ngrams_reader.DATA_BY_SIZE[0], num_words=5000)
    print 'read %i sentences, with %i word types. vocabulary is %i types' % (n_grams.n_examples, len(n_grams.frequency_counts), n_grams.vocab_size)
    # print 'dumping n_gram representation...'
    # with gzip.open('data/n_grams.pkl.gz', 'wb') as f:
    #     cPickle.dump(n_grams, f)

    print 'extracting data...'
    rng = np.random.RandomState(1234)
    data = n_grams.get_data(rng=rng, train_proportion=0.1, test_proportion=0.05)
    print 'constructing model...'
    params = {'rng':rng,
              'vocab_size':n_grams.vocab_size,
              'dimensions':20,
              'n_hidden':30,
              'L1_reg':0.0000,
              'L2_reg':0.0000,
              'save_model_basename':'data/small_20_30_l1_reg',
              'epochs':np.inf}
    print params
    params['data'] = data
    classifier = test_nlm(**params)
