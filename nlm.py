import theano
import theano.tensor as T
import numpy as np
from model import NLM
import cPickle

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

def test_nlm(vocab_size, dimensions, n_hidden, data, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0000, n_epochs=1000, batch_size=20, save_model_fname=None, epochs=np.inf):
    print '... building the model'

    ( train_set_x, train_set_y ), ( test_set_x, test_set_y ) = data

    n_train_instances, sequence_length = train_set_x.shape
    n_test_instances, _ = test_set_x.shape

    y = T.iscalar('y')
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

    predict_model = theano.function(inputs=[classifier.one_hot_input],
                                    outputs=classifier.log_regression_layer.y_pred)

    # validate_model = theano.function(inputs=[classifier.one_hot_input],
    #                                  outputs=classifier.errors(y))

    # update the params of the model using the gradients
    updates = [(param, param - learning_rate * T.grad(cost, param))
               for param in classifier.params]

    train_model = theano.function(inputs=[classifier.one_hot_input, y],
                                  outputs=cost,
                                  updates=updates)

    predict_probs = theano.function(inputs=[classifier.one_hot_input],
                                    outputs=classifier.log_regression_layer.p_y_given_x)

    # get_minibatch = lambda data, batch_index: classifier.one_hot_from_batch(data[batch_index * batch_size : (batch_index + 1) * batch_size])

    print test_set_y[0:1000:50]
    print [predict_probs(classifier.one_hot_from_symbols(test_set_x[test_index]))
           for test_index in xrange(n_test_instances)][0:1000:50]
    print [int(predict_model(classifier.one_hot_from_symbols(test_set_x[test_index])))
           for test_index in xrange(n_test_instances)][0:1000:50]
    print '... training'

    epoch = 0
    while epoch < epochs:
        epoch += 1
        # for minibatch_index in xrange(n_train_batches):
        for train_index in xrange(n_train_instances):
            train_model(classifier.one_hot_from_symbols(train_set_x[train_index]), train_set_y[train_index])

        test_losses = [test_model(classifier.one_hot_from_symbols(test_set_x[test_index]),
                                  test_set_y[test_index])
                       for test_index in xrange(n_test_instances)]
        this_test_loss = np.mean(test_losses)

        print test_set_y[0:1000:50]
        print [predict_probs(classifier.one_hot_from_symbols(test_set_x[test_index]))
               for test_index in xrange(n_test_instances)][0:1000:50]
        print [int(predict_model(classifier.one_hot_from_symbols(test_set_x[test_index])))
               for test_index in xrange(n_test_instances)][0:1000:50]
        print 'losses', test_losses[0:1000:50]
        print 'losses max', np.max(test_losses)


        print 'epoch %i, test error %f %%' % (epoch, this_test_loss * 100.)

        if save_model_fname:
            with open(save_model_fname, 'wb') as f:
                cPickle.dump(classifier, f)

    return classifier

if __name__ == '__main__':
    classifier = test_nlm(vocab_size=10, dimensions=5, n_hidden=10, data=fake_data(), save_model_fname='model.pkl', epochs=10)
