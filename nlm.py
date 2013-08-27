import theano
import theano.tensor as T
import numpy as np
from model import NLM
import time
import sys
import os

def get_data():
    return # [(train_X, train_Y), (test_X, test_Y)]

def test_mlp(vocab_size, dimensions, n_hidden, sequence_length=5, n_out=1, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20):
    print '... building the model'

    train_set_x, train_set_y, test_set_x, test_set_y = get_data()

    n_train_batches = train_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    y = T.ivector('y') # labvels presented as a 1D vector of [int] labels
    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = NLM(rng, vocab_size, dimensions, sequence_length, n_hidden, n_out)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms; cost is expressed here
    # symbolically
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[classifier.one_hot_input, y],
                                 outputs=classifier.errors(y))

    # validate_model = theano.function(inputs=[classifier.one_hot_input],
    #                                  outputs=classifier.errors(y))

    # update the params of the model using the gradients
    updates = [(param, param - learning_rate * T.grad(cost, param))
               for param in classifier.params]

    train_model = theano.function(inputs=[classifier.one_hot_input, y],
                                  outputs=cost,
                                  updates=updates)


    print '... training'

    # early-stopping parameters
    patience = 10000 # look at this many examples regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.995 # a relative importance of this much is considered significant

    # go through this many minibatches before checking the network on the
    # validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print 'epoch %i, minibatch %i/%i, validation error %f %%' % (
                    epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print '\tepoch %i, minibatch %i/%i, test error of best model %f %%' % (
                            epoch, minibatch_index + 1, n_train_batches, test_score * 100.
                        )

                if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print 'Optimization complete. Best validation score of %f %% obtained at iteration %i, w/ test perf of\
    %f %%' % (best_validation_loss * 100., best_iter + 1, test_score * 100.)

    print >> sys.stderr, 'The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_time - start_time) / 60.)

if __name__ == '__main__':
    test_mlp()
