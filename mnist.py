import os
import gzip
import cPickle
import theano
import theano.tensor as T
import numpy as np
import time
from network import MLP

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an np.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #np.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='data/mnist.pkl.gz', batch_size=20, n_hidden=500):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    index = T.lscalar() # index to a [mini]batch
    x = T.matrix('x') # data presented as rasterized images
    y = T.ivector('y') # labvels presented as a 1D vector of [int] labels

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=n_hidden, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms; cost is expressed here
    # symbolically
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                     y: test_set_y[index * batch_size:(index + 1) * batch_size]
                                 })

    validate_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(y),
                                     givens={
                                         x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size:(index + 1) * batch_size]
                                     })

    # update the params of the model using the gradients
    updates = [(param, param - learning_rate * T.grad(cost, param))
               for param in classifier.params]

    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size]
                                  })


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
