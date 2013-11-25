import theano
import theano.tensor as T
import numpy as np
from picklable import Picklable

def hessian_diagonal(scalar, x, grad=None):
    if grad is None:
        grad = T.grad(scalar, x)

    flat_grad = T.flatten(grad)

    flat_grad2, updates = theano.scan(fn = lambda i, x, flat_grad: T.flatten(T.grad(flat_grad[i], x))[i],
                                 sequences=T.arange(flat_grad.shape[0]), non_sequences=[x, flat_grad])
    return T.reshape(flat_grad2, x.shape)

def hessian_test():
    x = T.dvector('x')
    z = T.dscalar('z')
    y = T.sum(T.cos(x*z))

    x_test = np.random.normal(size=10)
    z_test = np.random.rand()

    grad2 = hessian_diagonal(y, x)

    print 'compiling'
    import time
    t = time.clock()
    f = theano.function([x,z], grad2)
    print time.clock() - t
    t = time.clock()
    print 'Right answer: %s\n computed answer: %s'%(-1 * z_test**2 *np.cos(x_test*z_test), f(x_test,z_test))
    print time.clock() - t

def hessian_test_matrix():
    x = T.dmatrix('x')
    z = T.dscalar('z')
    y = T.sum(T.cos(x*z))

    x_test = np.random.normal(size=(3,3))
    z_test = np.random.rand()

    grad2 = hessian_diagonal(y, x)

    print 'compiling'
    import time
    t = time.clock()
    f = theano.function([x,z], grad2)
    print time.clock() - t
    t = time.clock()
    print 'Right answer: %s\n computed answer: %s'%(-1 * z_test**2 *np.cos(x_test*z_test), f(x_test,z_test))
    print time.clock() - t

def hessian_test_scalar():
    x = T.dscalar('x')
    z = T.dscalar('z')
    y = T.sum(T.cos(x*z))

    x_test = np.random.rand()
    z_test = np.random.rand()

    grad2 = hessian_diagonal(y, x)

    print 'compiling'
    import time
    t = time.clock()
    f = theano.function([x,z], grad2)
    print time.clock() - t
    t = time.clock()
    print 'Right answer: %s\n computed answer: %s'%(-1 * z_test**2 *np.cos(x_test*z_test), f(x_test,z_test))
    print time.clock() - t

class Policy(Picklable):
    pass

class SGD(Policy):
    def _shared_attrs(self):
        return ['learning_rate']

    def _nonshared_attrs(self):
        return ['param']

    def __init__(self, learning_rate, param):
        learning_rate = np.cast[theano.config.floatX](learning_rate)
        self._set_attrs(learning_rate=learning_rate, param=param)

    def updates(self, cost):
        grad = T.grad(cost, self.param)
        return [(self.param, self.param - self.learning_rate * grad)]

    def updates_indexed(self, cost, index_list, slice_list):
        indices = T.stack(*index_list)
        grad = T.stack(*T.grad(cost, slice_list))
        return [(self.param, T.inc_subtensor(self.param[indices],
                                             -1 * self.learning_rate * grad))]

class VSGD(Policy):
    def _shared_attrs(self):
        return [
            'learning_rate',
            'tau',
            'h_avg',
            'g_avg',
            'v_avg',
        ]

    def _nonshared_attrs(self):
        return ['param']

    def __init__(self, learning_rate, param):
        learning_rate = np.cast[theano.config.floatX](learning_rate)
        param_value = param.get_value()
        h_avg = np.zeros_like(param_value, dtype=theano.config.floatX)
        g_avg = np.zeros_like(param_value, dtype=theano.config.floatX)
        v_avg = np.zeros_like(param_value, dtype=theano.config.floatX)
        tau = np.ones_like(param_value, dtype=theano.config.floatX)
        self._set_attrs(
            param=param,
            learning_rate=learning_rate,
            h_avg=h_avg,
            g_avg=g_avg,
            v_avg=v_avg,
            tau=tau,
        )

    def updates(self, cost):
        grad = T.grad(cost, self.param)
        grad2 = hessian_diagonal(cost, self.param, grad=grad)
        # calculate memory constants
        tau_rec = 1.0 / self.tau
        tau_inv_rec = 1.0 - tau_rec

        # new moving average of gradient
        g_avg_new = tau_inv_rec * self.g_avg + tau_rec * grad
        # new moving average of squared gradient
        v_avg_new = tau_inv_rec * self.v_avg + tau_rec * grad**2
        # new moving average of hessian diagonal
        h_avg_new = tau_inv_rec * self.h_avg + tau_rec * grad2

        rate = (g_avg_new ** 2) / (v_avg_new * h_avg_new)
        tau_new = (1 - (g_avg_new ** 2) / v_avg_new) * self.tau + 1

        return [(self.g_avg, g_avg_new),
                (self.v_avg, v_avg_new),
                (self.h_avg, h_avg_new),
                (self.tau, tau_new),
                (self.param, self.param - rate * grad)]

    def updates_indexed(self, cost, index_list, slice_list):
        pass
