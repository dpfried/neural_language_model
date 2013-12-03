import theano
import theano.tensor as T
import numpy as np
from picklable import Picklable
import pprint
from hessian import hessian_diagonal

class Policy(Picklable):
    """
    implements a gradient descent policy for a given parameter
    """
    def __init__(self, learning_rate, param):
        learning_rate = np.cast[theano.config.floatX](learning_rate)
        self._set_attrs(learning_rate=learning_rate, param=param)

    def updates(self, cost):
        """
        return the update for this param
        """
        return []

    def updates_indexed(self, cost, index_list, slice_list):
        """
        used by methods that only update a subset of parameters
        """
        return []

    def burn_in_updates(self, cost):
        """
        if this policy requires initialization (i.e. vSGD), do it with these updates
        """
        return []

    def burn_in_updates_indexed(self, cost, index_list, slice_list):
        """
        if this policy requires initialization (i.e. vSGD), do it with these updates,
        used by methods that only update a subset of parameters
        """
        return []

    def afterburn(self):
        """
        initialize parameters after the burn-in period
        """
        pass

class SGD(Policy):
    """
    stochastic gradient descent, non-annealing
    """
    def _shared_attrs(self):
        return ['learning_rate']

    def _nonshared_attrs(self):
        return ['param']

    def updates(self, cost):
        grad = T.grad(cost, self.param)
        return [(self.param, self.param - self.learning_rate * grad)]

    def updates_indexed(self, cost, index_list, slice_list):
        indices = T.stack(*index_list)
        grad = T.stack(*T.grad(cost, slice_list))
        return [(self.param, T.inc_subtensor(self.param[indices],
                                             -1 * self.learning_rate * grad))]

class VSGD(Policy):
    """
    unfinished implementation of variance-based SGD, from Schaul et al ``No More Pesky Learning Rates''

    not used, symbolic method of finding hessians with Theano is too slow!
    better to use finite differences method?
    """
    def _shared_attrs(self):
        return [
            'learning_rate',
            'tau',
            'h_avg',
            'g_avg',
            'v_avg',
            'last_grad',
            'last_grad2',
            'last_rate',
            'N',
        ]

    def _nonshared_attrs(self):
        return ['param', ('n0', 100), ('C', 10), ('epsilon', 1e-20), ('burned', False)]

    def __repr__(self):
        return pprint.pformat({
            'h': self.h_avg.get_value(),
            'g': self.g_avg.get_value(),
            'v': self.v_avg.get_value(),
            'tau': self.tau.get_value(),
            'last_grad': self.last_grad.get_value(),
            'last_grad2': self.last_grad2.get_value(),
            'last_rate': self.last_rate.get_value(),
            'N': self.N.get_value(),
        })

    def __init__(self, learning_rate, param, n0=100, C=10, epsilon=1e-20):
        learning_rate = np.cast[theano.config.floatX](learning_rate)
        param_value = param.get_value()
        h_avg = np.zeros_like(param_value, dtype=theano.config.floatX)
        g_avg = np.zeros_like(param_value, dtype=theano.config.floatX)
        v_avg = np.zeros_like(param_value, dtype=theano.config.floatX)
        last_grad = np.zeros_like(param_value, dtype=theano.config.floatX)
        last_grad2 = np.zeros_like(param_value, dtype=theano.config.floatX)
        last_rate = np.zeros_like(param_value, dtype=theano.config.floatX)
        tau = np.zeros_like(param_value, dtype=theano.config.floatX)
        self._set_attrs(
            param=param,
            learning_rate=learning_rate,
            h_avg=h_avg,
            g_avg=g_avg,
            v_avg=v_avg,
            last_grad=last_grad,
            last_grad2=last_grad2,
            last_rate=last_rate,
            tau=tau,
            N=0,
            n0=n0,
            C=C,
            epsilon=epsilon,
            burned=False,
        )

    def burn_in_updates(self, cost):
        grad = T.grad(cost, self.param)
        grad2 = hessian_diagonal(cost, self.param, grad=grad)
        print 'burn in updates for %s' % self.param
        return [
            (self.g_avg, self.g_avg + grad),
            (self.h_avg, self.h_avg + T.abs_(grad2)),
            (self.v_avg, self.v_avg + grad**2),
            (self.N, self.N + 1)
        ]

    def afterburn(self):
        if self.burned:
            print 'already afterburned!'
            return
        else:
            print 'afterburning %s' % self.param
        self.burned = True
        self.g_avg.set_value(self.g_avg.get_value() / self.N.get_value())
        hess = self.h_avg.get_value() / self.N.get_value() * self.C
        self.h_avg.set_value(np.where(hess < self.epsilon, self.epsilon, hess))
        self.v_avg.set_value(self.v_avg.get_value() / self.N.get_value() * self.C)
        self.tau.set_value(np.ones_like(self.tau.get_value()) * self.N.get_value())

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
        h_avg_new = tau_inv_rec * self.h_avg + tau_rec * T.abs_(grad2)

        rate_unsafe = (g_avg_new ** 2) / (v_avg_new * h_avg_new)
        rate = T.switch(T.isinf(rate_unsafe) | T.isnan(rate_unsafe), self.learning_rate, rate_unsafe)

        tau_unsafe = (1 - (g_avg_new ** 2) / v_avg_new) * self.tau + 1
        tau_new = T.switch(T.isnan(tau_unsafe) | T.isinf(tau_unsafe), self.tau, tau_unsafe)

        return [(self.g_avg, g_avg_new),
                (self.v_avg, v_avg_new),
                (self.h_avg, h_avg_new),
                (self.tau, tau_new),
                (self.last_grad, grad),
                (self.last_grad2, grad2),
                (self.last_rate, rate),
                (self.param, self.param - rate * grad)]

    def updates_indexed(self, cost, index_list, slice_list):
        # TODO: finish this
        return self.updates(cost)
        # slice_grads = T.grad(cost, slice_list)
        # slice_hessians = []
        # for slice_, slice_grad in zip(slice_list, slice_grads):
        #     slice_grad = T.grad(cost, slice_)
        #     slice_hessians.append(hessian_diagonal(cost, slice_, grad=slice_grad))
        # grad = T.stack(*slice_grads)
        # grad2 = T.stack(*slice_hessians)

        # indices = T.stack(*index_list)

        # # calculate memory constants
        # tau_rec = 1.0 / self.tau[indices]
        # tau_inv_rec = 1.0 - tau_rec

        # # new moving average of gradient
        # g_avg_new = tau_inv_rec * self.g_avg[indices] + tau_rec * grad
        # # new moving average of squared gradient
        # v_avg_new = tau_inv_rec * self.v_avg[indices] + tau_rec * grad**2
        # # new moving average of hessian diagonal
        # h_avg_new = tau_inv_rec * self.h_avg[indices] + tau_rec * grad2

        # rate = (g_avg_new ** 2) / (v_avg_new * h_avg_new)
        # tau_new = (1 - (g_avg_new ** 2) / v_avg_new) * self.tau[indices] + 1

        # return [(self.g_avg, T.set_subtensor(self.g_avg[indices], g_avg_new)),
        #         (self.v_avg, T.set_subtensor(self.v_avg[indices], v_avg_new)),
        #         (self.h_avg, T.set_subtensor(self.h_avg[indices], h_avg_new)),
        #         (self.tau, T.set_subtensor(self.tau[indices], tau_new)),
        #         (self.param, T.inc_subtensor(self.param[indices], - rate * grad))]
