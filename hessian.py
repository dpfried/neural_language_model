import theano
import theano.tensor as T
import numpy as np

def hessian_diagonal(scalar, x, grad=None):
    if grad is None:
        grad = T.grad(scalar, x)

    flat_grad = T.flatten(grad)

    flat_grad2, updates = theano.scan(fn = lambda i, x, flat_grad: T.flatten(T.grad(flat_grad[i], x))[i],
                                 sequences=T.arange(flat_grad.shape[0]), non_sequences=[x, flat_grad])
    return T.reshape(flat_grad2, x.shape)

def hessian_test_harness(x, x_test):
    z = T.dscalar('z')
    y = T.sum(T.cos(x*z))

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

def hessian_test_vector():
    x = T.dvector('x')
    x_test = np.random.normal(size=10)
    hessian_test_harness(x, x_test)

def hessian_test_matrix():
    x = T.dmatrix('x')
    x_test = np.random.normal(size=(3,3))
    hessian_test_harness(x, x_test)

def hessian_test_scalar():
    x = T.dscalar('x')
    x_test = np.random.rand()
    hessian_test_harness(x, x_test)

