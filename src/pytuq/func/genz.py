#!/usr/bin/env python
"""
    Genz function family module."""
import numpy as np

from .func import Function


class GenzBase(Function):
    """Base class for Genz function family

    Attributes:
        weights (np.ndarray): Dimensional weights.
    """

    def __init__(self, weights=[1.0], domain=None, name='GenzBase'):
        super().__init__()

        self.weights = np.array(weights)
        self.dim = self.weights.shape[0]
        self.name = name
        self.outdim = 1

        if domain is None:
            self.domain = np.dot(np.ones((self.dim, 2)), np.diag((0., 1.)))
        else:
            self.setDimDom(domain=domain)

        return


class GenzOscillatory(GenzBase):
    r"""Genz Oscillatory function

    Reference: [https://www.sfu.ca/~ssurjano/oscil.html]

    .. math::
        f(x) = \cos\left(2 \pi s + w^T x \right)

    Default values are :math:`s = 0` and :math:`w = [1.0]`.

    """
    def __init__(self, shift=0.0, weights=[1.0], domain=None,
                 name='GenzOscillatory'):
        super().__init__(weights=weights, domain=domain, name=name)
        self.shift = shift

    def __call__(self, x):

        self.checkDim(x)

        y = np.cos(2. * np.pi * self.shift + np.dot(x, self.weights))

        return y.reshape(-1,1)

    def grad(self, x):
        self.checkDim(x)

        grad = np.empty((x.shape[0], self.outdim, self.dim))
        for i in range(x.shape[1]):
            grad[:, 0, i] = -self.weights[i]*np.sin(2. * np.pi * self.shift + np.dot(x, self.weights))

        return grad

    def intgl(self):
        return np.cos(2. * np.pi * self.shift + 0.5 * np.sum(self.weights) ) * \
               np.prod(2. * np.sin(0.5 * self.weights) / self.weights)


class GenzSum(GenzBase):
    r"""Genz Summation function

    .. math::
        f(x) = s + w^T x

    Default values are :math:`s = 0` and :math:`w = [1.0]`.


    """
    def __init__(self, shift=0.0, weights=[1.0], domain=None,
                 name='GenzSummation'):
        super().__init__(weights=weights, domain=domain, name=name)
        self.shift = shift

    def __call__(self, x):


        self.checkDim(x)

        y = self.shift + np.dot(x, self.weights)

        return y.reshape(-1,1)

    def grad(self, x):
        self.checkDim(x)

        grad = np.empty((x.shape[0], self.outdim, self.dim))
        for i in range(self.dim):
            grad[:, 0, i] = self.weights[i]*np.ones(x.shape[0])

        return grad

class GenzCornerPeak(GenzBase):
    r"""Genz Corner Peak function

    Reference: [https://www.sfu.ca/~ssurjano/copeak.html]

    .. math::
        f(x) = \frac{1}{(1 + w^T x)^{d+1}}

     Default values are :math:`w = [1.0]`.


    """
    def __init__(self, weights=[1.0], domain=None, name='GenzCornerPeak'):
        super().__init__(weights=weights, domain=domain, name=name)

    def __call__(self, x):


        self.checkDim(x)
        y = 1. / (1. + np.dot(x, self.weights))**(self.dim + 1)
        return y.reshape(-1,1)

    def grad(self, x):
        self.checkDim(x)

        grad = np.empty((x.shape[0], self.outdim, self.dim))
        for i in range(x.shape[1]):
            grad[:, 0, i] = -(self.dim+1.)* self.weights[i]/ (1. + np.dot(x, self.weights))**(self.dim + 2)

        return grad


