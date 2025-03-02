#!/usr/bin/env python

import numpy as np

from .func import Function


class GenzBase(Function):
    """Base class for Genz function family.

    Sets up shift, dimensional weights and domain.

    Attributes
    -----------
    shifts : {number}
        Shift parameter.
    weights : {numpy array, 1d}
        Dimensional weights.

    """

    def __init__(self, weights=[1.0], domain=None, name='Genz'):
        """Initialization.

        Parameters
        ----------
        shift : {number}, optional
            Shift parameter (the default is 0.0, which means no shift)
        weights : {list or numpy array, 1d}, optional
            Dimensional weights (the default is [1.0],
                                 which means 1d function with weight 1)
        domain : {list or numpy array, 2d}, optional
            Input domain of the function (the default is None,
                                          which means [0,1]^d)
        """

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
    """Genz Oscillatory function.

    """
    def __init__(self, shift=0.0, weights=[1.0], domain=None,
                 name='Genz Oscillatory'):
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
    """Genz Oscillatory function.

    """
    def __init__(self, shift=0.0, weights=[1.0], domain=None,
                 name='Genz Summation'):
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
    """Genz Corner Peak function.

    """
    def __init__(self, weights=[1.0], domain=None, name='Genz Corner Peak'):
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


