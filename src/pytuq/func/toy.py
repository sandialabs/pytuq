#!/usr/bin/env python
"""
    Toy benchmark functions module."""
import numpy as np

from .func import Function


class Constant(Function):
    r"""Multioutput Constant function

    Returns a constant vector for any input.

    .. math::
        f(x) = (c, c, \ldots, c)

    """
    def __init__(self, dim=1, const=np.array([1.0]), name='Constant'):
        super().__init__()
        self.setDimDom(dimension=dim)
        self.name = name
        self.const = const
        self.outdim = const.shape[0]

        return

    def __call__(self, x):
        self.checkDim(x)

        nx = x.shape[0]
        yy = np.tile(self.const, (nx, 1))

        return yy


    def grad(self, x):
        return np.zeros((x.shape[0], self.outdim, self.dim))


class Identity(Function):
    r"""Identity function

    Returns the input unchanged.

    .. math::
        f(x) = x

    """
    def __init__(self, dim=1, name='Identity'):
        super().__init__()
        self.setDimDom(dimension=dim)
        self.name = name
        self.outdim = dim

        return

    def __call__(self, x):
        self.checkDim(x)

        return x


    def grad(self, x):
        nx = x.shape[0]
        grad = np.tile(np.eye(self.dim), (nx, 1, 1))
        return grad


class Quad(Function):
    r"""Quadratic function.

    .. math::
        f(x) = 3 + x - x^2

    """
    def __init__(self, name='Quad'):
        super().__init__()
        self.setDimDom(dimension=1)
        self.name = name
        self.outdim = 1

    def __call__(self, x):
        #: Ensure x is of the right dimensionality
        self.checkDim(x)


        yy = 3. + x[:, 0] - x[:, 0]**2

        return yy.reshape(-1,1)

    def grad(self, x):
        return (1. - 2.* x[:,0])[:, np.newaxis, np.newaxis]


class Quad2d(Function):
    r"""2D Quadratic function

    .. math::
        f(x) = 3 + x_1 - x_2^2

    """
    def __init__(self, name='Quad2d'):
        super().__init__()
        self.setDimDom(dimension=2)
        self.name = name
        self.outdim = 1

    def __call__(self, x):
        #: Ensure x is of the right dimensionality
        self.checkDim(x)


        yy = 3. + x[:, 0] - x[:, 1]**2

        return yy.reshape(-1,1)

    def grad(self, x):
        grad = np.empty((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = np.ones((x.shape[0],))
        grad[:, 0, 1] = -2. * x[:, 1]

        return grad



class Exp(Function):
    r"""Exponential function with weighted input

    .. math::
        f(x) = e^{w^T x}

    """
    def __init__(self, weights=[1.], name='Exp'):
        super().__init__()
        self.name = name
        self.weights = np.array(weights)
        self.setDimDom(dimension=len(weights))
        self.outdim = 1

    def __call__(self, x):
        #: Ensure x is of the right dimensionality
        self.checkDim(x)


        yy = np.exp(np.dot(x, self.weights))

        return yy.reshape(-1,1)


    def grad(self, x):

        return np.dot(self.__call__(x), self.weights.reshape(1, self.dim))[:, np.newaxis, :]



class Log(Function):
    r"""Logarithm function with weighted input

    .. math::
        f(x) = \log|w^T x|
    """
    def __init__(self, weights=[1.], name='Log'):
        super().__init__()
        self.name = name
        self.weights = np.array(weights)
        self.setDimDom(dimension=len(weights))
        self.outdim = 1

    def __call__(self, x):
        #: Ensure x is of the right dimensionality
        self.checkDim(x)


        yy = np.log(np.abs(np.dot(x, self.weights)))

        return yy.reshape(-1,1)


    def grad(self, x):

        return np.dot((1./np.dot(x, self.weights)).reshape(-1, 1), self.weights.reshape(1, self.dim))[:, np.newaxis, :]
