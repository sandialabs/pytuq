#!/usr/bin/env python

import numpy as np

from .func import Function


class Constant(Function):
    """Multioutput Constant function [REF]

    """
    def __init__(self, dim, const, name='Constant'):
        super().__init__()
        self.setDimDom(dimension=dim)
        self.name = name
        self.const = const
        self.outdim = const.shape[0]

        return

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2dim
            Nxd array of N points in d=1 dimensions

        Returns
        -------
        numpy array, 1dim
            Vector of N values
        """

        self.checkDim(x)

        nx = x.shape[0]
        yy = np.tile(self.const, (nx, 1))

        return yy


    def grad(self, x):
        return np.zeros((x.shape[0], self.outdim, self.dim))


class Identity(Function):
    """Identity

    """
    def __init__(self, dim, name='Identity'):
        super().__init__()
        self.setDimDom(dimension=dim)
        self.name = name
        self.outdim = dim

        return

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2dim
            Nxd array of N points in d=1 dimensions

        Returns
        -------
        numpy array, 1dim
            Vector of N values
        """

        self.checkDim(x) # TODO: MAKE THIS A DECORATOR?

        return x


    def grad(self, x):
        nx = x.shape[0]
        grad = np.tile(np.eye(self.dim), (nx, 1, 1))
        return grad


class Quad(Function):
    def __init__(self, name='Quad'):
        super().__init__()
        self.setDimDom(dimension=1)
        self.name = name
        self.outdim = 1

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2d
            Nxd array of N points in d dimensions

        Returns
        -------
        numpy array, 1d
            Vector of N values
        """

        #: Ensure x is of the right dimensionality
        self.checkDim(x)


        yy = 3. + x[:, 0] - x[:, 0]**2

        return yy.reshape(-1,1)

    def grad(self, x):
        return (1. - 2.* x[:,0])[:, np.newaxis, np.newaxis]


class Quad2d(Function):
    def __init__(self, name='Quad2d'):
        super().__init__()
        self.setDimDom(dimension=2)
        self.name = name
        self.outdim = 1

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2d
            Nxd array of N points in d dimensions

        Returns
        -------
        numpy array, 1d
            Vector of N values
        """

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
    def __init__(self, weights=[1.], name='Exp'):
        super().__init__()
        self.name = name
        self.weights = np.array(weights)
        self.setDimDom(dimension=len(weights))
        self.outdim = 1

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2d
            Nxd array of N points in d dimensions

        Returns
        -------
        numpy array, 1d
            Vector of N values
        """

        #: Ensure x is of the right dimensionality
        self.checkDim(x)


        yy = np.exp(np.dot(x, self.weights))

        return yy.reshape(-1,1)


    def grad(self, x):

        return np.dot(self.__call__(x), self.weights.reshape(1, self.dim))[:, np.newaxis, :]



class Log(Function):
    def __init__(self, weights=[1.], name='Log'):
        super().__init__()
        self.name = name
        self.weights = np.array(weights)
        self.setDimDom(dimension=len(weights))
        self.outdim = 1

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2d
            Nxd array of N points in d dimensions

        Returns
        -------
        numpy array, 1d
            Vector of N values
        """

        #: Ensure x is of the right dimensionality
        self.checkDim(x)


        yy = np.log(np.abs(np.dot(x, self.weights)))

        return yy.reshape(-1,1)


    def grad(self, x):

        return np.dot((1./np.dot(x, self.weights)).reshape(-1, 1), self.weights.reshape(1, self.dim))[:, np.newaxis, :]
