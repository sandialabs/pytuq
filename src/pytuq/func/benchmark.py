#!/usr/bin/env python

import sys
import numpy as np
from scipy.stats import multivariate_normal

from .func import Function


class Franke(Function):
    def __init__(self, name='Franke'):
        super().__init__()
        self.name = name
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain = np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):


        tt1 = 0.75*np.exp(-((9*x[:,0] - 2)**2 + (9*x[:,1] - 2)**2)/4.)
        tt2 = 0.75*np.exp(-((9*x[:,0] + 1)**2)/49 - (9*x[:,1] + 1)/10.)
        tt3 = 0.5*np.exp(-((9*x[:,0] - 7)**2 + (9*x[:,1] - 3)**2)/4.)
        tt4 = -0.2*np.exp(-(9*x[:,0] - 4)**2 - (9*x[:,1] - 7)**2)



        return (tt1 + tt2 + tt3 + tt4).reshape(-1,self.outdim)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        tt1 = 0.75*np.exp(-((9*x[:,0] - 2)**2 + (9*x[:,1] - 2)**2)/4.)
        tt2 = 0.75*np.exp(-((9*x[:,0] + 1)**2)/49 - (9*x[:,1] + 1)/10.)
        tt3 = 0.5*np.exp(-((9*x[:,0] - 7)**2 + (9*x[:,1] - 3)**2)/4.)
        tt4 = -0.2*np.exp(-(9*x[:,0] - 4)**2 - (9*x[:,1] - 7)**2)

        grad[:,0,0] = -2*(9*x[:,0] - 2)*9/4 * tt1 - 2*(9*x[:,0] + 1)*9/49 * tt2 + \
              -2*(9*x[:,0] - 7)*9/4 * tt3 - 2*(9*x[:,0] - 4)*9 * tt4
        grad[:,0,1] = -2*(9*x[:,1] - 2)*9/4 * tt1 - 9./10. * tt2 + \
              -2*(9*x[:,1] - 3)*9/4 * tt3 - 2*(9*x[:,1] - 7)*9 * tt4

        return grad

class Sobol(Function):
    # from https://www.sfu.ca/~ssurjano/gfunc.html
    def __init__(self, name='Sobol', dim=5):
        super().__init__()
        self.name = name
        self.dim = dim
        self.outdim = 1

        self.setDimDom(domain = np.ones((self.dim, 1)) * np.array([0., 1.]))
        self.a = np.array([(i-2.)/2. for i in range(1,self.dim+1)])


    def __call__(self, x):
        sam = x.shape[0]
        self.checkDim(x)

        ydata=np.empty((sam, self.outdim))
        for j in range(sam):
            val=1.
            for k in range(self.dim):
                val *= ( (abs(4.*x[j,k]-2.)+self.a[k])/(1.+self.a[k]) )
            ydata[j,0]=val

        return ydata



class Ishigami(Function):
    # from https://www.sfu.ca/~ssurjano/ishigami.html
    def __init__(self, name='Ishigami'):
        super().__init__()
        self.name = name
        self.dim = 3
        self.outdim = 1


        self.setDimDom(np.ones((self.dim, 1)) * np.array([-np.pi, np.pi]))
        self.a = 7
        self.b = 0.1


    def __call__(self, x):
        sam = x.shape[0]
        self.checkDim(x)

        ydata=np.empty((sam, self.outdim))

        for j in range(sam):
            ydata[j, 0]=np.sin(x[j,0])+self.a*np.sin(x[j,1])**2+self.b*np.sin(x[j,0])*x[j,2]**4

        return ydata



class NegAlpineN2(Function):
    """Negative Alpine function [http://benchmarkfcns.xyz/benchmarkfcns/alpinen2fcn.html]

    """
    def __init__(self, name='Alpine N2', dim=2):
        super().__init__()
        self.setDimDom(domain=np.array(np.tile([0.001, 10.], (dim, 1))))
        self.name = name
        self.outdim = 1

        return

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2dim
            Nxd array of N points in d dimensions

        Returns
        -------
        numpy array, 1dim
            Vector of N values
        """

        self.checkDim(x)

        yy = - np.sqrt(np.prod(x, axis=1)) * np.prod(np.sin(x), axis=1)

        return yy.reshape(-1, 1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        for j in range(self.dim):
            grad[:, 0, j] = (0.5 / x[:, j] + 1./np.tan(x[:, j])) * self.__call__(x)[:,0]

        return grad


class Adjiman(Function):
    """Adjiman function [http://benchmarkfcns.xyz/benchmarkfcns/adjimanfcn.html]

    """
    def __init__(self, name='Adjiman'):
        super().__init__()
        self.setDimDom(domain=np.array([[-1., 2.], [-1., 1.]]))
        self.name = name
        self.outdim = 1

        return

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2dim
            Nxd array of N points in d=2 dimensions

        Returns
        -------
        numpy array, 1dim
            Vector of N values
        """

        self.checkDim(x)

        yy = np.cos(x[:, 0]) * np.sin(x[:, 1]) - \
            x[:, 0] / (1.0 + x[:, 1] * x[:, 1])

        return yy.reshape(-1,1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        grad[:, 0, 0] = - np.sin(x[:, 0]) * np.sin(x[:, 1]) - \
            1. / (1.0 + x[:, 1] * x[:, 1])
        grad[:, 0, 1] = np.cos(x[:, 0]) * np.cos(x[:, 1]) + \
            2. * x[:, 0] * x[:, 1] / (1.0 + x[:, 1] * x[:, 1])**2

        return grad


class Branin(Function):
    """Branin function [https://www.sfu.ca/~ssurjano/branin.html]

    """
    def __init__(self, name='Branin'):
        super().__init__()
        self.setDimDom(domain=np.array([[-5., 10.], [0., 15.]]))
        self.name = name
        self.outdim = 1

        self.a_ = 1.
        self.b_ = 5.1/(4*np.pi**2)
        self.c_ = 5./np.pi
        self.r_ = 6.
        self.s_ = 10.
        self.t_ = 1./(8.*np.pi)

        return

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2dim
            Nxd array of N points in d=2 dimensions

        Returns
        -------
        numpy array, 1dim
            Vector of N values
        """

        self.checkDim(x)

        yy = self.a_ * (x[:, 1] - self.b_ * x[:, 0]**2 + self.c_ * x[:, 0] - self.r_)**2
        yy += self.s_*(1.-self.t_)*np.cos(x[:, 0])
        yy += self.s_

        return yy.reshape(-1,1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = 2.0 * self.a_ * \
            (x[:, 1] - self.b_ * x[:, 0]**2 + self.c_ * x[:, 0] - self.r_) * \
            (-2.0 * self.b_ * x[:, 0] + self.c_) - \
            self.s_ * (1. - self.t_) * np.sin(x[:, 0])
        grad[:, 0, 1] = 2.0 * self.a_ * \
            (x[:, 1] - self.b_ * x[:, 0]**2 + self.c_ * x[:, 0] - self.r_)


        return grad



class SumSquares(Function):
    """SumSquares function [http://benchmarkfcns.xyz/benchmarkfcns/sumsquaresfcn.html]

    """
    def __init__(self, name='SumSquares', dim=5):
        super().__init__()
        self.setDimDom(domain=np.array(np.tile([-10.0, 10.0], (dim, 1))))
        self.name = name
        self.outdim = 1

        return

    def __call__(self, x):
        """Function call.

        Parameters
        ----------
        x : numpy array, 2dim
            Nxd array of N points in d dimensions

        Returns
        -------
        numpy array, 1dim
            Vector of N values
        """

        self.checkDim(x)

        yy = np.zeros((x.shape[0], self.outdim))
        for j in range(self.dim):
            yy[:,0] += (1. + j) * x[:, j]**2

        return yy.reshape(-1,1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        for j in range(self.dim):
            grad[:, 0, j] = 2.0 * (1. + j) * x[:, j]

        return grad


class Quadratic(Function):
    """MVN function [REF]

    """
    def __init__(self, center, hess, name='Quadratic'):
        super().__init__()

        self.center = np.array(center, dtype=float)
        self.hess = np.array(hess, dtype=float)

        # TODO: use Hessian to inform the domain
        domain = np.tile(self.center.reshape(-1,1), (1,2))
        domain[:,0] -= self.dmax*np.ones_like(self.center)
        domain[:,1] += self.dmax*np.ones_like(self.center)

        self.setDimDom(domain=domain)
        self.name = name
        self.outdim = 1

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

        nsam = x.shape[0]
        yy = np.empty(nsam,)
        for i in range(nsam):
            yy[i] = 0.5 * np.dot(x[i, :]-self.center, np.dot(self.hess, x[i, :]-self.center))


        return yy.reshape(-1,1)


class MVN(Function):
    """MVN function [REF]

    """
    def __init__(self, mean, cov, name='MVN'):
        super().__init__()

        self.mean = np.array(mean, dtype=float)
        self.cov = np.array(cov, dtype=float)

        # TODO: use cov to inform the domain
        domain = np.tile(self.mean.reshape(-1,1), (1,2))
        domain[:,0] -= self.dmax*np.ones_like(self.mean)
        domain[:,1] += self.dmax*np.ones_like(self.mean)

        self.setDimDom(domain=domain)
        self.name = name

        self.outdim = 1

        return

    def __call__(self, x):
        self.checkDim(x)

        yy = multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

        return yy.reshape(-1,1)




class TFData(Function):
    """Data generating model inspired by https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb#scrollTo=5zCEYpzu7bDX.
    """

    def __init__(self, name='tfdata'):
        super().__init__()
        self.name = name

        self.dim = 1
        self.outdim = 1

        self.w0 = 0.125
        self.b0 = 5.
        self.a = -20.
        self.b = 60.

        self.setDimDom(domain=np.array([[self.a, self.b]]))

        return

    def __call__(self, x):

        y = (self.w0 * x * (1. + np.sin(x)) + self.b0)

        return y
