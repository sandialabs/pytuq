#!/usr/bin/env python
"""Module for a general multioutput function class and some basic operations."""

import sys
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from ..utils.maps import scale01ToDom
from ..utils.stats import intersect_domain
from ..utils.plotting import plot_1d, plot_2d


class Function():
    r"""Base class for a function.

    Attributes:
        dim (int): Input dimensionality, :math:`d`.
        dmax (float): Default domain half-size. That is, default domain is :math:`[-d_{max}, + d_{max}]`.
        domain (np.ndarray): A 2d array of size :math:`(d,2)` indicating the input domain of the function.
        eval (callable): Callable evaluator of the function.
        name (str): Name of the function.
        outdim (int): Output dimensionality, :math:`o`.
    """

    def __init__(self, name='Base'):
        """Initialization.

        Args:
            name (str, optional): Function name. Defaults to "Base".
        """
        super().__init__()
        self.dim = None
        self.outdim = None
        self.domain = None
        self.name = name
        self.eval = None
        self.dmax = 10.0


    def __call__(self, x):
        r"""Call function.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 2d array of size :math:`(N,o)`.

        Raises:
            NotImplementedError: If eval function is not implemented.
        """
        if self.eval is None:
            raise NotImplementedError("Function evaluation is not implemented in the base class.")
        else:
            return self.eval(x)

    def grad(self, x):
        """Gradient evaluator.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 3d array of size :math:`(N,o,d)`.
        """
        print("Analytical gradient not implemented. Fallback to numerical.")
        return self.grad_(x)

    def hess(self, x):
        """Hessian evaluator.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 4d array of size :math:`(N,o,d,d)`.
        """
        print("Analytical hessian not implemented. Fallback to numerical.")
        return self.hess_(x)

    def setCall(self, eval):
        r"""Setting the evaluator function.

        Args:
            eval (callable): Callable function from :math:`(N,d)` to :math:`(N,o)`.
        """
        self.eval = eval

    def __add__(self, other):
        """Adding operation.

        Args:
            other (Function): The Function to be added.

        Returns:
            Function: The resulting Function.
        """
        assert(isinstance(other, Function))
        return AddFcn(self, other)

    def __sub__(self, other):
        """Subtracting operation.

        Args:
            other (Function): The Function to be subtracted.

        Returns:
            Function: The resulting Function.
        """
        assert(isinstance(other, Function))
        return SubFcn(self, other)

    def __mul__(self, other):
        """Multiplying operation.

        Args:
            other (Function): The Function to be multiplied with.

        Returns:
            Function: The resulting Function.
        """
        assert(isinstance(other, Function))
        return MultFcn(self, other)

    def __truediv__(self, other):
        """Dividing operation.

        Args:
            other (Function): The Function to be divided on.

        Returns:
            Function: The resulting Function.
        """
        assert(isinstance(other, Function))
        return DivFcn(self, other)

    def __pow__(self, power):
        """Power operation.

        Args:
            power (float): Power value.

        Returns:
            Function: The resulting Function.
        """
        return PowFcn(self, power)

    def _setDim(self, dim):
        r"""Setting the input dimensionality of the function.

        Args:
            dim (int): Dimensionality, :math:`d`.

        Note:
            No need to use this function externally. Use :func:`setDimDom()` instead.
        """
        self.dim = dim


    def _setDomain(self, domain):
        r"""Setting the input domain of the function.

        Args:
            domain (np.ndarray): A 2d array of size :math:`(d,2)` indicating the domain of the function.

        Note:
            No need to use this function externally. Use :func:`setDimDom()` instead.
        """
        assert(domain.shape[1]==2)
        self.domain = domain


    def setDimDom(self, domain=None, dimension=None):

        if domain is None:
            assert(dimension is not None)
            dim, dom = dimension, np.tile(np.array([-self.dmax, self.dmax]), (dimension, 1))
        else:
            assert(dimension is None)
            dim, dom = domain.shape[0], domain

        self._setDim(dim)
        self._setDomain(dom)


    def inDomain(self, x):
        assert(x.shape[1]==self.dim)

        return np.array([(x[:, i] >= self.domain[i, 0]) *
                         (x[:, i] <= self.domain[i, 1])
                         for i in range(self.dim)]).all()

    def checkDomain(self, x):
        if not self.inDomain(x):
            raise ValueError("Function input is outside the domain. Exiting.")


    def checkDim(self, x):
        assert(len(x.shape)==2)
        assert(x.shape[1]==self.dim)


    def sample_uniform(self, nsam):

        rsams = np.random.rand(nsam, self.dim)
        sams = scale01ToDom(rsams, self.domain)

        return sams


    def plot_1d(self, nom=None, ngr=133):

        for odim in range(self.outdim):
            for idim in range(self.dim):
                plot_1d(self.__call__, self.domain, idim=idim, odim=odim,
                            nom=nom, ngr=ngr, figname=self.name+f'_o{odim}_d{idim}.png')
                plt.clf()

        return


    def plot_2d(self, nom=None, ngr=33):

        for odim in range(self.outdim):
            for idim in range(self.dim):
                for jdim in range(idim+1, self.dim):
                    plot_2d(self.__call__, self.domain, idim=idim, jdim=jdim, odim=odim,
                            nom=nom, ngr=ngr, figname=self.name+f'_o{odim}_d{idim}_d{jdim}.png')
                    plt.clf()

        return


    def grad_(self, x, eps=1.e-5):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        for idim in range(self.dim):
            xx2 = x.copy()
            xx2[:, idim]+=eps
            xx1 = x.copy()
            xx1[:, idim]-=eps
            grad[:, :, idim] = (self.__call__(xx2) - self.__call__(xx1))/(2.*eps)

        return grad

    def hess_(self, x, eps=1.e-5):
        hess = np.zeros((x.shape[0], self.outdim, self.dim, self.dim))

        for idim in range(self.dim):
            for jdim in range(idim, self.dim):
                xx4 = x.copy()
                xx4[:, idim]+=eps
                xx4[:, jdim]+=eps
                xx3 = x.copy()
                xx3[:, idim]+=eps
                xx3[:, jdim]-=eps
                xx2 = x.copy()
                xx2[:, idim]-=eps
                xx2[:, jdim]+=eps
                xx1 = x.copy()
                xx1[:, idim]-=eps
                xx1[:, jdim]-=eps
                hess[:, :, idim, jdim] = (self.__call__(xx4) + self.__call__(xx1)
                 - self.__call__(xx3) - self.__call__(xx2))/(4.*eps*eps)
                if jdim>idim:
                    hess[:, :, jdim, idim] = hess[:, :, idim, jdim]
        return hess

    def minimize(self, odim=0, return_res=False):
        # Function wrapper for a single evaluation
        def ff(x, fcn):
            return fcn(x.reshape(1, fcn.dim))[0, odim]
        # Gradient wrapper for a single evaluation
        def gg(x, fcn):
            return fcn.grad(x.reshape(1, fcn.dim))[0, odim]

        x0 = self.sample_uniform(1)[0]
        res = minimize(ff, x0, args=(self,), method='L-BFGS-B', jac=gg,
               tol=None, bounds=self.domain,
               callback=None, options=None)
        # print(res)
        if return_res:
            return res
        else:
            return res.x

    # def check_hess(self):
    #     res = self.minimize(return_res=True)
    #     print(res.hess_inv)
    #     hess_min = np.linalg.inv(res.hess_inv.matmat(np.eye(self.dim)))
    #     hess_ = self.hess_(res.x.reshape(1, -1), eps=1.e-7).reshape(-1,self.dim, self.dim)
    #     print(hess_min)
    #     print(hess_[0])
    #     return np.allclose(hess_min, hess_[0])

    def eval_slice(self, x, ind=[0], nom=None):
        nx = x.shape[0]
        assert(x.shape[1]==len(ind))
        if nom is None:
            nom = np.mean(self.domain, axis=1)
        assert(nom.shape[0]==self.dim) # full nominal, even if some entries won't be used

        xg = np.tile(nom, (nx, 1))
        xg[:, ind] = x

        return self.__call__(xg)

    def evalgrad_slice(self, x, ind=[0], nom=None):
        nx = x.shape[0]
        assert(x.shape[1]==len(ind))
        if nom is None:
            nom = np.mean(self.domain, axis=1)
        assert(nom.shape[0]==self.dim) # full nominal, even if some entries won't be used

        xg = np.tile(nom, (nx, 1))
        xg[:, ind] = x

        return self.grad(xg)[:, :, ind]



class AddFcn(Function):
    def __init__(self, fcn1, fcn2, name='Sum'):
        super().__init__()
        assert(fcn1.dim==fcn2.dim)
        assert(fcn1.outdim==fcn2.outdim)
        self.fcn1 = fcn1
        self.fcn2 = fcn2
        self.name = name
        domain = intersect_domain(fcn1.domain, fcn2.domain)
        self.setDimDom(domain=domain)
        self.outdim = fcn1.outdim

    def __call__(self, x):
        return self.fcn1(x) + self.fcn2(x)

    def grad(self, x):
        return self.fcn1.grad(x) + self.fcn2.grad(x)



class SubFcn(Function):
    def __init__(self, fcn1, fcn2, name='Subtraction'):
        super().__init__()
        assert(fcn1.dim==fcn2.dim)
        assert(fcn1.outdim==fcn2.outdim)

        self.fcn1 = fcn1
        self.fcn2 = fcn2
        self.name = name

        domain = intersect_domain(fcn1.domain, fcn2.domain)
        self.setDimDom(domain=domain)
        self.outdim = fcn1.outdim


    def __call__(self, x):
        return self.fcn1(x) - self.fcn2(x)

    def grad(self, x):
        return self.fcn1.grad(x) - self.fcn2.grad(x)




class MultFcn(Function):
    def __init__(self, fcn1, fcn2, name='Product'):
        super().__init__()
        assert(fcn1.dim==fcn2.dim)
        self.fcn1 = fcn1
        self.fcn2 = fcn2
        self.name = name
        domain = intersect_domain(fcn1.domain, fcn2.domain)
        self.setDimDom(domain=domain)
        self.outdim = fcn1.outdim
        assert(self.outdim == fcn2.outdim)

        return

    def __call__(self, x):
        return self.fcn1(x) * self.fcn2(x)

    def grad(self, x):
        return self.fcn1.grad(x) * self.fcn2(x)[:, :, np.newaxis] + self.fcn2.grad(x) * self.fcn1(x)[:, :, np.newaxis]


class DivFcn(Function):
    def __init__(self, fcn1, fcn2, name='Quotient'):
        super().__init__()
        assert(fcn1.dim==fcn2.dim)
        self.fcn1 = fcn1
        self.fcn2 = fcn2
        self.name = name
        domain = intersect_domain(fcn1.domain, fcn2.domain)
        self.setDimDom(domain=domain)
        self.outdim = fcn1.outdim
        assert(self.outdim == fcn2.outdim)

        return

    def __call__(self, x):
        return self.fcn1(x) / self.fcn2(x)

    def grad(self, x):
        return ( self.fcn1.grad(x) * self.fcn2(x)[:, :, np.newaxis] - self.fcn2.grad(x) * self.fcn1(x)[:, :, np.newaxis] ) / self.fcn2(x)[:, :, np.newaxis]**2

class PowFcn(Function):
    def __init__(self, fcn, power, name='Power'):
        super().__init__()
        self.fcn = fcn
        self.name = name
        self.power = power
        self.setDimDom(domain=fcn.domain)
        self.outdim = fcn.outdim

        return

    def __call__(self, x):
        return np.power(self.fcn(x), self.power)

    def grad(self, x):
        return self.power * self.fcn.grad(x) * np.power(self.fcn(x), self.power-1)[:, :, np.newaxis]


class ModelWrapperFcn(Function):
    def __init__(self, model, ndim, modelpar=None, name='ModelWrapper'):
        super().__init__(name=name)
        self.model = model
        self.modelpar = modelpar
        self.ndim = ndim
        self.setDimDom(dimension=ndim)
        self.outdim = 1

    def __call__(self, x):
        self.checkDim(x)
        if self.modelpar is None:
            return self.model(x).reshape(-1,1)
        else:
            return self.model(x, self.modelpar).reshape(-1,1)
