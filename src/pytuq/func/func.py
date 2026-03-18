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
        dim (int): Input dimensionality, `d`.
        dmax (float): Default domain half-size. That is, default domain is :math:`[-d_{max}, + d_{max}]`.
        domain (np.ndarray): A 2d array of size :math:`(d,2)` indicating the input domain of the function.
        eval (callable): Callable evaluator of the function.
        name (str): Name of the function.
        outdim (int): Output dimensionality, `o`.
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

    def call1d(self, odim=0):
        r"""Wrapper function for single output evaluation.

        Args:
            odim (int, optional): Output dimension index. Defaults to 0.

        Returns:
            callable: A callable function from :math:`(N,d)` to :math:`(N,)`.
        """
        return lambda x: self.__call__(x)[:, odim]

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
            dim (int): Dimensionality, `d`.

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
        r"""Set input dimensionality and domain.

        Exactly one of ``domain`` or ``dimension`` must be provided.
        If ``dimension`` is given, the domain defaults to
        :math:`[-d_{max}, +d_{max}]^d`.

        Args:
            domain (np.ndarray, optional): A 2d array of size :math:`(d,2)` specifying the domain.
            dimension (int, optional): Input dimensionality `d`.

        Raises:
            AssertionError: If both or neither of ``domain`` and ``dimension`` are provided.
        """
        if domain is None:
            assert(dimension is not None)
            dim, dom = dimension, np.tile(np.array([-self.dmax, self.dmax]), (dimension, 1))
        else:
            assert(dimension is None)
            dim, dom = domain.shape[0], domain

        self._setDim(dim)
        self._setDomain(dom)


    def inDomain(self, x):
        r"""Check whether all points lie inside the function domain.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            bool: True if all rows of ``x`` are within the domain.
        """
        assert(x.shape[1]==self.dim)

        return np.array([(x[:, i] >= self.domain[i, 0]) *
                         (x[:, i] <= self.domain[i, 1])
                         for i in range(self.dim)]).all()

    def checkDomain(self, x):
        r"""Raise an error if any point lies outside the function domain.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Raises:
            ValueError: If any row of ``x`` is outside the domain.
        """
        if not self.inDomain(x):
            raise ValueError("Function input is outside the domain. Exiting.")


    def checkDim(self, x):
        r"""Assert that input has the correct shape.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Raises:
            AssertionError: If ``x`` is not 2d or the second dimension does not match ``dim``.
        """
        assert(len(x.shape)==2)
        assert(x.shape[1]==self.dim)


    def sample_uniform(self, nsam):
        r"""Draw uniform random samples from the function domain.

        Args:
            nsam (int): Number of samples.

        Returns:
            np.ndarray: A 2d array of size :math:`(N,d)` with samples.
        """
        rsams = np.random.rand(nsam, self.dim)
        sams = scale01ToDom(rsams, self.domain)

        return sams


    def plot_1d(self, nom=None, ngr=133):
        """Plot 1d slices of the function for all input-output dimension pairs.

        Args:
            nom (np.ndarray, optional): Nominal input values for non-plotted dimensions.
                Defaults to None (midpoint of the domain).
            ngr (int, optional): Number of grid points. Defaults to 133.
        """
        for odim in range(self.outdim):
            for idim in range(self.dim):
                plot_1d(self.__call__, self.domain, idim=idim, odim=odim,
                            nom=nom, ngr=ngr, figname=self.name+f'_o{odim}_d{idim}.png')
                plt.clf()

        return


    def plot_2d(self, nom=None, ngr=33):
        """Plot 2d slices of the function for all input dimension pairs.

        Args:
            nom (np.ndarray, optional): Nominal input values for non-plotted dimensions.
                Defaults to None (midpoint of the domain).
            ngr (int, optional): Number of grid points per dimension. Defaults to 33.
        """
        for odim in range(self.outdim):
            for idim in range(self.dim):
                for jdim in range(idim+1, self.dim):
                    plot_2d(self.__call__, self.domain, idim=idim, jdim=jdim, odim=odim,
                            nom=nom, ngr=ngr, figname=self.name+f'_o{odim}_d{idim}_d{jdim}.png')
                    plt.clf()

        return


    def grad_(self, x, eps=1.e-5):
        r"""Numerical gradient via central finite differences.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.
            eps (float, optional): Finite-difference step size. Defaults to 1e-5.

        Returns:
            np.ndarray: Gradient 3d array of size :math:`(N,o,d)`.
        """
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        for idim in range(self.dim):
            xx2 = x.copy()
            xx2[:, idim]+=eps
            xx1 = x.copy()
            xx1[:, idim]-=eps
            grad[:, :, idim] = (self.__call__(xx2) - self.__call__(xx1))/(2.*eps)

        return grad

    def hess_(self, x, eps=1.e-5):
        r"""Numerical Hessian via central finite differences.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.
            eps (float, optional): Finite-difference step size. Defaults to 1e-5.

        Returns:
            np.ndarray: Hessian 4d array of size :math:`(N,o,d,d)`.
        """
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
        """Minimize the function using L-BFGS-B.

        Args:
            odim (int, optional): Output dimension index to minimize. Defaults to 0.
            return_res (bool, optional): If True, return the full optimization result.
                If False, return only the optimal input. Defaults to False.

        Returns:
            np.ndarray or scipy.optimize.OptimizeResult: Optimal input array,
                or the full result object if ``return_res`` is True.
        """
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
        r"""Evaluate the function on a lower-dimensional slice.

        Non-sliced dimensions are fixed at their nominal values.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N, |ind|)` with values
                for the selected dimensions.
            ind (list of int, optional): Indices of the active input dimensions.
                Defaults to [0].
            nom (np.ndarray, optional): Full nominal input of length `d`.
                Defaults to None (midpoint of the domain).

        Returns:
            np.ndarray: Output 2d array of size :math:`(N,o)`.
        """
        nx = x.shape[0]
        assert(x.shape[1]==len(ind))
        if nom is None:
            nom = np.mean(self.domain, axis=1)
        assert(nom.shape[0]==self.dim) # full nominal, even if some entries won't be used

        xg = np.tile(nom, (nx, 1))
        xg[:, ind] = x

        return self.__call__(xg)

    def evalgrad_slice(self, x, ind=[0], nom=None):
        r"""Evaluate the gradient on a lower-dimensional slice.

        Non-sliced dimensions are fixed at their nominal values.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N, |ind|)` with values
                for the selected dimensions.
            ind (list of int, optional): Indices of the active input dimensions.
                Defaults to [0].
            nom (np.ndarray, optional): Full nominal input of length `d`.
                Defaults to None (midpoint of the domain).

        Returns:
            np.ndarray: Gradient 3d array of size :math:`(N, o, |ind|)`.
        """
        nx = x.shape[0]
        assert(x.shape[1]==len(ind))
        if nom is None:
            nom = np.mean(self.domain, axis=1)
        assert(nom.shape[0]==self.dim) # full nominal, even if some entries won't be used

        xg = np.tile(nom, (nx, 1))
        xg[:, ind] = x

        return self.grad(xg)[:, :, ind]



class AddFcn(Function):
    """Sum of two functions.

    Attributes:
        fcn1 (Function): First operand.
        fcn2 (Function): Second operand.
    """

    def __init__(self, fcn1, fcn2, name='Sum'):
        """Initialization.

        Args:
            fcn1 (Function): First function.
            fcn2 (Function): Second function.
            name (str, optional): Name. Defaults to 'Sum'.
        """
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
        r"""Evaluate the sum.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 2d array of size :math:`(N,o)`.
        """
        return self.fcn1(x) + self.fcn2(x)

    def grad(self, x):
        r"""Evaluate the gradient of the sum.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Gradient 3d array of size :math:`(N,o,d)`.
        """
        return self.fcn1.grad(x) + self.fcn2.grad(x)



class SubFcn(Function):
    """Difference of two functions.

    Attributes:
        fcn1 (Function): First operand.
        fcn2 (Function): Second operand.
    """

    def __init__(self, fcn1, fcn2, name='Subtraction'):
        """Initialization.

        Args:
            fcn1 (Function): First function.
            fcn2 (Function): Second function.
            name (str, optional): Name. Defaults to 'Subtraction'.
        """
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
        r"""Evaluate the difference.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 2d array of size :math:`(N,o)`.
        """
        return self.fcn1(x) - self.fcn2(x)

    def grad(self, x):
        r"""Evaluate the gradient of the difference.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Gradient 3d array of size :math:`(N,o,d)`.
        """
        return self.fcn1.grad(x) - self.fcn2.grad(x)




class MultFcn(Function):
    """Element-wise product of two functions.

    Attributes:
        fcn1 (Function): First operand.
        fcn2 (Function): Second operand.
    """

    def __init__(self, fcn1, fcn2, name='Product'):
        """Initialization.

        Args:
            fcn1 (Function): First function.
            fcn2 (Function): Second function.
            name (str, optional): Name. Defaults to 'Product'.
        """
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
        r"""Evaluate the product.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 2d array of size :math:`(N,o)`.
        """
        return self.fcn1(x) * self.fcn2(x)

    def grad(self, x):
        r"""Evaluate the gradient of the product via the product rule.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Gradient 3d array of size :math:`(N,o,d)`.
        """
        return self.fcn1.grad(x) * self.fcn2(x)[:, :, np.newaxis] + self.fcn2.grad(x) * self.fcn1(x)[:, :, np.newaxis]


class DivFcn(Function):
    """Element-wise quotient of two functions.

    Attributes:
        fcn1 (Function): Numerator function.
        fcn2 (Function): Denominator function.
    """

    def __init__(self, fcn1, fcn2, name='Quotient'):
        """Initialization.

        Args:
            fcn1 (Function): Numerator function.
            fcn2 (Function): Denominator function.
            name (str, optional): Name. Defaults to 'Quotient'.
        """
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
        r"""Evaluate the quotient.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 2d array of size :math:`(N,o)`.
        """
        return self.fcn1(x) / self.fcn2(x)

    def grad(self, x):
        r"""Evaluate the gradient of the quotient via the quotient rule.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Gradient 3d array of size :math:`(N,o,d)`.
        """
        return ( self.fcn1.grad(x) * self.fcn2(x)[:, :, np.newaxis] - self.fcn2.grad(x) * self.fcn1(x)[:, :, np.newaxis] ) / self.fcn2(x)[:, :, np.newaxis]**2

class PowFcn(Function):
    """Element-wise power of a function.

    Attributes:
        fcn (Function): Base function.
        power (float): Exponent.
    """

    def __init__(self, fcn, power, name='Power'):
        """Initialization.

        Args:
            fcn (Function): Base function.
            power (float): Exponent value.
            name (str, optional): Name. Defaults to 'Power'.
        """
        super().__init__()
        self.fcn = fcn
        self.name = name
        self.power = power
        self.setDimDom(domain=fcn.domain)
        self.outdim = fcn.outdim

        return

    def __call__(self, x):
        r"""Evaluate the power function.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 2d array of size :math:`(N,o)`.
        """
        return np.power(self.fcn(x), self.power)

    def grad(self, x):
        r"""Evaluate the gradient via the power rule.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Gradient 3d array of size :math:`(N,o,d)`.
        """
        return self.power * self.fcn.grad(x) * np.power(self.fcn(x), self.power-1)[:, :, np.newaxis]


class ModelWrapperFcn(Function):
    """Wrapper that turns a generic callable into a Function.

    Wraps a model callable (with optional parameters) so it can be
    used wherever a :class:`Function` is expected.

    Attributes:
        model (callable): The underlying model callable.
        modelpar: Optional model parameters passed as a second argument.
        ndim (int): Input dimensionality.
    """

    def __init__(self, model, ndim, modelpar=None, name='ModelWrapper'):
        r"""Initialization.

        Args:
            model (callable): A callable that accepts an :math:`(N,d)` array
                (and optionally model parameters) and returns an array of
                length `N`.
            ndim (int): Input dimensionality, `d`.
            modelpar (optional): Extra parameter object passed to ``model``
                as a second argument. Defaults to None.
            name (str, optional): Name. Defaults to 'ModelWrapper'.
        """
        super().__init__(name=name)
        self.model = model
        self.modelpar = modelpar
        self.ndim = ndim
        self.setDimDom(dimension=ndim)
        self.outdim = 1

    def __call__(self, x):
        r"""Evaluate the wrapped model.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`.

        Returns:
            np.ndarray: Output 2d array of size :math:`(N,1)`.
        """
        self.checkDim(x)
        if self.modelpar is None:
            return self.model(x).reshape(-1,1)
        else:
            return self.model(x, self.modelpar).reshape(-1,1)
