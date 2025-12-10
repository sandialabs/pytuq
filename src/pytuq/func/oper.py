#!/usr/bin/env python


import sys
import numpy as np

from ..utils.stats import intersect_domain
from .func import Function

class CartesProdFcn(Function):
    """Cartesian product of two functions.
    
    Computes the element-wise product of two functions evaluated on different
    subspaces of the input domain.
    
    Args:
        fcn1: First function to multiply.
        fcn2: Second function to multiply.
        name: Name of the composite function. Defaults to 'CartesProduct'.
    
    Attributes:
        fcn1: The first function in the product.
        fcn2: The second function in the product.
        domain: Combined domain from both functions.
        outdim: Output dimension (must match for both functions).
    """
    def __init__(self, fcn1, fcn2, name='CartesProduct'):
        super().__init__()
        self.fcn1 = fcn1
        self.fcn2 = fcn2

        self.name = name
        domain = np.vstack((self.fcn1.domain, self.fcn2.domain))
        assert(domain is not None)
        self.setDimDom(domain=domain)
        self.outdim = self.fcn1.outdim
        assert(self.outdim ==  self.fcn2.outdim)

        return

    def __call__(self, x):
        dim1 = self.fcn1.dim
        return self.fcn1(x[:, :dim1]) * self.fcn2(x[:, dim1:])

    def grad(self, x):
        dim1 = self.fcn1.dim
        g1 = self.fcn1.grad(x[:, :dim1]) * self.fcn2(x[:, dim1:])[:, :, np.newaxis]
        g2 = self.fcn2.grad(x[:, dim1:]) * self.fcn1(x[:, :dim1])[:, :, np.newaxis]

        return np.concatenate((g1, g2), axis=2)

class GradFcn(Function):
    """Gradient extraction function.
    
    Extracts the gradient with respect to a specific input dimension from a function.
    
    Args:
        fcn: The function to extract gradient from.
        idim: Index of the input dimension for gradient extraction.
        name: Name of the gradient function. Defaults to 'GradFcn'.
    
    Attributes:
        fcn: The underlying function.
        idim: Input dimension index for gradient extraction.
        outdim: Output dimension (inherited from fcn).
        domain: Domain of the function (inherited from fcn).
    """
    def __init__(self, fcn, idim, name='GradFcn'):
        super().__init__()
        self.fcn = fcn
        self.idim = idim
        self.outdim = self.fcn.outdim

        self.name = name
        if hasattr(self.fcn, 'name'):
            self.name += f'_{self.fcn.name}'
        domain = self.fcn.domain.copy()
        self.setDimDom(domain=domain)

        return

    def __call__(self, x):
        full_grad = self.fcn.grad(x)
        return full_grad[:, :, self.idim]


class ComposeFcn(Function):
    """Function composition.
    
    Composes two functions such that the output is fcn2(fcn1(x)).
    The output dimension of fcn1 must match the input dimension of fcn2.
    
    Args:
        fcn1: The inner function to be evaluated first.
        fcn2: The outer function to be evaluated on fcn1's output.
        name: Name of the composite function. Defaults to 'Composite'.
    
    Attributes:
        fcn1: The inner function.
        fcn2: The outer function.
        outdim: Output dimension (inherited from fcn2).
        domain: Domain of the composite function (inherited from fcn1).
    """
    def __init__(self, fcn1, fcn2, name='Composite'):
        super().__init__()
        self.fcn1 = fcn1
        self.fcn2 = fcn2
        assert(self.fcn2.dim==self.fcn1.outdim)

        self.outdim = self.fcn2.outdim

        self.name = name

        self.setDimDom(domain=self.fcn1.domain)

        return

    def __call__(self, x):
        return self.fcn2(self.fcn1(x))

    def grad(self, x):

        grad = np.empty((x.shape[0], self.outdim, self.dim))
        for i in range(x.shape[0]):
            grad[i, :, :] = np.dot(self.fcn2.grad(self.fcn1(x))[i], self.fcn1.grad(x)[i])
        return grad

class SliceFcn(Function):
    """Sliced function evaluation.
    
    Evaluates a function on a subset of input dimensions while fixing others at
    nominal values.
    
    Args:
        fcn: The function to slice.
        name: Name of the sliced function. Defaults to 'Slice'.
        ind: List of input dimension indices to keep active. Defaults to [0].
        nom: Nominal values for fixed dimensions. Defaults to None.
    
    Attributes:
        fcn: The underlying function.
        ind: Active dimension indices.
        nom: Nominal values for fixed dimensions.
        domain: Sliced domain containing only active dimensions.
        outdim: Output dimension (inherited from fcn).
    """
    def __init__(self, fcn, name='Slice', ind=[0], nom=None):
        super().__init__()
        self.fcn = fcn
        self.name = name
        self.name += (' ' + self.fcn.name)
        for ii in ind:
            self.name += (' ' + str(ii))
        self.nom = nom
        self.ind = ind
        self.setDimDom(domain=self.fcn.domain[ind, :])
        self.outdim = self.fcn.outdim


    def __call__(self, x):
        return self.fcn.eval_slice(x, ind=self.ind, nom=self.nom)

    def grad(self, x):
        return self.fcn.evalgrad_slice(x, ind=self.ind, nom=self.nom)



class ShiftFcn(Function):
    """Shifted function.
    
    Applies a spatial shift to the input domain of a function, evaluating
    fcn(x - shift).
    
    Args:
        fcn: The function to shift.
        shift: Shift vector to subtract from input.
        domain: New domain for the shifted function. If None, automatically
            computed from fcn's domain. Defaults to None.
        name: Name of the shifted function. Defaults to 'Shift'.
    
    Attributes:
        fcn: The underlying function.
        shift: The shift vector.
        domain: Domain of the shifted function.
        outdim: Output dimension (inherited from fcn).
    """
    def __init__(self, fcn, shift, domain=None,name='Shift'):
        super().__init__()
        assert(fcn.dim==len(shift))
        self.fcn = fcn
        self.shift = np.array(shift)
        self.name = name
        if domain is None:
            domain = self.fcn.domain-self.shift.reshape(-1,1)

        self.setDimDom(domain=domain)
        self.outdim = self.fcn.outdim

    def __call__(self, x):
        return self.fcn(x-self.shift)

    def grad(self, x):
        return self.fcn.grad(x-self.shift)

class LinTransformFcn(Function):
    """Linear transformation of function output.
    
    Applies a linear transformation to the function output: scale * fcn(x) + shift.
    
    Args:
        fcn: The function to transform.
        scale: Scaling factor for the output.
        shift: Shift/offset to add to the scaled output.
        name: Name of the transformed function. Defaults to 'LinTransform'.
    
    Attributes:
        fcn: The underlying function.
        scale: Output scaling factor.
        shift: Output shift/offset.
        domain: Domain of the function (inherited from fcn).
        outdim: Output dimension (inherited from fcn).
    """
    def __init__(self, fcn, scale, shift, name='LinTransform'):
        super().__init__()
        self.fcn = fcn
        self.scale = scale
        self.shift = shift
        self.name = name
        self.setDimDom(domain=self.fcn.domain)
        self.outdim = self.fcn.outdim

        return

    def __call__(self, x):
        return self.fcn(x)*self.scale+self.shift

    def grad(self, x):
        return self.fcn.grad(x)*self.scale


class PickDim(Function):
    """Dimension picking function.
    
    Selects a single dimension from the input vector and optionally scales it.
    Returns cf * x[pdim] as a scalar output.
    
    Args:
        dim: Total number of input dimensions.
        pdim: Index of the dimension to pick.
        cf: Scaling coefficient for the picked dimension. Defaults to 1.0.
        name: Name of the function. Defaults to 'Dimension-Pick'.
    
    Attributes:
        cf: Scaling coefficient.
        pdim: Index of picked dimension.
        dim: Total input dimensions.
        outdim: Output dimension (always 1).
    """
    def __init__(self, dim, pdim, cf=1.0, name='Dimension-Pick'):
        super().__init__()
        self.cf = cf
        self.setDimDom(dimension=dim)
        self.name = name
        self.pdim = pdim
        self.outdim = 1

    def __call__(self, x):

        self.checkDim(x)

        yy = self.cf*x[:, self.pdim]

        return yy.reshape(-1,1)


    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        nx = x.shape[0]

        grad[:, 0, self.pdim] = self.cf * np.ones((nx, ))
        return grad
