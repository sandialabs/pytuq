#!/usr/bin/env python


import sys
import numpy as np

from ..utils.stats import intersect_domain
from .func import Function

class CartesProdFcn(Function):
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
    """Picking dimension function [REF]

    """
    def __init__(self, dim, pdim, cf=1.0, name='Dimension-Pick'):
        super().__init__()
        self.cf = cf
        self.setDimDom(dimension=dim)
        self.name = name
        self.pdim = pdim
        self.outdim = 1

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

        yy = self.cf*x[:, self.pdim]

        return yy.reshape(-1,1)


    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        nx = x.shape[0]

        grad[:, 0, self.pdim] = self.cf * np.ones((nx, ))
        return grad
