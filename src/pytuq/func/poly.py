#!/usr/bin/env python
"""
    Polynomial expansions module.
"""
import numpy as np

from .func import Function


class PolyBase(Function):
    r"""Base class for polynomial functions

    Attributes:
        mindex (numpy array, 2d): Multi-index array for polynomial terms.
        cfs (numpy array, 1d): Coefficients for polynomial terms.
        nbases (int): Number of basis functions.
        name (str): Name of the polynomial function.
        max_deg (numpy array, 1d): Maximum degree for each dimension.
        outdim (int): Output dimension (default is 1).
        bases1d (list): List of 1D basis functions.
        bases1d_deriv (list): List of derivatives of 1D basis functions.

    """

    def __init__(self, mindex=np.array([[0], [1]]), cfs=None, domain=None, name='Poly'):
        super().__init__()


        nbases = mindex.shape[0]
        if cfs is None:
            cfs = np.random.rand(nbases)

        self.mindex = mindex
        self.cfs = cfs
        self.nbases = self.cfs.shape[0]
        self.name = name
        self.max_deg = np.max(self.mindex, axis=0)
        self.outdim = 1
        self.bases1d = None
        self.bases1d_deriv = None

        if domain is None:
            self.dmax = 1.0
            self.setDimDom(dimension=self.mindex.shape[1])
        else:
            self.setDimDom(domain=domain)
            assert(self.dim==self.mindex.shape[1])

        return

    def __call__(self, x):

        #: Ensure x is in the domain
        self.checkDomain(x)
        self.checkDim(x)

        assert(self.bases1d is not None)

        evals = []
        for idim in range(self.dim):
            vv = np.empty((x.shape[0], self.max_deg[idim] + 1))
            for deg in range(self.max_deg[idim] + 1):
                vv[:, deg] = self.bases1d[deg](x[:, idim])
            evals.append(vv)

        pval = 0.0
        for ik in range(self.nbases):
            val = self.cfs[ik] * np.ones(x.shape[0],)
            for idim in range(self.dim):
                val *= evals[idim][:, self.mindex[ik, idim]]
            pval += val

        return pval.reshape(-1,1)

    def grad(self, x):

        #: Ensure x is in the domain
        self.checkDomain(x)
        self.checkDim(x)

        assert(self.bases1d is not None)
        evals = []
        for idim in range(self.dim):
            vv = np.empty((x.shape[0], self.max_deg[idim] + 1))
            for deg in range(self.max_deg[idim] + 1):
                vv[:, deg] = self.bases1d[deg](x[:, idim])
            evals.append(vv)


        assert(self.bases1d_deriv is not None)
        evals_deriv = []
        for idim in range(self.dim):
            vv = np.empty((x.shape[0], self.max_deg[idim] + 1))
            for deg in range(self.max_deg[idim] + 1):
                vv[:, deg] = self.bases1d_deriv[deg](x[:, idim])
            evals_deriv.append(vv)

        gval = np.empty((x.shape[0], self.outdim, self.dim))
        for jdim in range(self.dim):
            pval = 0.0
            for ik in range(self.nbases):
                val = self.cfs[ik] * np.ones(x.shape[0],)
                for idim in range(self.dim):
                    if idim == jdim:
                        val *= evals_deriv[idim][:, self.mindex[ik, idim]]
                    else:
                        val *= evals[idim][:, self.mindex[ik, idim]]
                pval += val
            gval[:, 0, jdim] = pval

        return gval

class Leg(PolyBase):
    r"""Legendre polynomial expansion

    Reference: [https://en.wikipedia.org/wiki/Legendre_polynomials]
    """
    def __init__(self, mindex=np.array([[0], [1]]), cfs=None, domain=None, name='Legendre_Poly'):
        super().__init__(mindex=mindex, cfs=cfs, domain=domain, name=name)

        cfs_ = np.zeros(np.max(self.max_deg) + 1)
        poly = np.polynomial.legendre.Legendre(cfs_)
        self.bases1d = []
        self.bases1d_deriv = []
        for iord in range(np.max(self.max_deg) + 1):
            self.bases1d.append(poly.basis(iord))
            self.bases1d_deriv.append(poly.basis(iord).deriv())



class Mon(PolyBase):
    r"""Monomial polynomial expansion

    Reference: [https://en.wikipedia.org/wiki/Monomial]
    """
    def __init__(self, mindex=np.array([[0], [1]]), cfs=None, domain=None, name='Monomial_Poly'):
        super().__init__(mindex=mindex, cfs=cfs, domain=domain, name=name)

        cfs_ = np.zeros(np.max(self.max_deg) + 1)
        poly = np.polynomial.polynomial.Polynomial(cfs_)
        self.bases1d = []
        self.bases1d_deriv = []
        for iord in range(np.max(self.max_deg) + 1):
            self.bases1d.append(poly.basis(iord))
            self.bases1d_deriv.append(poly.basis(iord).deriv())

