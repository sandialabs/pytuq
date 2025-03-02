#!/usr/bin/env python
"""Class for Rosenblatt transformation."""

import sys
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import minimize, bisect

from ..utils.xutils import get_opt_bw

class Rosenblatt():
    r"""Rosenblatt transformation class. KDE-based computations follow :cite:t:`Sargsyan:2010`.

    Attributes:
        sigmas (np.ndarray): Per-dimension bandwidths.
        xsam (np.ndarray): The data based on which Rosenblatt map is built. A 2d array of size :math:`(N, d)`.
    """
    def __init__(self, xsam, sigmas=None, bwfactor=1.0):
        super(Rosenblatt, self).__init__()
        self.xsam = xsam
        if sigmas is None:
            self.sigmas = get_opt_bw(xsam, bwf=bwfactor)
        else:
            self.sigmas = sigmas

        #print(self.sigmas)
        return

    def __call__(self, x):
        r"""Forward Rosenblatt call function.

        Args:
            x (np.ndarray): Where Rosenblatt is evaluated. A 2d array of size :math:`(N, d)`.

        Returns:
            np.ndarray: Rosenblatt output, an array of size :math:`d`.
        """
        nsam, ndim = self.xsam.shape
        assert(x.shape[0]==ndim)

        rosen_cdf = np.zeros(ndim)
        ww = np.ones(nsam)
        for idim in range(ndim):
            if idim>0:
                xkk = (x[idim-1] - self.xsam[:, idim-1])/self.sigmas[idim-1]
                ww *= np.exp(-xkk**2/2.)
            denom = np.sum(ww)
            xk = (x[idim] - self.xsam[:, idim])/self.sigmas[idim]
            numer = np.sum(ww*ss.norm.cdf(xk))
            rosen_cdf[idim] = numer/denom

        return rosen_cdf

    def inv_bfgs(self, u, f=0.05):
        r"""Inverse Rosenblatt evaluator, where minimization is done with LBFGS.

        Args:
            u (np.ndarray): An input array of size :math:`d`, all elements of which should in :math:`[0,1]`.
            f (float, optional): Cushion factor for the root search.

        Returns:
            np.ndarray: Output array of size :math:`d`.

        Note:
            This is not being used, since for monotone function we can use hand-made bisection method, see inv().
        """
        ndim,  = u.shape
        def resid(x, u):
            return np.linalg.norm(self.__call__(x)-u)**2
        xmins, xmaxs = np.min(self.xsam, axis=0), np.max(self.xsam, axis=0)
        x0 = 0.5*(xmaxs+xmins)
        # This should be fixed like inv() to allow single data Rosenblatt when xmins=xmaxs
        res = minimize(resid, x0, args=(u,), method='L-BFGS-B',
                       bounds=[(xmin-f*(xmax-xmin), xmax+f*(xmax-xmin)) for xmin, xmax in zip(xmins, xmaxs)],
                       tol=1.e-12)
        # #print(res.x)
        # if res.fun>1.e-6:
        #     print("AAAAAAAAAAA ", u, res.fun)

        return res.x


    def residual(self, x, u, idim):
        r"""Residual function to help the inversion.

        Args:
            x (np.ndarray): Input 2d array of size :math:`(N,d)`
            u (np.ndarray): The 'uniform' sample, an array of size :math:`d`.
            idim (int): Which dimension we are computing the residual for.

        Returns:
            float: Single residual value.
        """
        w = self._ww.copy()
        # if idim>0:
        #     xkk = (self.xsam[:, idim-1]-self._xprev)/self.sigmas[idim-1]
        #     w *= np.exp(-xkk**2/2.)
        #print(idim, w)
        denom = np.sum(w)
        xk = (x-self.xsam[:, idim])/self.sigmas[idim]
        numer = np.sum(w*ss.norm.cdf(xk))
        rosen_cdf = numer/denom
        return (rosen_cdf-u[idim])

    def inv(self, u, f=1.5):
        r"""Inverse Rosenblatt evaluator, where minimization is done with bisection.

        Args:
            u (np.ndarray): An input array of size :math:`d`, all elements of which should in :math:`[0,1]`.
            f (float, optional): Cushion factor for the root search.

        Returns:
            np.ndarray: Output array of size :math:`d`.
        """
        u[u==0]=1.e-6
        u[u==1]=1.-1.e-6
        nsam, ndim = self.xsam.shape
        assert(u.shape[0]==ndim)

        x = np.empty(ndim,)
        xmins, xmaxs = np.min(self.xsam, axis=0), np.max(self.xsam, axis=0)
        delta = xmaxs-xmins
        self._ww = np.ones(nsam)
        for iidim in range(ndim):
            if delta[iidim]==0.0:
                delta[iidim]=1.0

            fa = 0.0
            fb = 0.0
            while fa*fb>=0 and f<1.e+4:
                fa = self.residual(xmins[iidim]-f*delta[iidim], u, iidim)
                fb = self.residual(xmaxs[iidim]+f*delta[iidim], u, iidim)
                f*=2.0

            x[iidim] = bisect(self.residual,
                              xmins[iidim]-f*delta[iidim],
                              xmaxs[iidim]+f*delta[iidim],
                              args=(u,iidim))

            #
            #     print(u, xmins[iidim], xmaxs[iidim])
            #     print(xmins[iidim]-f*delta[iidim], fa)
            #     print(xmaxs[iidim]+f*delta[iidim], fb)
            #     print("Bisection not well-defined. Falling back to LBFGS")
            #     def ff(x,u,idim):
            #         return self.residual(x, u, idim)**2
            #     x0 = 0.5*(xmaxs[iidim]+xmins[iidim])
            #     res = minimize(ff, x0, args=(u,iidim), method='L-BFGS-B',
            #            bounds=[(xmins[iidim], xmaxs[iidim])],
            #            tol=1.e-12)
            #     x[iidim] = res.x

            self._xprev = x[iidim]+0.0
            xkk = (self._xprev - self.xsam[:, iidim])/self.sigmas[iidim]
            self._ww *= np.exp(-xkk**2/2.)

        return x

