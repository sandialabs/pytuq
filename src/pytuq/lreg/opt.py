#!/usr/bin/env python
"""A module for a simple quadratic optimization to fit the coefficients."""

import numpy as np
from scipy.optimize import minimize

from .lreg import lreg

def distance(x, aw, bw):
    r"""Quadratic (least-squares) distance metric.

    Args:
        x (np.ndarray): A 1d coefficient array of size :math:`K`.
        aw (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
        bw (np.ndarray): An 1d array of size :math:`N` holding the data.

    Returns:
        float: Squared-residual distance.
    """
    return np.linalg.norm(np.dot(aw, x) - bw)**2


def distance_grad(x, aw, bw):
    r"""Gradient of the distance metric, analytically avaliable.

    Args:
        x (np.ndarray): A 1d coefficient array of size :math:`K`.
        aw (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
        bw (np.ndarray): An 1d array of size :math:`N` holding the data.

    Returns:
        np.ndarray: Gradient array of size :math:`K`.
    """
    return 2.0*np.dot(aw.T, np.dot(aw, x) - bw)



class opt(lreg):
    """A class of a simple quadratic optimization to fit the coefficients."""
    def __init__(self):
        """Initialization."""
        super().__init__()


    def fita(self, Amat, y):
        r"""Fit given A-matrix of basis evaluations and data array.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            y (np.ndarray): An 1d array of size :math:`N` holding the data.
        """
        param_ini = np.random.randn(Amat.shape[1], )
        res = minimize(distance, param_ini, args=(Amat, y), method='BFGS', options={'gtol': 1e-13}, jac=distance_grad)
        self.cf = res.x
        self.cf_cov = np.zeros((Amat.shape[1], Amat.shape[1]))

        self.fitted = True

