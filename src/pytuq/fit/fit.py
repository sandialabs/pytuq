#!/usr/bin/env python
"""Module for general regression base class."""

import numpy as np

class fitbase(object):
    """Base fit class.

    Attributes:
        fitted (bool): Whether the current instance is fitted or not.
    """

    def __init__(self):
        """Initialization."""
        self.fitted = False

    def fit(self, x, y):
        r"""Fit method.

        Args:
            x (np.ndarray): 2d x-data array of size :math:`(N,d)`.
            y (np.ndarray): 1d y-data array of size :math:`N`.

        Raises:
            NotImplementedError: Base class fit not implemented. This should be implemented in a children class.
        """
        raise NotImplementedError

    def predict(self, x, msc=0, pp=False):
        r"""Predict method.

        Args:
            x (np.ndarray): 2d array of size :math:`(N,d)`.
            msc (int, optional): Prediction mode: 0 (mean-only), 1 (mean and variance), or 2 (mean, variance and covariance). Defaults to 0.
            pp (bool, optional): Whether to compute posterior-predictive (i.e. add data variance) or not.

        Raises:
            NotImplementedError: Base class predict not implemented. This should be implemented in a children class.
        """
        raise NotImplementedError


    def predict_wstd(self, xc, pp=False):
        """Predict mean and standard deviation only.

        Args:
            xc (np.ndarray): 2d array of size `(N,d)`.
            pp (bool, optional): Whether to compute posterior-predictive (i.e. add data variance) or not.

        Returns:
            tuple(np.ndarray, np.ndarray): Tuple of mean and standard deviation arrays each of size `N`.
        """
        yc, yc_var, _  = self.predict(xc, msc=1, pp=pp)
        return yc, np.sqrt(yc_var)
