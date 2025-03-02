#!/usr/bin/env python
"""Module for SVD-based dimensionality reduction."""

import numpy as np

from .linred import LinRed


class KLE(LinRed):
    r"""Class for Karhunen-Loeve expansion.

    Attributes:
        eigval (np.ndarray): An array of size `K` storing eigenvalues.
        mean (np.ndarray): An array of size `N` storing the mean.
        modes (np.ndarray): A 2d array of size `(N,K)` consisting of all eigenvectors.
        weights2 (np.ndarray): Working array of integration weights.
        xi (np.ndarray): An array of size `(M,K)` for latent features of KLE.
        built (bool): Flag to indicate if KLE is built or not.

    Note:
        Unless truncation is done, `K=N`.
    """

    def __init__(self):
        """Initialization."""
        super().__init__()

        self.weights2 = None

    def build(self, data, plot=True):
        r"""Building KLE dimensionality reduction.

        Args:
            data (np.ndarray): Data array of size `(N,M)`.
            plot (bool, optional): Flag indicating if auxilliary figures need to be made.
        """
        nx, nsam = data.shape
        #assert(nx>1) # no need to do this for 1 condition

        self.mean = np.mean(data, axis=1)
        cov = np.cov(data)

        # Set trapesoidal rule weights
        self.weights2 = np.ones(nx)
        self.weights2[0] = 0.5
        self.weights2[-1] = 0.5
        weights = np.sqrt(self.weights2)

        cov_sc = np.outer(weights, weights) * cov

        self.eigval, eigvec = np.linalg.eigh(cov_sc)
        self.eigval = self.eigval[::-1]
        eigvec = eigvec[:, ::-1]

        self.modes = eigvec / weights.reshape(-1, 1) # nx, neig

        #self.modes_clip() # do this to avoid large nx x nx matrix

        self.built = True

        self.eigval_clip()
        self.xi = self.project(data)

        if plot:
            self.plot_all()

    def project(self, data, subtract_mean=True):
        """Projecting data to the built bases.

        Args:
            data (np.ndarray): Data array of size `(N,M)`.
            subtract_mean (bool, optional): Whether to subtract the mean before projection or not. Defaults to True. Useful without subtraction, e.g. if we are projecting perturbations or standard deviations.

        Returns:
            np.ndarray: Array of latent features of size `(N,K)`.
        """
        assert(self.built)
        # check the number of modes kept
        nmodes = self.modes.shape[1]
        xi = np.dot(data.T - int(subtract_mean)*self.mean, self.modes * self.weights2.reshape(-1, 1)) / np.sqrt(self.eigval[:nmodes]) #nsam, neig

        return xi


