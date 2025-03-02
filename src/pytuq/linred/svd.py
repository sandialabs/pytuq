#!/usr/bin/env python
"""Module for SVD-based dimensionality reduction."""

import numpy as np

from .linred import LinRed

class SVD(LinRed):
    r"""Class for SVD-based dimensionality reduction.

    Attributes:
        eigval (np.ndarray): An array of size `K` storing eigenvalues.
        mean (np.ndarray): An array of size `N` storing the mean.
        modes (np.ndarray): A 2d array of size `(N,K)` consisting of all eigenvectors.
        xi (np.ndarray): An array of size `(M,K)` for left eigenvectors of SVD.
        built (bool): Flag to indicate if SVD is built or not.

    Note:
        Unless truncation is done, `K=N`.
    """


    # TODO: need to implement project() function, too, similar to KLE, and not rely on parent project().
    def __init__(self):
        """Initialization."""
        super().__init__()

    def build(self, data, plot=True):
        """Building SVD-based dimensionality reduction.

        Args:
            data (np.ndarray): Data array of size `(N,M)`.
            plot (bool, optional): Flag indicating if auxilliary figures need to be made.
        """
        nx, nsam = data.shape
        #assert(nx>1) # no need to do this for 1 condition

        self.mean = np.mean(data, axis=1)

        U, S, Vt = np.linalg.svd(data.T-self.mean, full_matrices=False)
        max_abs_rows = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[range(Vt.shape[0]), max_abs_rows])
        U *= signs
        Vt *= signs[:, np.newaxis]

        self.eigval = S**2 / (nsam-1)
        self.modes = Vt.T

        #self.modes_clip() # do this to avoid large nx x nx matrix

        self.xi = U * np.sqrt(nsam-1)

        self.built = True

        self.eigval_clip()

        if plot:
            self.plot_all()



