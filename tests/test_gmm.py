#!/usr/bin/env python
"""Test script for truncated Gaussian mixture model integration."""

import numpy as np
from pytuq.rv.mrv import GMM



def test_gmm():
    # Create location of GMM means
    means = np.array([[2.0, 0.0], [12.0, 4.0]])
    ncl, dim = means.shape
    # Weights of each mixture
    weights = [4., 2.]
    # Create the GMM object
    mygmm = GMM(means, weights=weights)

    # Create a domain of interest that is large enough
    domain = np.array([[-20., 30.], [-10., 20.]])

    # Integrate within the volume
    pdf_integral = mygmm.volume_indomain(domain)

    # Assert it integrates to 1.0
    assert(np.isclose(pdf_integral, 1.0))


if __name__ == '__main__':
    test_gmm()
