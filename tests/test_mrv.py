#!/usr/bin/env python
"""Test script for multivariate random variables."""

import numpy as np
import pytest
from pytuq.rv.mrv import Mixture, Normal_1d
from pytuq.rv.pcrv import PCRV_mvn


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_mixture_sample():
    # Mixture components return single samples as size-1 arrays of different
    # shapes (Normal_1d and 1d PCRV_mvn); the mixture should return a 1d array
    np.random.seed(42)
    rv1 = Normal_1d(mu=-10.0, sigma=1.0)
    rv2 = PCRV_mvn(1, mean=np.array([10.0]), cov=np.array([[1.0]]))
    mix = Mixture([rv1, rv2], weights=[0.3, 0.7])

    nsam = 5000
    xsam = mix.sample(nsam)

    assert(xsam.shape == (nsam,))
    # Mixture mean is 0.3*(-10) + 0.7*10 = 4
    assert(np.isclose(np.mean(xsam), 4.0, atol=0.5))
    # Components are well separated: the fraction of positive samples is the weight of the second one
    assert(np.isclose(np.mean(xsam > 0), 0.7, atol=0.03))


if __name__ == '__main__':
    test_mixture_sample()
