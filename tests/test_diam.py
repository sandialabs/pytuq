#!/usr/bin/env python
"""Test script for data diameter and domain computation."""

import numpy as np
from pytuq.utils.stats import diam, get_domain


def test_stat():
    # Create random sample set
    sam = 1111
    dim = 3
    xsam = np.random.rand(sam, dim)

    # Compute its diameter
    diameter = diam(xsam)

    # Assert diameter size since we are in unit hypercube
    assert(diameter < np.sqrt(dim))

    # Compute domain from data
    dom = get_domain(xsam)

    # Assert domain edges are in the unit hypercube
    assert(np.all(dom[:, 0] > 0.0))
    assert(np.all(dom[:, 1] < 1.0))


if __name__ == '__main__':
    test_stat()
