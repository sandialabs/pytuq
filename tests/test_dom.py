#!/usr/bin/env python
"""Test script for domain mapping."""

import numpy as np
from pytuq.utils.maps import scale01ToDom, scaleDomTo01


def test_dom():
    # Create random sample set
    sam = 111
    dim = 3
    xsam_01 = np.random.rand(sam, dim)

    # Create a domain
    domain = np.ones((dim, 1)) * np.array([-2., 3.])

    # Map to domain
    xsam = scale01ToDom(xsam_01, domain)

    # Map back to [0, 1]
    xsam_01_ = scaleDomTo01(xsam, domain)

    # Assert they landed at original locations
    assert(np.allclose(xsam_01, xsam_01_))

    # Back to domain again
    xsam_ = scale01ToDom(xsam_01_, domain)

    # Assert they landed at same locations
    assert(np.allclose(xsam, xsam_))


if __name__ == '__main__':
    test_dom()
