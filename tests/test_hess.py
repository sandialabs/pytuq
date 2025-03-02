#!/usr/bin/env python
"""Test script for numerical Hessian computation."""

import numpy as np
from pytuq.func import toy, oper



def test_hess():
    # Create to Identity functions and make their Cartesian product, that is f(x,y) = xy
    fcn = oper.CartesProdFcn(toy.Identity(1), toy.Identity(1))
    fcn = fcn * fcn
    # Evaluate the Hessian at randomy selected points
    x = np.random.rand(111, fcn.dim)
    hess = fcn.hess_(x)

    # Assert the Hessian evaluation coincides with known analytical answers
    assert(np.allclose(hess[:, 0, 0, 0], 2 * x[:, 1]**2))
    assert(np.allclose(hess[:, 0, 1, 0], 4. * x[:, 0] * x[:, 1]))
    assert(np.allclose(hess[:, 0, 0, 1], 4. * x[:, 0] * x[:, 1]))
    assert(np.allclose(hess[:, 0, 1, 1], 2 * x[:, 0]**2))


if __name__ == '__main__':
    test_hess()
