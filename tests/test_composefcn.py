#!/usr/bin/env python
"""Test script for function composition."""

import numpy as np
from pytuq.func import oper, toy


def test_composefcn():
    # Create the composite function
    dim = 3 # Dimensionality
    f1 = oper.PickDim(dim, 0) # Create the first function
    f2 = toy.Exp(np.array([-2.])) # Create the second function
    fcn = oper.ComposeFcn(f1, f2)

    # Evaluate at random points
    NN = 13
    xx = np.random.rand(NN, dim)

    # Check the function and gradient with known values
    assert(np.linalg.norm(fcn(xx)[:, 0] - np.exp(- 2. * xx[:, 0])) < 1.e-16)
    assert(np.linalg.norm(fcn.grad(xx)[:, 0, 0] + 2. * np.exp(- 2. * xx[:, 0])) < 1.e-16)


if __name__ == '__main__':
    test_composefcn()
