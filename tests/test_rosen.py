#!/usr/bin/env python
"""Test script for Rosenblatt transformation."""

import numpy as np
from pytuq.rv.rosen import Rosenblatt


def test_rosen_forward_range():
    # Forward Rosenblatt should map to [0, 1]
    np.random.seed(42)
    nsam = 500
    dim = 2
    xsam = np.random.randn(nsam, dim)

    rosen = Rosenblatt(xsam)

    # Evaluate at a few points within the data range
    for _ in range(10):
        x = np.random.randn(dim)
        u = rosen(x)
        assert(np.all(u >= 0.0) and np.all(u <= 1.0))


def test_rosen_forward_median():
    # For symmetric data, the median should map to ~0.5
    np.random.seed(42)
    nsam = 2000
    dim = 1
    xsam = np.random.randn(nsam, dim)

    rosen = Rosenblatt(xsam)
    u = rosen(np.array([0.0]))

    assert(np.isclose(u[0], 0.5, atol=0.05))


def test_rosen_forward_monotone_1d():
    # Forward Rosenblatt should be monotonically increasing in 1d
    np.random.seed(42)
    nsam = 500
    xsam = np.random.randn(nsam, 1)

    rosen = Rosenblatt(xsam)

    x_vals = np.linspace(-2, 2, 20)
    u_vals = np.array([rosen(np.array([x]))[0] for x in x_vals])

    # Check monotonicity
    assert(np.all(np.diff(u_vals) > 0))


def test_rosen_roundtrip_1d():
    # inv(forward(x)) should approximately recover x
    np.random.seed(42)
    nsam = 1000
    xsam = np.random.randn(nsam, 1)

    rosen = Rosenblatt(xsam)

    x_test = np.array([0.5])
    u = rosen(x_test)
    x_rec = rosen.inv(u)

    assert(np.allclose(x_rec, x_test, atol=0.1))


def test_rosen_roundtrip_2d():
    # inv(forward(x)) should approximately recover x in 2d
    np.random.seed(42)
    nsam = 1000
    dim = 2
    xsam = np.random.randn(nsam, dim)

    rosen = Rosenblatt(xsam)

    x_test = np.array([0.3, -0.2])
    u = rosen(x_test)
    x_rec = rosen.inv(u.copy())

    assert(np.allclose(x_rec, x_test, atol=0.2))


def test_rosen_inv_range():
    # Inverse Rosenblatt should produce values within data range (approximately)
    np.random.seed(42)
    nsam = 500
    dim = 2
    xsam = np.random.randn(nsam, dim)

    rosen = Rosenblatt(xsam)

    u_test = np.array([0.5, 0.5])
    x_rec = rosen.inv(u_test)

    # Should be near the center of the data
    assert(np.all(np.abs(x_rec) < 3.0))


def test_rosen_inv_bfgs_1d():
    # inv_bfgs should also approximately recover x
    np.random.seed(42)
    nsam = 1000
    xsam = np.random.randn(nsam, 1)

    rosen = Rosenblatt(xsam)

    x_test = np.array([0.3])
    u = rosen(x_test)
    x_rec = rosen.inv_bfgs(u)

    assert(np.allclose(x_rec, x_test, atol=0.15))


def test_rosen_custom_sigmas():
    # Should accept custom bandwidth sigmas
    np.random.seed(42)
    nsam = 200
    dim = 2
    xsam = np.random.randn(nsam, dim)

    sigmas = np.array([0.5, 0.5])
    rosen = Rosenblatt(xsam, sigmas=sigmas)

    assert(np.allclose(rosen.sigmas, sigmas))

    u = rosen(np.array([0.0, 0.0]))
    assert(np.all(u >= 0.0) and np.all(u <= 1.0))


def test_rosen_uniform_samples():
    # Forward Rosenblatt of many samples should produce roughly uniform marginals
    np.random.seed(42)
    nsam = 2000
    xsam = np.random.randn(nsam, 1)

    rosen = Rosenblatt(xsam)

    ntest = 500
    x_test = np.random.randn(ntest)
    u_test = np.array([rosen(np.array([x]))[0] for x in x_test])

    # Check that the transformed samples are roughly uniform
    # Mean should be near 0.5
    assert(np.isclose(np.mean(u_test), 0.5, atol=0.05))


if __name__ == '__main__':
    test_rosen_forward_range()
    test_rosen_forward_median()
    test_rosen_forward_monotone_1d()
    test_rosen_roundtrip_1d()
    test_rosen_roundtrip_2d()
    test_rosen_inv_range()
    test_rosen_inv_bfgs_1d()
    test_rosen_custom_sigmas()
    test_rosen_uniform_samples()
