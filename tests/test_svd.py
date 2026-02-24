#!/usr/bin/env python
"""Test script for SVD dimensionality reduction."""

import numpy as np
from pytuq.linred.svd import SVD


def test_svd_build():
    # Build SVD and check attributes are set
    np.random.seed(42)
    nx = 50
    nsam = 20
    data = np.random.randn(nx, nsam)

    svd = SVD()
    svd.build(data, plot=False)

    assert(svd.built)
    assert(svd.mean.shape == (nx,))
    assert(svd.eigval is not None)
    assert(svd.modes is not None)


def test_svd_reconstruction():
    # Full SVD reconstruction should recover original data
    np.random.seed(42)
    nx = 30
    nsam = 10
    data = np.random.randn(nx, nsam)

    svd = SVD()
    svd.build(data, plot=False)

    # Project and reconstruct with all eigenvalues
    xi = svd.project(data)
    data_rec = svd.eval(xi, add_mean=True)

    assert(np.allclose(data, data_rec, atol=1.e-10))


def test_svd_truncated():
    # Truncated SVD should capture most variance
    np.random.seed(42)
    nx = 50
    nsam = 30

    # Create low-rank data (rank 3) with noise
    U = np.random.randn(nx, 3)
    V = np.random.randn(3, nsam)
    data = U @ V + 0.01 * np.random.randn(nx, nsam)

    svd = SVD()
    svd.build(data, plot=False)

    # Using 3 eigenvalues should capture almost all variance
    neig = 3
    xi = svd.project(data)  # shape (nsam, neig_total)
    data_rec = svd.eval(xi, neig=neig, add_mean=True)  # shape (nx, nsam)

    rel_err = np.linalg.norm(data - data_rec) / np.linalg.norm(data)
    assert(rel_err < 0.05)


def test_svd_mean():
    # Mean should match data mean
    np.random.seed(42)
    nx = 20
    nsam = 15
    data = np.random.randn(nx, nsam) + 5.0

    svd = SVD()
    svd.build(data, plot=False)

    assert(np.allclose(svd.mean, np.mean(data, axis=1)))


def test_svd_eigenvalues_sorted():
    # Eigenvalues should be sorted in descending order
    np.random.seed(42)
    nx = 30
    nsam = 20
    data = np.random.randn(nx, nsam)

    svd = SVD()
    svd.build(data, plot=False)

    eigvals = svd.eigval
    assert(np.all(eigvals[:-1] >= eigvals[1:]))


if __name__ == '__main__':
    test_svd_build()
    test_svd_reconstruction()
    test_svd_truncated()
    test_svd_mean()
    test_svd_eigenvalues_sorted()
