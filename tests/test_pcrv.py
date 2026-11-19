#!/usr/bin/env python
"""Test script for PC random variables."""

import numpy as np
import pytest
from pytuq.rv.pcrv import PC1d, PCRV, PCRV_mvn


def test_pc1d_legendre_bases():
    # Legendre polynomials: P_0=1, P_1=x, evaluated at x=0 should give [1, 0]
    pc = PC1d('LU')
    bases = pc(np.array([0.0]), 2)

    assert(np.isclose(bases[0][0], 1.0))  # P_0(0) = 1
    assert(np.isclose(bases[1][0], 0.0))  # P_1(0) = 0


def test_pc1d_normsq_legendre():
    # Norm-squared for Legendre: ||P_n||^2 = 1/(2n+1) on [-1,1]
    pc = PC1d('LU')

    for n in range(5):
        expected = 1.0 / (2 * n + 1)
        assert(np.isclose(pc.normsq(n), expected))


def test_pc1d_normsq_hermite():
    # Norm-squared for Hermite: ||H_n||^2 = n!
    pc = PC1d('HG')
    import math

    for n in range(5):
        expected = float(math.factorial(n))
        assert(np.isclose(pc.normsq(n), expected))


def test_pc1d_quadrature():
    # Quadrature should integrate polynomials of degree <= 2k-1 exactly
    pc = PC1d('LU')
    k = 5
    pts, wts = pc.quad(k)

    # Integrate 1 on [-1, 1]: weights sum to 1 (normalized quadrature)
    integral_one = np.sum(wts)
    assert(np.isclose(integral_one, 1.0))

    # Integrate x^2 on [-1, 1] with normalized weights: exact = 1/3
    integral = np.sum(wts * pts**2)
    assert(np.isclose(integral, 1.0 / 3.0, atol=1.e-10))


def test_pcrv_mean_var():
    # A PCRV with known coefficients should give known mean and variance
    pdim = 1
    sdim = 1
    pctype = 'LU'

    mi = np.array([[0], [1], [2]])
    # cfs shape is (pdim, npc)
    cfs = np.array([[3.0, 2.0, 1.0]])

    pcrv = PCRV(pdim, sdim, pctype, mi=mi, cfs=cfs)

    # Mean is just the 0-th coefficient
    mean = pcrv.computeMean()
    assert(np.isclose(mean[0], 3.0))

    # Variance = sum_{i>0} cfs[i]^2 * normsq(mi[i])
    pc = PC1d('LU')
    var_expected = 2.0**2 * pc.normsq(1) + 1.0**2 * pc.normsq(2)
    var = pcrv.computeVar()
    assert(np.isclose(var[0], var_expected))


def test_pcrv_setcfs_column_vector_list():
    # A list of (K,1) column-vector coefficients (e.g. from a regression on an
    # (N,1) data array) should be stored as 1d arrays, giving the same moments
    mi = np.array([[0], [1], [2]])
    cfs = np.array([3.0, 2.0, 1.0])

    pcrv_ref = PCRV(1, 1, 'LU', mi=mi, cfs=[cfs])
    pcrv = PCRV(1, 1, 'LU', mi=mi)
    pcrv.setCfs([cfs.reshape(-1, 1)])

    assert(pcrv.coefs[0].shape == (3,))
    assert(np.allclose(pcrv.computeMean(), pcrv_ref.computeMean()))
    assert(np.allclose(pcrv.computeVar(), pcrv_ref.computeVar()))


def test_pcrv_setcfs_column_vector_array_pdim1():
    # For a single physical dimension, a (K,1) array is unambiguously a
    # column vector of K coefficients
    mi = np.array([[0], [1], [2]])
    cfs = np.array([3.0, 2.0, 1.0])

    pcrv = PCRV(1, 1, 'LU', mi=mi, cfs=cfs.reshape(-1, 1))

    assert(pcrv.coefs[0].shape == (3,))
    assert(np.allclose(pcrv.coefs[0], cfs))


def test_pcrv_setcfs_rejects_multicolumn():
    # A (K,2) array per physical dimension is not a valid coefficient array
    mi = np.array([[0], [1], [2]])
    pcrv = PCRV(1, 1, 'LU', mi=mi)

    with pytest.raises(ValueError, match="1d"):
        pcrv.setCfs([np.ones((3, 2))])


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_pcrv_slice():
    # Fixing a germ dimension at a nominal value gives a PC in the remaining
    # dimension that agrees with the original PC along that slice
    mi = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0]])
    cfs = np.array([1.0, 2.0, -1.0, 0.5, 0.3])
    pcrv = PCRV(1, 2, 'LU', mi=mi, cfs=cfs)

    nominal = np.array([0.4, 0.0])
    pcrv_sliced = pcrv.slicePC(fixind=[0], nominal=nominal)

    xi = np.linspace(-1., 1., 7).reshape(-1, 1)
    xi_full = np.hstack((np.full_like(xi, nominal[0]), xi))
    assert(pcrv_sliced.sdim == 1)
    assert(np.allclose(pcrv_sliced.evalPC(xi), pcrv.evalPC(xi_full)))


def test_pcrv_sample_shape():
    # Sampling should produce correct shape
    pdim = 2
    sdim = 3
    pctype = 'LU'

    mi = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cfs = np.random.randn(2, 4)  # shape (pdim, npc)

    pcrv = PCRV(pdim, sdim, pctype, mi=mi, cfs=cfs)

    nsam = 100
    samples = pcrv.sample(nsam, seed=42)
    assert(samples.shape == (nsam, pdim))


def test_pcrv_mvn_mean_cov():
    # PCRV_mvn should reproduce specified mean and covariance
    pdim = 2
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])

    pcrv = PCRV_mvn(pdim, mean=mean, cov=cov)

    # Check mean from PC coefficients
    computed_mean = pcrv.computeMean()
    assert(np.allclose(computed_mean, mean))

    # Check variance
    computed_var = pcrv.computeVar()
    assert(np.allclose(computed_var, np.diag(cov), atol=1.e-10))


def test_pcrv_mvn_sampling():
    # Large sample from PCRV_mvn should have stats close to target
    pdim = 2
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])

    pcrv = PCRV_mvn(pdim, mean=mean, cov=cov)

    nsam = 50000
    samples = pcrv.sample(nsam, seed=42)

    # Sample mean and covariance should be close
    assert(np.allclose(np.mean(samples, axis=0), mean, atol=0.05))
    assert(np.allclose(np.cov(samples.T), cov, atol=0.1))


if __name__ == '__main__':
    test_pc1d_legendre_bases()
    test_pc1d_normsq_legendre()
    test_pc1d_normsq_hermite()
    test_pc1d_quadrature()
    test_pcrv_mean_var()
    test_pcrv_setcfs_column_vector_list()
    test_pcrv_setcfs_column_vector_array_pdim1()
    test_pcrv_setcfs_rejects_multicolumn()
    test_pcrv_slice()
    test_pcrv_sample_shape()
    test_pcrv_mvn_mean_cov()
    test_pcrv_mvn_sampling()
