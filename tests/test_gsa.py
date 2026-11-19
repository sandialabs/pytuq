#!/usr/bin/env python
"""Test script for global sensitivity analysis."""

import numpy as np
import pytest
from pytuq.gsa.gsa import model_sens, PCSobol


def test_gsa_additive_model():
    # For an additive model f(x1,x2) = x1 + x2, main sensitivities should be ~0.5 each
    def additive_model(x, params):
        return (x[:, 0] + x[:, 1]).reshape(-1, 1)

    domain = np.array([[0., 1.], [0., 1.]])

    sens_main, sens_tot = model_sens(additive_model, {}, domain,
                                      method='SamSobol', nsam=5000, plot=False)

    # For additive model, main sens ≈ total sens ≈ 0.5 each
    assert(np.allclose(sens_main[0], 0.5, atol=0.1))
    assert(np.allclose(sens_tot[0], 0.5, atol=0.1))


def test_gsa_single_variable():
    # f(x1, x2) = x1 => all sensitivity on x1
    def single_var_model(x, params):
        return x[:, 0:1]

    domain = np.array([[0., 1.], [0., 1.]])

    sens_main, sens_tot = model_sens(single_var_model, {}, domain,
                                      method='SamSobol', nsam=5000, plot=False)

    # x1 has all sensitivity, x2 has none
    assert(sens_main[0, 0] > 0.8)
    assert(sens_main[0, 1] < 0.2)


def test_gsa_output_shape():
    # Output shapes should be (nout, ndim)
    def model(x, params):
        return np.column_stack([x[:, 0] + x[:, 1], x[:, 0] * x[:, 1]])

    domain = np.array([[0., 1.], [0., 1.]])

    sens_main, sens_tot = model_sens(model, {}, domain,
                                      method='SamSobol', nsam=1000, plot=False)

    assert(sens_main.shape == (2, 2))
    assert(sens_tot.shape == (2, 2))


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("as_column", [False, True])
def test_pcsobol_single_output(as_column):
    # f(x1,x2) = x1 + x2^2 on [-1,1]^2 is exactly representable by an order-3
    # Legendre PC, so Sobol indices are exact: Var(x1)=1/3, Var(x2^2)=4/45.
    # Model outputs given as (N,) or as an (N,1) column must both work.
    domain = np.array([[-1., 1.], [-1., 1.]])
    sMethod = PCSobol(domain, pctype='LU', order=3)
    xsam = sMethod.sample(2000)
    ysam = xsam[:, 0] + xsam[:, 1]**2
    if as_column:
        ysam = ysam.reshape(-1, 1)

    sens = sMethod.compute(ysam)

    var1, var2 = 1./3., 4./45.
    assert(sMethod.pcrv.coefs[0].ndim == 1)
    assert(np.allclose(sens['main'], [var1/(var1+var2), var2/(var1+var2)]))
    assert(np.allclose(sens['total'], sens['main']))


def test_pcsobol_rejects_multioutput():
    # PCSobol.compute handles one output at a time (model_sens loops over outputs)
    domain = np.array([[-1., 1.], [-1., 1.]])
    sMethod = PCSobol(domain, pctype='LU', order=2)
    xsam = sMethod.sample(200)
    ysam = np.column_stack([xsam[:, 0], xsam[:, 1]])

    with pytest.raises(ValueError, match="single output"):
        sMethod.compute(ysam)


if __name__ == '__main__':
    test_gsa_additive_model()
    test_gsa_single_variable()
    test_gsa_output_shape()
    test_pcsobol_single_output(False)
    test_pcsobol_single_output(True)
    test_pcsobol_rejects_multioutput()
