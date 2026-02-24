#!/usr/bin/env python
"""Test script for global sensitivity analysis."""

import numpy as np
from pytuq.gsa.gsa import model_sens


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


if __name__ == '__main__':
    test_gsa_additive_model()
    test_gsa_single_variable()
    test_gsa_output_shape()
