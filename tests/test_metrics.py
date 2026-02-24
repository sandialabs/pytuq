#!/usr/bin/env python
"""Test script for metrics utilities."""

import numpy as np
from pytuq.utils.metrics import rel_l2, rmse, rel_rmse, norm_rmse, mae, crps_gauss


def test_perfect_predictions():
    # Perfect predictions should yield zero error
    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = targets.copy()

    assert(rmse(predictions, targets) == 0.0)
    assert(rel_l2(predictions, targets) == 0.0)
    assert(mae(predictions, targets) == 0.0)
    assert(rel_rmse(predictions, targets) == 0.0)
    assert(norm_rmse(predictions, targets) == 0.0)


def test_rmse_known_value():
    # RMSE of [1,2,3] vs [4,5,6] should be 3.0
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([4.0, 5.0, 6.0])

    assert(np.isclose(rmse(predictions, targets), 3.0))


def test_mae_known_value():
    # MAE of [1,2,3] vs [4,6,8] should be 3.0
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([4.0, 6.0, 8.0])

    # Errors are 3, 4, 5 => mean = 4.0
    assert(np.isclose(mae(predictions, targets), 4.0))


def test_rel_l2_known_value():
    # Relative L2 = ||pred - targ|| / ||targ||
    predictions = np.array([0.0, 0.0])
    targets = np.array([3.0, 4.0])

    # ||pred - targ|| = 5.0, ||targ|| = 5.0, rel_l2 = 1.0
    assert(np.isclose(rel_l2(predictions, targets), 1.0))


def test_crps_gauss_perfect():
    # CRPS for Gaussian: when sigma -> 0, CRPS approx |y - mu|
    y = np.array([2.0])
    mu = np.array([2.0])
    sig = np.array([1.e-10])

    crps_val = crps_gauss(y, mu, sig)

    assert(crps_val[0] < 1.e-5)


if __name__ == '__main__':
    test_perfect_predictions()
    test_rmse_known_value()
    test_mae_known_value()
    test_rel_l2_known_value()
    test_crps_gauss_perfect()
