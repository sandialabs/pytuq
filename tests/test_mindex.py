#!/usr/bin/env python
"""Test script for multiindex utilities."""

import numpy as np
from pytuq.utils.mindex import get_mi, get_npc, encode_mindex, mi_addfront


def test_mi_shape():
    # Multiindex array should have correct shape
    dim = 3
    order = 4
    mi = get_mi(order, dim)
    npc = get_npc(order, dim)

    assert(mi.shape == (npc, dim))


def test_mi_first_row_zeros():
    # First row of multiindex should be all zeros (constant term)
    dim = 4
    order = 3
    mi = get_mi(order, dim)

    assert(np.all(mi[0] == 0))


def test_mi_total_order_truncation():
    # Every row should have total order <= specified order
    dim = 3
    order = 5
    mi = get_mi(order, dim)

    row_sums = np.sum(mi, axis=1)
    assert(np.all(row_sums <= order))


def test_npc_known_values():
    # Known values: C(p+d, d)
    # dim=1, order=3 => C(4,1) = 4
    assert(get_npc(3, 1) == 4)

    # dim=2, order=2 => C(4,2) = 6
    assert(get_npc(2, 2) == 6)

    # dim=3, order=2 => C(5,3) = 10
    assert(get_npc(2, 3) == 10)

    # dim=1, order=0 => 1
    assert(get_npc(0, 1) == 1)


def test_encode_mindex():
    # Encode multiindex should produce valid encoding
    mi = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    encoded = encode_mindex(mi)

    assert(len(encoded) == mi.shape[0])


def test_mi_addfront():
    # Adding front should expand multiindex set
    dim = 2
    order = 1
    mi = get_mi(order, dim)
    npc_orig = mi.shape[0]

    mi_new, added, front = mi_addfront(mi)

    # New set should be larger
    assert(mi_new.shape[0] > npc_orig)
    # Front elements should be valid multiindices
    assert(front.shape[1] == dim)


def test_npc_dim1():
    # For dim=1, npc should be order+1
    for order in range(10):
        assert(get_npc(order, 1) == order + 1)


if __name__ == '__main__':
    test_mi_shape()
    test_mi_first_row_zeros()
    test_mi_total_order_truncation()
    test_npc_known_values()
    test_encode_mindex()
    test_mi_addfront()
    test_npc_dim1()
