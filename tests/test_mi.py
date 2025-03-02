#!/usr/bin/env python
"""Test script for multiindices."""

from pytuq.utils.mindex import get_mi, get_npc


def test_mi():
    # Create a multiindex set with known dimensionality and order and total-order truncation rule
    dim = 7
    order = 5
    mi = get_mi(order, dim)

    # Compute the number of multiindices with total-order truncation
    npc = get_npc(order, dim)

    # Assert the shapes match
    assert(mi.shape[0] == npc)
    assert(mi.shape[1] == dim)


if __name__ == '__main__':
    test_mi()
