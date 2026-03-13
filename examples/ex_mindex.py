#!/usr/bin/env python
"""Example demonstrating multiindex generation and encoding operations.

This script shows how to generate polynomial chaos multiindices and encode
them for efficient storage and manipulation.
"""

from pytuq.utils.mindex import get_mi, encode_mindex

# Generate total-order multiindex of order 5 in 3 dimensions
mindex = get_mi(5, 3)
print(mindex)

# Encode and print the second-to-last entry
print(encode_mindex(mindex)[-2])
