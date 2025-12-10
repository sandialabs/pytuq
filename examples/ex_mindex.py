#!/usr/bin/env python
"""Example demonstrating multiindex generation and encoding operations.

This script shows how to generate polynomial chaos multiindices and encode
them for efficient storage and manipulation.
"""

from pytuq.utils.mindex import get_mi, encode_mindex


mindex = get_mi(5, 3)
print(mindex)

print(encode_mindex(mindex)[-2])
