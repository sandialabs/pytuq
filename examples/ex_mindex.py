#!/usr/bin/env python
"""Example of multiindex manipulations."""

import numpy as np

from pytuq.utils.mindex import get_mi, encode_mindex


mindex = get_mi(5, 3)
print(mindex)

print(encode_mindex(mindex)[-2])
