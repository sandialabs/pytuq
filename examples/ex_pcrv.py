#!/usr/bin/env python
"""Example demonstrating polynomial chaos random variable slicing operations.

This script shows how to slice a PC random variable by fixing certain dimensions
at nominal values to obtain a reduced-dimension PCRV.
"""
import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi



pdim = 1
sdim = 5
pc_type='HG'
mi=get_mi(4,sdim)

# Create a PCRV with random coefficients
pcrv = PCRV(pdim, sdim, pctype=pc_type, mi=mi)
pcrv.setRandomCfs()
pcrv.printInfo()

# Slice the PC by fixing dimensions 0, 1, 3 at nominal value 1.0
print("==========")
pcrv_new = pcrv.slicePC(fixind=[0,1,3],nominal=np.ones((sdim,)))
pcrv_new.printInfo()
