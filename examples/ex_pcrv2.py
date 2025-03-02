#!/usr/bin/env python

import sys
import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi

order = 3
dim = 2

mi = get_mi(order, dim)
cfs = np.random.rand(mi.shape[0])

hgrv = PCRV(1, dim, 'HG', mi=mi, cfs=cfs)
# Can also do mixed PC where some dimensions are HG and some are LU
# hgrv = PCRV(dim, ['HG', 'LU'], mi=mi, cfs=cfs)


print(hgrv)
hgrv.printInfo()
print(f"Mean = {hgrv.computeMean()}")
print(f"Variance = {hgrv.computeVar()}")
print(f"Basis Norms-Squared : {hgrv.evalBasesNormsSq(0)}")

print(f"Some random samples: {hgrv.sample(nsam=10)}")
