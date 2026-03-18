#!/usr/bin/env python
"""Example demonstrating PCRV compression and random dimension selection.

This script creates a multivariate normal PCRV with specified random dimensions,
samples from it, and demonstrates PC compression operations.
"""

import numpy as np

from pytuq.rv.pcrv import PCRV_mvn
from pytuq.utils.plotting import plot_xrv, myrc

myrc()

dim = 3

# Only dimension 1 is random, others are deterministic
rndind = [1]

# Create PCRV_mvn with random coefficients
pc_rv = PCRV_mvn(dim, rndind=rndind)
pc_rv.setRandomCfs()

# Sample and save
xsam = pc_rv.sample(1000)


np.savetxt('xsam.txt', xsam)
plot_xrv(xsam)

print(pc_rv)

# Compress to remove deterministic dimensions
pc_rv_new = pc_rv.compressPC()
print(pc_rv_new)
