#!/usr/bin/env python
"""A test for some operations with the PCRV class."""

import numpy as np

from pytuq.rv.pcrv import PCRV_mvn
from pytuq.utils.plotting import plot_xrv, myrc

myrc()

dim = 3

rndind = [1]

pc_rv = PCRV_mvn(dim, rndind=rndind)
pc_rv.setRandomCfs()

xsam = pc_rv.sample(1000)


np.savetxt('xsam.txt', xsam)
plot_xrv(xsam)

print(pc_rv)

pc_rv_new = pc_rv.compressPC()
print(pc_rv_new)
