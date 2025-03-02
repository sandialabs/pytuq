#!/usr/bin/env python
"""A test for sampling routines."""

import matplotlib.pyplot as plt

from pytuq.rv.pcrv import PCRV
from pytuq.utils.plotting import myrc

myrc()

pdim = 2
pc = PCRV(pdim, pdim, 'HG')

qdpts, wghts = pc.quadGerm(pts=[15, 7])
plt.plot(qdpts[:,0], qdpts[:,1], 'o')
plt.savefig('qdpts.png')

