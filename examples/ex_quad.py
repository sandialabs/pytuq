#!/usr/bin/env python
"""Example demonstrating quadrature point generation for PC germ variables.

This script generates and visualizes quadrature points for polynomial chaos
germ variables using tensor product quadrature rules.
"""

import matplotlib.pyplot as plt

from pytuq.rv.pcrv import PCRV
from pytuq.utils.plotting import myrc

myrc()

pdim = 2
pc = PCRV(pdim, pdim, 'HG')

qdpts, wghts = pc.quadGerm(pts=[15, 7])
plt.plot(qdpts[:,0], qdpts[:,1], 'o')
plt.savefig('qdpts.png')

