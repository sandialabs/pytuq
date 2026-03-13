#!/usr/bin/env python
"""Example demonstrating quadrature point generation for PC germ variables.

This script generates and visualizes quadrature points for polynomial chaos
germ variables using tensor product quadrature rules.
"""

import matplotlib.pyplot as plt

from pytuq.rv.pcrv import PCRV
from pytuq.utils.plotting import myrc

myrc()

# 2D PC object with Hermite-Gauss germ
pdim = 2
pc = PCRV(pdim, pdim, 'HG')

# Generate tensor-product quadrature points with 15x7 grid
qdpts, wghts = pc.quadGerm(pts=[15, 7])
plt.plot(qdpts[:,0], qdpts[:,1], 'o')
plt.savefig('qdpts.png')

