#!/usr/bin/env python
"""Example demonstrating 1D polynomial chaos basis evaluation and plotting.

This script evaluates and plots Hermite polynomial basis functions of various
orders to illustrate orthogonal polynomial behavior.
"""

import numpy as np
import matplotlib.pyplot as plt

from pytuq.rv.pcrv import PC1d
from pytuq.utils.plotting import myrc

##################################################

myrc()

# Create 1D Hermite PC basis
hgpc = PC1d('HG')

# Evaluate basis polynomials up to order 3 on a grid
x = np.linspace(-3.0, 3.0, 100)
maxord = 3
y_list = hgpc(x, maxord)
y = np.array(y_list)

# Plot each order
for jord in range(maxord+1):
    plt.plot(x, y[jord,:], '-', label=f'Order {jord}')

plt.xlabel(r'$x$')
plt.ylabel(r'$H_k(x)$')
plt.legend()
plt.savefig('hermite.png')
