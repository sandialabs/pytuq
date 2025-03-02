#!/usr/bin/env python
"""A"""

import numpy as np
import matplotlib.pyplot as plt

from pytuq.rv.pcrv import PC1d
from pytuq.utils.plotting import myrc

##################################################

myrc()

hgpc = PC1d('HG')

x = np.linspace(-3.0, 3.0, 100)
maxord = 3
y_list = hgpc(x, maxord)
y = np.array(y_list)

for jord in range(maxord+1):
    plt.plot(x, y[jord,:], '-', label=f'Order {jord}')

plt.xlabel(r'$x$')
plt.ylabel(r'$H_k(x)$')
plt.legend()
plt.savefig('hermite.png')
