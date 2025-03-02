#!/usr/bin/env python

"""[summary]

[description]
"""

import sys
import numpy as np
from matplotlib import pyplot as plt

from pytuq.utils.mindex import get_mi, get_npc
from pytuq.func import toy, genz, chem, benchmark, poly, oper, func
from pytuq.utils.plotting import myrc

myrc()

################################################
################################################

fcn = chem.LennardJones()


print("Gradient check")
x = np.random.rand(111, fcn.dim)
assert(np.allclose(fcn.grad_(x, eps=1.e-7), fcn.grad(x)))


xx = np.linspace(fcn.domain[0, 0], fcn.domain[0, 1], 111)
yy = fcn(xx.reshape(-1,1))
gg = fcn.grad(xx.reshape(-1,1))[:,0,0]


fig = plt.figure(figsize=(15,9))
plt.plot(xx, yy, 'b-')

ax1 = plt.gca()
ax1.set_ylabel('x')
ax1.set_ylabel('Function   f(x)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(xx, gg, 'r-')
ax2.set_ylabel("Derivative   f'(x)", color='r')
ax2.tick_params(axis='y', labelcolor='r')

fig.tight_layout()
plt.savefig('funcgrad.png')
