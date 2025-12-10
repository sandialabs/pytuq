#!/usr/bin/env python
"""Example demonstrating 1D Genz test functions.

This script evaluates and plots various 1D Genz functions including oscillatory,
corner peak, and sum functions across their domains.
"""
import numpy as np
from matplotlib import pyplot as plt

from pytuq.func import genz
from pytuq.utils.plotting import myrc

myrc()

ngrid = 111
x = np.linspace(0., 1., ngrid).reshape(-1, 1)

gz_list = [
           genz.GenzOscillatory(shift=1., weights=[5.]), \
           genz.GenzCornerPeak(weights=[5.]),\
           genz.GenzSum()
           ]

#: dsa
for gz in gz_list:
    print("The function name is %s" % (gz.name,))
    print("The function domain is ", gz.domain)

    plt.plot(x[:, 0], gz(x), '-', label=gz.name)

plt.grid(False)
plt.xlabel('x')
plt.ylabel('Gz(x)')
plt.legend()
plt.savefig('ex_genz1d.png')
