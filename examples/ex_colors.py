#!/usr/bin/env python
"""Example that creates RGB triples."""

from matplotlib import pyplot as plt

from pytuq.utils.plotting import myrc, set_colors


myrc()


ncol = 30
clrs = set_colors(ncol)
#print(clrs)

plt.barh(range(1, ncol + 1),
         [1] * ncol,
         color=clrs)
plt.xticks()
plt.grid(False)
plt.savefig('ex_colors.png')
