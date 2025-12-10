#!/usr/bin/env python
"""Example demonstrating color palette generation and visualization.

This script creates and displays a set of RGB color triples using PyTUQ's
plotting utilities, useful for creating consistent color schemes in plots.
"""

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
