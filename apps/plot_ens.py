#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytuq.utils.plotting import myrc

myrc()

usage_str = 'Script to plot ensemble.'
parser = argparse.ArgumentParser(description=usage_str)
parser.add_argument("-y", "--ydata", dest="ydata", type=str, default='ytrain.dat',
                    help="Ydata file")

args = parser.parse_args()

ydata = np.loadtxt(args.ydata)
if len(ydata.shape)==1:
    ydata = ydata[:, np.newaxis]


nsam, nout = ydata.shape



for i in range(nsam):
    plt.plot(np.arange(1, nout+1), ydata[i, :], 'bo-', markeredgecolor='w', lw=0.5)

plt.xticks(np.arange(1, nout+1))
plt.xtickslabels = [str(i) for i in range(1, nout+1)]
plt.xlabel('Output Id')
plt.ylabel('Output Value')
plt.savefig('ensemble.png')
