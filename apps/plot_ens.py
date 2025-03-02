#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt



usage_str = 'Script to plot ensemble.'
parser = argparse.ArgumentParser(description=usage_str)
#parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")

parser.add_argument("-y", "--ydata", dest="ydata", type=str, default='ytrain.dat',
                    help="Ydata file")

args = parser.parse_args()

ydata = np.loadtxt(args.ydata)


nsam, nout = ydata.shape

ind_plot=np.arange(nout)

nout_plot=len(ind_plot)

for i in range(nsam):
    plt.plot(np.arange(1, nout+1)[ind_plot], ydata[i, ind_plot], 'b-', lw=0.1)

plt.savefig('ensemble.png')
