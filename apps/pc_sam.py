#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi


usage_str = 'Sampling multivariate PC given coefficient file.'
parser = argparse.ArgumentParser(description=usage_str)
parser.add_argument("-f", "--pcf", dest="pcf_file", type=str, default='pcf.txt', help="PC coefficient file: each column is PC coefficient vector for the corresponding dimension.")
parser.add_argument("-t", "--pct", dest="pc_type", type=str, default='HG', help="PC type", choices=['LU', 'HG'])
parser.add_argument("-n", "--nsam", dest="nsam", type=int, default=111, help="Number of requested samples.")
args = parser.parse_args()


pcf_file = args.pcf_file
pc_type = args.pc_type
nsam = args.nsam

pccf = np.loadtxt(pcf_file)
if len(pccf.shape)==1:
	pccf = pccf[:, np.newaxis]

_, pdim = pccf.shape



pc = PCRV(pdim, pdim, pc_type, mi=get_mi(1, pdim), cfs=[pccf[:, i] for i in range(pdim)])

qsam = pc.sampleGerm(nsam=nsam)
psam = pc.evalPC(qsam)

np.savetxt('psam.txt', psam)
np.savetxt('qsam.txt', qsam)
