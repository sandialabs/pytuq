#!/usr/bin/env python

import os
import sys
import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi


pccf_file = sys.argv[1]
pc_type = sys.argv[2]
nsam = int(sys.argv[3])

pccf = np.loadtxt(pccf_file)
if len(pccf.shape)==1:
	pccf = pccf[:, np.newaxis]

_, pdim = pccf.shape



pc = PCRV(pdim, pdim, pc_type, mi=get_mi(1, pdim), cfs=[pccf[:, i] for i in range(pdim)])

qsam = pc.sampleGerm(nsam=nsam)
psam = pc.evalPC(qsam)

np.savetxt('psam.txt', psam)
np.savetxt('qsam.txt', qsam)
