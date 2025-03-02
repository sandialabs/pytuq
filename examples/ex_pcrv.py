#!/usr/bin/env python

"""[summary]

[description]
"""
import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi



pdim = 1
sdim = 5
pc_type='HG'
mi=get_mi(4,sdim)

pcrv = PCRV(pdim, sdim, pctype=pc_type, mi=mi)
pcrv.setRandomCfs()
pcrv.printInfo()
print("==========")
pcrv_new = pcrv.slicePC(fixind=[0,1,3],nominal=np.ones((sdim,)))
pcrv_new.printInfo()
