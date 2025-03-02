#!/usr/bin/env python


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pytuq.rv.mrv import Mixture, Weibull_1d
from pytuq.rv.pcrv import PCRV_mvn

k = 2.0
lam = 6.0

nsam=10000



wb1 = Weibull_1d(lam, k)
wb2 = PCRV_mvn(1, mean=np.array([-40]), cov=np.array([[16.0]]))
wb3 = PCRV_mvn(1, mean=np.array([40]), cov=np.array([[4.0]]))
wb = Mixture([wb1, wb2, wb3], weights=[0.6, 0.3, 0.1])
print(wb)


xsam = wb.sample(nsam)



count, bins, ignored = plt.hist(xsam)
x = np.linspace(-100., 100., 1000)
scale = count.max()/wb.pdf(x).max()
plt.plot(x, wb.pdf(x)*scale)
plt.savefig('mixture.png')
