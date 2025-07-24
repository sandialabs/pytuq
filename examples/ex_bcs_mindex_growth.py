#!/usr/bin/env python
"""An example demonstrating bcs and multiindex growth.
"""
import numpy as np


from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.utils.plotting import myrc, lighten_color, plot_dm, plot_jsens
from pytuq.utils.maps import scale01ToDom
from pytuq.lreg.bcs import bcs
from pytuq.utils.mindex import mi_addfront_cons, mi_addfront

import pytuq.utils.funcbank as fcb


myrc()

########################################################
########################################################
########################################################


N = 144
dim = 2
order = 3
datastd = 0.01
niter = 3 # number of growth iterations


true_model = fcb.sinsum # (in 2d, powers with odd sum survive)

########################################################
########################################################
########################################################



# x-domain
domain = np.ones((dim, 2))*np.array([-1.,1.])

# Sample x
x = scale01ToDom(np.random.rand(N,dim), domain)

# Evaluate y
y = true_model(x)
y+=datastd*np.random.randn(N,)

# Build multiindex for PC fit
mindex = get_mi(order, dim)

for ii in range(niter):
    print("=========================================")
    print("Current multiindex")
    print(mindex)
    # PC object
    pcrv = PCRV(1, dim, 'LU', mi=mindex)
    # Evaluate PC bases to get the A-matrix
    Amat = pcrv.evalBases(x, 0)

    # BCS object
    lreg = bcs(eta=1.e-5) # eta is the tolerance parameter: usually, the smaller it is, the more terms are retained

    # BCS fit
    lreg.fita(Amat, y)

    print("Indices of survived bases:")
    print(lreg.used)
    print("Reduced multiindex and corresponding coefficients")
    for m, c in zip(mindex[lreg.used,:], lreg.cf):
        print(m, c)

    print("Growing the basis...")
    mindex, _, _ = mi_addfront(mindex[lreg.used,:]) # use mi_addfront_cons for a more 'conservative' growth


# Plot diagonal fit
yy_pred, yy_pred_var, _ = lreg.predicta(Amat[:, lreg.used], msc=1)
plot_dm([y], [yy_pred], errorbars=[[np.sqrt(yy_pred_var), np.sqrt(yy_pred_var)]],
        labels=['Training'], colors=['b'],
        axes_labels=[f'Model', f'Poly'],
        figname='fitdiag.png',
        legendpos='in', msize=13)
