#!/usr/bin/env python

"""[summary]

[description]
"""
import numpy as np
import matplotlib.pyplot as plt


from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.utils.plotting import myrc, lighten_color, plot_dm, plot_jsens
from pytuq.utils.maps import scale01ToDom
from pytuq.lreg.bcs import bcs
from pytuq.utils.mindex import encode_mindex

import pytuq.utils.funcbank as fcb


myrc()

########################################################
########################################################
########################################################


N = 144
dim = 2
order = 5
datastd = 0.01

#true_model = fcb.const # only constant is supposed to survive, but interestingly a lot of noise is picked up, too.
#true_model = fcb.f2d # exact polynomial, and only specific terms survive
#true_model = fcb.cosine # (even powers survive)
#true_model = fcb.sinsum # (in 2d, powers with odd sum survive)
true_model = fcb.prodabs

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
# PC object
pcrv = PCRV(1, dim, 'LU', mi=mindex)
# Evaluate PC bases to get the A-matrix
Amat = pcrv.evalBases(x, 0)

# BCS obkect
lreg = bcs(eta=1.e-11) # eta is the tolerance parameter: usually, the smaller it is, the more terms are retained

# BCS fit
lreg.fita(Amat, y)


print("Full multiindex basis:")
print(mindex)
print("Indices of survived bases:")
print(lreg.used)
print("Reduced multiindex and corresponding coefficients")
for m, c in zip(mindex[lreg.used,:], lreg.cf):
    print(m, c)

sp_mindex = encode_mindex(mindex[lreg.used,:])


# Compute sensitivities
pcrv.setMiCfs([mindex[lreg.used,:]], [lreg.cf])
mainsens = pcrv.computeTotSens()
print("Main Sensitiviies:  ", mainsens[0])
jointsens = pcrv.computeJointSens()
print("Joint Sensitiviies: ", jointsens[0])

plot_jsens(mainsens[0], jointsens[0])

# Plot diagonal fit
yy_pred, yy_pred_var, _ = lreg.predicta(Amat[:, lreg.used], msc=1)
plot_dm([y], [yy_pred], errorbars=[[np.sqrt(yy_pred_var), np.sqrt(yy_pred_var)]],
        labels=['Training'], colors=['b'],
        axes_labels=[f'Model', f'Poly'],
        figname='fitdiag.png',
        legendpos='in', msize=13)

# If 1d, plot the fit
if dim==1:
    xg = np.linspace(-1., 1., 100).reshape(-1, 1)
    yg = true_model(xg)
    Agr = pcrv.evalBases(xg, 0)
    ygr_pred, ygr_pred_var, _ = lreg.predicta(Agr[:, lreg.used], msc=1)

    plt.plot(x, y, 'ko', ms=11, label='Data')
    plt.plot(xg, yg, 'k--', label='True model')
    ygr_pred_std = np.sqrt(ygr_pred_var)

    p, = plt.plot(xg[:, 0], ygr_pred, 'm-', linewidth=5, label='Fit mean')
    lc = lighten_color(p.get_color(), 0.5)
    plt.fill_between(xg[:, 0],
                     ygr_pred - ygr_pred_std,
                     ygr_pred + ygr_pred_std,
                     color=lc, zorder=-1000, alpha=1.0,
                     label='Fit stdev')
    plt.legend()
    plt.savefig('fit1d.png')

