#!/usr/bin/env python
"""Example demonstrating linear regression with measurement error.

This script shows how to perform polynomial chaos regression accounting for
measurement errors in the data using the MERR (Measurement Error in Regression) method.
"""
import numpy as np
import matplotlib.pyplot as plt


from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.utils.plotting import myrc, lighten_color
from pytuq.utils.maps import scale01ToDom
from pytuq.lreg.merr import lreg_merr

myrc()

########################################################
########################################################
########################################################

N = 14
order = 4
true_model = lambda x: x[:,0]**4 - 2.*x[:,0]**3 #fcb.sin4
dim = 1

########################################################
########################################################
########################################################


domain = np.ones((dim, 2))*np.array([-1.,1.])

x = scale01ToDom(np.random.rand(N,dim), domain)


y = true_model(x)

pcrv = PCRV(1, dim, 'LU', mi=get_mi(order, dim))

Amat = pcrv.evalBases(x, 0)

print(Amat.shape)

# dvinfo = {'fcn': lambda v, p: v*(1.*(p-1.))**4, 'npar': 1, 'aux': x.reshape(-1,)}
# dvinfo = {'fcn': lambda v, p: (10.*(p-0.5))**4, 'npar': 0, 'aux': y}
dvinfo = {'fcn': lambda v, p: 0.01*np.ones_like(p), 'npar': 0, 'aux': y}
#dvinfo = {'fcn': lambda v, p: np.exp(v)*np.ones_like(p), 'npar': 1, 'aux': y}

ind_embed = [0, 1, 2] #range(nbases)
lreg = lreg_merr(ind_embed=ind_embed, dvinfo=dvinfo, merr_method='iid', opt_method='mcmc', embedding='mvn', mean_fixed=True)
#lreg = anl(method='vi')
#lreg = lsq()
#lreg = opt()
lreg.fita(Amat, y)




if dim==1:
    xg = np.linspace(-1., 1., 100).reshape(-1, 1)
    yg = true_model(xg)
    Agr = pcrv.evalBases(xg, 0)
    ygr_pred, ygr_pred_var, _ = lreg.predicta(Agr, msc=1)
    print(ygr_pred_var)
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
    plt.savefig('fit.png')
    plt.clf()

lreg.predict_plot([Amat], [y], labels=['Training'], colors=None)
