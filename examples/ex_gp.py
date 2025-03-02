#!/usr/bin/env python
"""A Gaussian Process fit example."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from pytuq.fit.gp import gp
from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.utils.funcbank import sin4
from pytuq.utils.maps import scale01ToDom
from pytuq.utils.plotting import myrc, lighten_color, plot_vars, plot_dm

myrc()

########################################################
########################################################
########################################################


########################################################
########################################################
########################################################

########################################################
########################################################
########################################################

# 1D example
ntrn = 11
ntst = 33
true_model = sin4
dim = 1
datastd = 0.02

domain = np.ones((dim, 2))*np.array([-1.,1.])

order = 4
pcrv = PCRV(1, dim, 'LU', mi=get_mi(order, dim))

def basisevaluator(x, pars):
    pcrv, = pars
    return pcrv.evalBases(x, 0)


x = scale01ToDom(np.random.rand(ntrn,dim), domain)
y = true_model(x)+datastd*np.random.randn(ntrn)

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ExpSineSquared
# gpr = GaussianProcessRegressor(kernel=RBF(1.0)+ WhiteKernel(1e-1))

# gpr = gp('sin', (0.1427, 2.3), kernel_params_range=[[0.01,10.], [0.01,10.]], nugget=1.e-12, sigma2=None)
gpr = gp('RBF', (0.1,), nugget=1.e-12, sigma2=None,
          kernel_params_range=[[0.01,1.0]], sigma2prior=[0.0, 0.0],
          basis = [basisevaluator, (pcrv,)])

# res = minimize(gpr.neglogmarglik, [1.0], args=(x,y), method='L-BFGS-B', jac=None, tol=None, bounds=[[0.1, 10.]], callback=print, options=None)

# print("=================================")
# print(res.x)
# sys.exit()

gpr.fit(x, y)
print("Kernel parameters:", gpr.kernel_params)
#y_pred, y_std = gpr.predict(x, return_std=True)
y_pred, y_std = gpr.predict_wstd(x)

agrid = np.linspace(0.01, 1., 133)
plt.plot(agrid, np.array([gpr.neglogmarglik([aa], x, y) for aa in agrid]))
plt.ylim(ymin=-10., ymax=100.)
plt.savefig('neglogmarglik.png')


xtst = scale01ToDom(np.random.rand(ntst,dim), domain)
ytst = true_model(xtst)+datastd*np.random.randn(ntst)

#ytst_pred, ytst_predstd = gpr.predict(xtst, return_std=True)
ytst_pred, ytst_predstd = gpr.predict_wstd(xtst)


plot_dm([y, ytst], [y_pred, ytst_pred],
        errorbars=[[y_std,y_std], [ytst_predstd,ytst_predstd]],
        labels=['Training', 'Testing'],
        axes_labels=['Data', 'GP Fit'], figname='dm.png', msize=6)

if dim==1:
    ngr = 111
    xgrid = np.linspace(domain[0,0]-0., domain[0,1]+0., ngr)[:, np.newaxis]
    ygrid = true_model(xgrid)
    #ygrid_pred, ygrid_predstd = gpr.predict(xgrid, return_std=True)
    ygrid_pred, ygrid_predstd = gpr.predict_wstd(xgrid)

    plt.plot(x[:,0], y, 'ko', ms=11, label='Data', zorder=1000)
    plt.plot(xgrid[:,0], ygrid, 'k--', label='True model', zorder=1000)
    plot_vars(xgrid[:,0], ygrid_pred,
              variances=ygrid_predstd[:,np.newaxis]**2,
              varlabels=['Fit stdev'],
              varcolors=[lighten_color('m', 0.5)],
              ax=plt.gca())

    plt.legend()
    plt.savefig('gpfit.png')

