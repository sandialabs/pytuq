#!/usr/bin/env python
"""Example demonstrating 2D function slicing and visualization.

This script shows how to plot 2D slices of functions at various anchor points,
useful for exploring function behavior around specific parameter values.
"""

import numpy as np

from pytuq.utils.plotting import myrc, plot_2d_anchored
from pytuq.func import chem


myrc()



def FcnWrapper(fcn):
    assert(fcn.outdim==1)
    def model(x, p):
        return fcn(x)
    return model



#model1 = lambda x,p : x[:,0]**p[0]+np.sin(x[:,1]**p[0])
model1 = FcnWrapper(chem.MullerBrown())
model2 = lambda x,p : np.cos(x[:,0]**p[0])+x[:,1]**p[0]

mb1 = np.array([-0.55822361,  1.44172581]) #np.array([-1.5, 0]) #np.array([-0.55822361,  1.44172581]) #np.zeros(2,)
mb2 = np.array([0.5,        0.03578135]) #np.array([-1.5, 0.2])#np.array([0.5,        0.03578135])
mb3 = np.array([0.5, 1])

# plot_1d_anchored([model1, model1], [[3], [3]], mb1, scale=1.,
#               clearax=False, legend_show=True, ncolrow=(3,5))
# plot_1d_anchored([FcnWrapper(chem.MullerBrown())], [None], np.zeros(2,), scale=1.,
#               clearax=False, legend_show=True, ncolrow=(3,5))
# plot_1d_anchored_single([model1, model1], [[3], [3]], mb1, anchor2=mb2, verbose_labels=True, clearax=False, figname='fcn_1dslice.png')



# plot_2d_anchored_single([model1, model1], [[2], [3]], mb1, anchor2=None, anchor3=None, scale=1.0,
#              squished=False, colorful=False, clearax=False,
#              pad=0.01, figname='fcn_2dslice.png')
plot_2d_anchored([model1, model2], [[2], [3]], mb1, anchor2=None, scale=1.0,
             squished=False, colorful=False, clearax=False,
             pad=0.01, ncolrow=(3,5))
