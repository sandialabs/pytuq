#!/usr/bin/env python
"""Routines that are wrappers of UQTk apps' system calls."""


import os
import numpy as np

def pce_eval(xdata, pctype, mi, pccf):
    """Evaluates Polynomial Chaos (PC) expansion.

    Args:
        xdata (np.ndarray): 2d input array of size `(N,d)`.
        pctype (str): PC type.
        mi (np.ndarray): 2d integer array of multiindex, of size `(K,d)`.
        pccf (np.ndarray): 1d array of coefficients of size `K`.

    Returns:
        np.ndarray: 1d array of PC outputs of size `N`.
    """
    uqtkbin = os.environ['UQTK_INS'] + os.sep + 'bin'
    np.savetxt('mi', mi, fmt="%d")
    np.savetxt('pccf', pccf)
    np.savetxt('xdata.dat', xdata)

    cmd = uqtkbin + os.sep + 'pce_eval -x PC_mi -r mi -f pccf -s ' + pctype + ' > pceval.log'
    os.system(cmd)


    ydata = np.loadtxt('ydata.dat')

    return ydata


####################################################################


def pce_sens(pctype, mi, pccf, mv=False):
    """Evaluate input sensitivities of a PC expansion.

    Args:
        pctype (str): PC type.
        mi (np.ndarray): 2d integer array of multiindex, of size `(K,d)`.
        pccf (np.ndarray): 1d array of coefficients of size `K`.
        mv (bool, optional): Whether to return mean and variance as well

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): tuple of main (1d array of size `d`), total (1d array of size `d`) and joint (2d array of size `(d,d)`) sensitivities. Optionally, extra two elements are returned, mean and variance.
    """
    uqtkbin = os.environ['UQTK_INS'] + os.sep + 'bin'
    np.savetxt('mi', mi, fmt="%d")
    np.savetxt('pccf', pccf)

    cmd = uqtkbin + os.sep + 'pce_sens -m mi -f pccf -x ' + pctype + ' > pcsens.log'
    os.system(cmd)

    mainsens = np.loadtxt('mainsens.dat')
    totsens = np.loadtxt('totsens.dat')
    jointsens = np.loadtxt('jointsens.dat')
    varfrac = np.atleast_1d(np.loadtxt('varfrac.dat'))

    if (mv):
        mean = pccf[0]
        var = mean**2 / varfrac[0]

        return mainsens, totsens, jointsens, mean, var

    else:
        return mainsens, totsens, jointsens


####################################################################

def get_pdf_uqtk(data,target, verbose=1):
    """
    Compute PDF given data at target points with the UQTk app.

    Args:
        data (np.ndarray): `(N,d) array of N samples in d dimensions.
        target (np.ndarray):  `(M,d) array of target points. Can be an integer, in which case it is interpreted as the number of grid points per dimension for a target grid.
        verbose (int, optional): Verbosity on the screen, 0, 1, or 2. Defaults to 1.

    Returns:
        np.ndarray: target points (same as target, or a grid, if target is an integer).
        np.ndarray: PDF values at xtarget.
    """
    np.savetxt('data',data)
    assert(np.prod(np.var(data, axis=0))>0.0)

    # Wrapper around the UQTk app

    if (verbose>1):
        outstr=''
    else:
        outstr=' > pdfcl.log'

    if(type(target)==int):
        cmd='pdf_cl -i data -g '+str(target)+outstr
        if (verbose>0):
            print('Running %s'&(cmd))
        os.system(cmd)

    else:
        np.savetxt('target',target)
        cmd='pdf_cl -i data -x target'+outstr
        if (verbose>0):
            print('Running %s' % (cmd))

        os.system(cmd)

    xtarget=np.loadtxt('dens.dat')[:,:-1]
    dens=np.loadtxt('dens.dat')[:,-1]


    # Return the target points and the probability density
    return xtarget,dens
