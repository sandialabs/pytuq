#!/usr/bin/env python
"""Module for PC fitting workflows."""

import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.rv.rosen import Rosenblatt


from pytuq.lreg.bcs import bcs
from pytuq.lreg.anl import anl
from pytuq.lreg.lreg import lsq
from pytuq.utils.plotting import plot_samples_pdfs

def pc_fit(x, y, order=3, pctype='LU', method='anl', **kwargs):
    """Fit a PC surrogate to input-output data via regression.

    For each output dimension, constructs a basis matrix from the PC representation
    and fits coefficients using the chosen linear regression method.

    Args:
        x (np.ndarray): A 2d input array of size :math:`(N, d)`.
        y (np.ndarray): A 2d output array of size :math:`(N, o)`.
        order (int, optional): PC order. Defaults to 3.
        pctype (str, optional): PC type, e.g. ``'LU'`` or ``'HG'``. Defaults to ``'LU'``.
        method (str, optional): Fitting method: ``'anl'``, ``'bcs'``, or ``'lsq'``. Defaults to ``'anl'``.
        **kwargs: Additional keyword arguments passed to the regression method (e.g. ``eta`` for BCS).

    Returns:
        tuple[PCRV, list]: A pair of the fitted PCRV object and a list of regression objects.
    """
    nsam, ndim = x.shape
    nsam_, nout = y.shape
    assert(nsam==nsam_)

    mindex=get_mi(order, ndim)
    pcrv = PCRV(nout, ndim, pctype, mi=mindex)


    #TODO: To impl. quadrature, see ex_uprop.py

    mindices_list=[]
    cfs_list = []
    lregs = []
    ypred = np.zeros_like(y)
    for iout in range(nout):
        print(f"Fitting output {iout+1} / {nout}")
        if method == 'bcs':
            if 'eta' in kwargs:
                lreg = bcs(eta=kwargs['eta'])
            else:
                lreg = bcs()
        elif method == 'anl':
            lreg = anl()
        elif method == 'lsq':
            lreg = lsq()

        Amat = pcrv.evalBases(x, iout)

        lreg.fita(Amat, y[:,iout])
        mindices_list.append(mindex[lreg.used, :])
        cfs_list.append(lreg.cf)
        lregs.append(lreg)

        #Amat_ = Amat[:, lreg.used]
        #ypred[:,iout], _, _ = lreg.predicta(Amat_)


    pcrv.setMiCfs(mindices_list, cfs_list)



    #pcrv.printInfo()
    #print(pcrv.computeMean())
    # print(pcrv.coefs)

    pcrv.setFunction()


    return pcrv, lregs


def pc_ros(xsam, pctype='HG', order=1, nreg=13, n_pcsam=None, bwfactor=1.0):
    """Create a PC representation from samples using a Rosenblatt map.

    Builds a Rosenblatt map from the input samples, generates germ samples,
    evaluates the inverse Rosenblatt, and fits PC coefficients per dimension.
    Optionally resamples from the resulting PC and plots comparison PDFs.

    Args:
        xsam (np.ndarray): A 2d sample array of size :math:`(N, d)`.
        pctype (str, optional): PC type, e.g. ``'HG'`` or ``'LU'``. Defaults to ``'HG'``.
        order (int, optional): PC order. Defaults to 1.
        nreg (int, optional): Number of regression points. Defaults to 13.
        n_pcsam (int or None, optional): Number of PC samples to draw. If None, uses `N`. Defaults to None.
        bwfactor (float, optional): Bandwidth scaling factor for the Rosenblatt map. Defaults to 1.0.

    Returns:
        PCRV: The fitted PCRV object.
    """
    nsam, odim = xsam.shape
    # Unless we know how sampling is done, we always take sdim==odim
    sdim = odim + 0

    # Create a Rosenblatt map
    ros = Rosenblatt(xsam, bwfactor=bwfactor)
    print("Bandwidths:", ros.sigmas)

    # Create the PC object
    pcrv = PCRV(odim, sdim, pctype, mi=get_mi(order, sdim))

    # Sample uniform through the germ
    germ_sam = pcrv.sampleGerm(nreg)
    unif_sam = np.zeros((nreg, sdim))
    for idim in range(sdim):
        unif_sam[:, idim] = pcrv.PC1ds[idim].germCdf(germ_sam[:, idim])

    # Evaluate inverse Rosenblatt as regression training data
    xreg = np.array([ros.inv(u) for u in unif_sam])

    # Regression fit of inverse Rosenblatt function
    all_cfs=[]
    for idim in range(odim):
        Amat = pcrv.evalBases(germ_sam, idim)
        lreg = anl()
        lreg.fita(Amat, xreg[:, idim])
        all_cfs.append(lreg.cf)


    # Resample PC
    pcrv.setCfs(all_cfs)
    if n_pcsam is None:
        n_pcsam = nsam

    if n_pcsam>0:
        pcsam = pcrv.sample(n_pcsam)
        np.savetxt('pcsam.txt', pcsam)

    # Plot PDFs for comparison
    plot_samples_pdfs([xsam, xreg, pcsam],
                      legends=['Orig. Samples', r'R$^{-1}$(U) Samples', 'PC Samples'],
                      colors=['b', 'g', 'r'],
                      file_prefix='xpdfpc')

    return pcrv
