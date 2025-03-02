#!/usr/bin/env python

import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.rv.rosen import Rosenblatt


from pytuq.lreg.bcs import bcs
from pytuq.lreg.anl import anl
from pytuq.lreg.lreg import lsq
from pytuq.utils.plotting import plot_samples_pdfs

def pc_fit(x, y, order=3, pctype='LU', method='anl', **kwargs):
    nsam, ndim = x.shape
    nsam_, nout = y.shape
    assert(nsam==nsam_)

    mindex=get_mi(order, ndim)
    pcrv = PCRV(nout, ndim, pctype, mi=mindex)

    Amat = pcrv.evalBases(x, 0)

    #TODO: To impl. quadrature, see ex_uprop.py

    mindices_list=[]
    cfs_list = []
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

        lreg.fita(Amat, y[:,iout])
        mindices_list.append(mindex[lreg.used, :])
        cfs_list.append(lreg.cf)


        #Amat_ = Amat[:, lreg.used]
        #ypred[:,iout], _, _ = lreg.predicta(Amat_)


    pcrv.setMiCfs(mindices_list, cfs_list)



    #pcrv.printInfo()
    #print(pcrv.computeMean())
    # print(pcrv.coefs)

    pcrv.setFunction()


    return pcrv


# Create PC given samples
def pc_ros(xsam, pctype='HG', order=1, nreg=13, n_pcsam=None, bwfactor=1.0):
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
