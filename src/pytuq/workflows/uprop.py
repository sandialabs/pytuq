#!/usr/bin/env python
"""Module for uncertainty propagation workflows via polynomial chaos."""

import numpy as np

def uprop_proj(in_pc, model, nqd, out_pc):
    """Uncertainty propagation via spectral projection.

    Evaluates the model on quadrature points induced by the input PC germ,
    and computes the output PC coefficients via Galerkin projection.

    Args:
        in_pc (PCRV): Input PC random variable.
        model (callable): Model function taking an :math:`(N, d)`-sized array and returning an :math:`(N, o)`-sized array.
        nqd (int): Number of quadrature points per stochastic dimension.
        out_pc (PCRV): Output PC random variable (coefficients are set in-place).
    """
    dim = in_pc.sdim
    qdpts, wghts = in_pc.quadGerm([nqd]*dim)


    qdpts_pc = in_pc.evalPC(qdpts)
    yqdpts = model(qdpts_pc)


    npc = out_pc.mindices[0].shape[0]

    assert(yqdpts.shape[1]==out_pc.pdim)
    out_cfs_all=[]
    for odim in range(out_pc.pdim):
        normsq = out_pc.evalBasesNormsSq(odim)
        bases = out_pc.evalBases(qdpts, odim)
        out_cfs = np.empty(npc)
        for i in range(npc):
            out_cfs[i] = np.dot(yqdpts[:, odim]*wghts, bases[:, i])/normsq[i]
        out_cfs_all.append(out_cfs)
    out_pc.setCfs(out_cfs_all)

    return

def uprop_regr(in_pc, model, nsam, out_pc):
    """Uncertainty propagation via regression.

    Evaluates the model on random germ samples from the input PC and computes
    the output PC coefficients via least-squares regression.

    Args:
        in_pc (PCRV): Input PC random variable.
        model (callable): Model function taking an :math:`(N, d)`-sized array and returning an :math:`(N, o)`-sized array.
        nsam (int): Number of random samples.
        out_pc (PCRV): Output PC random variable (coefficients are set in-place).
    """
    dim = in_pc.sdim
    samples_germ = in_pc.sampleGerm(nsam)

    samples_pc = in_pc.evalPC(samples_germ)
    ysamples = model(samples_pc)

    assert(ysamples.shape[1]==out_pc.pdim)
    out_cfs_all=[]
    for odim in range(out_pc.pdim):
        Amat = out_pc.evalBases(samples_germ, odim)
        out_cfs  = np.linalg.lstsq(Amat, ysamples[:, odim], rcond=None)[0]
        #print(out_cfs.shape)
        out_cfs_all.append(out_cfs)

    out_pc.setCfs(out_cfs_all)

    return
