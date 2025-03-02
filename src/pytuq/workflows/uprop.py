#!/usr/bin/env python

import numpy as np

def uprop_proj(in_pc, model, nqd, out_pc):
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
