#!/usr/bin/env python
"""Various routines dealing with multiindices."""

import numpy as np


def get_npc(ord, dim):
    """Get number of polynomial basis with a total-degree truncation.

    Args:
        ord (int): Order `p`
        dim (int): Dimension `d`

    Returns:
        int: Number of bases with order up to `p` and dimension `d`, i.e. `(p+d)!/p!d!`.
    """
    npc = 1

    for i in range(ord):
        npc = npc * (dim + i + 1)
    for i in range(ord):
        npc = npc / (i + 1)

    assert(npc==int(npc))

    return int(npc)


def get_mi(ord, dim):
    """Get multiindex array with a total-degree truncation.

    Args:
        ord (int): Order `p`
        dim (int): Dimension `d`

    Returns:
        int np.ndarray: Multiindex array of size `(K,d)`, where `K=(p+d)!/p!d!`.
    """

    assert(dim>0)
    npc = get_npc(ord, dim)
    ic = np.ones(dim, dtype='int')
    iup = 0
    mi = np.zeros((npc, dim), dtype='int')
    if (ord > 0):
        #: first order terms
        for idim in range(dim):
            iup += 1
            mi[iup, idim] = 1
    if (ord > 1):
        #: higher order terms
        for iord in range(2, ord + 1):
            lessiord = iup
            for idim in range(dim):
                for ii in range(idim + 1, dim):
                    ic[idim] += ic[ii]
            for idimm in range(dim):
                for ii in range(lessiord - ic[idimm] + 1, lessiord + 1):
                    iup += 1
                    mi[iup] = mi[ii].copy()
                    mi[iup, idimm] += 1
    return mi


def encode_mindex(mindex):
    """Encodes the multiindex into a list of dimension-order pairs.

    Args:
        mindex (np.ndarray): Integer 2d array of multiindices.

    Returns:
        list[tuple]: List of tuples, where each tuple contains two lists, dimension list and order list, corresponding to a multiindex.

    Note:
        This is convenient for sparse and high-dimensional multiindices, for readability and for analysis.
    """
    npc, ndim = mindex.shape
    print(f"Multiindex has {npc} terms")
    dims = []
    ords = []
    for ipc in range(npc):
        nzs = np.nonzero(mindex[ipc, :])[0].tolist()
        if len(nzs) == 0:
            dims.append([0])
            ords.append([0])
        else:
            dims.append([i + 1 for i in nzs])
            ords.append(mindex[ipc, nzs].tolist())
#            this_sp_mindex=np.vstack((nzs+1,mindex[ipc,nzs])).T
 #           this_sp_mindex=this_sp_mindex.reshape(1,-1)

        # effdim=len(np.nonzero(mindex[ipc,:])[0])
        # print effdim
  #      sp_mindex.append(this_sp_mindex)
    #return dims, ords  # sp_mindex
    return list(zip(dims,ords))


def micf_join(mindex_list, cfs_list):
    """Merge a list of multiindices and corresponding coefficients.
    TODO: what happens to common coefficients?

    Args:
        mindex_list (list[np.ndarray]): List of 2d multiindex arrays.
        cfs_list (list[np.ndarray]): List of 1d coefficient arrays

    Returns:
        (np.ndarray, np.ndarray): A tuple of multiindex and coefficient lists.
    """
    nout = len(mindex_list)
    assert(nout == len(cfs_list))

    mindex_list_ = mindex_list.copy()
    cfs_list_ = cfs_list.copy()

    mindex0 = np.zeros((1, mindex_list_[0].shape[1]), dtype=int)
    cfs0 = np.zeros((1,))

    mindex_list_.append(mindex0)
    cfs_list_.append(cfs0)
    ### Get common set of multiindex and coefficients
    mindex_all = np.unique(np.concatenate(mindex_list_), axis=0)
    npc = mindex_all.shape[0]

    cfs_all = np.zeros((nout, npc))
    for j in range(nout):
        for k in range(npc):
            bb = np.sum(np.abs(mindex_list_[j]-mindex_all[k, :]), axis=1)
            ind = np.where(bb==0)[0]
            if len(ind) > 0:
                cfs_all[j, k] = cfs_list_[j][ind[0]]

    return mindex_all, cfs_all


