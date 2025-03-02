#!/usr/bin/env python
"""Classes for various multivariate random variables."""


import sys
import functools
import numpy as np
from scipy.special import erf
from scipy.stats import multivariate_normal

from .mrv import MRV
from ..func.func import Function
from ..utils.xutils import cartes_list, safe_cholesky


class PCRV(MRV):
    r"""A class for a multivariate PC random variable.

    Attributes:
        sdim (int): Stochastic dimension :math:`s`, i.e. germ dimensionality.
        mindices (list[np.ndarray]): List of :math:`d` multiindex arrays, each of size :math:`(K_i,s)`.
        coefs (list[np.ndarray]): List of :math:`d` coefficient arrays, each of size :math:`K_i` for :math:`i=1,\dots,d`.
        maxOrd (np.ndarray): A 1d array of size :math:`s` indicating maximal order across all physical dimensions.
        pind (list[tuple]): List of tuples :math:`(i,k)` accounting for all coefficients: the first element is the physical dimension :math:`i` (between :math:`0` and :math:`d-1`), and the second element is the coefficient index :math:`k` (between :math:`0` and :math:`K_i`).
        rndind (list[int]): Indices of random physical dimensions, each entry is between :math:`0` and :math:`d-1`.
        detind (list[int]): Indices of deterministic physical dimensions, each entry is between :math:`0` and :math:`d-1`.
        function (callable): PC evaluator function, :math:`(N,s)\rightarrow N`.
        pctypes (list[str]): List of :math:`s` PC types, one for each stochastic dimension.
        PC1ds (list[PC1d]): List of :math:`s` 1d PC objects that comprise this multivariate PC random variable.
    """

    def __init__(self, pdim, sdim, pctype, mi=None, cfs=None):
        r"""Initialization.

        Args:
            pdim (int): Physical dimensionality :math:`d` of the PC random variable/vector.
            sdim (int): Stochastic dimensionality :math:`s` of the PC random variable/vector.
            pctype (str or list): PC type. Either a list of :math:`s` strings (one per stochastic dimension), or a single string for all dimensions.
            mi (list or np.ndarray, optional): List of :math:`d` multiindex arrays, each of size :math:`(K_i,s)` for :math:`i=1, \dots, d`. Or a single multiindex array of size :math:`(K,s)`, meaning all dimensions get the same multiindex. Defaults to None, which is a single 1d constant random variable i.e. a multiindex of all zeros.
            cfs (list or np.ndarray, optional): List of :math:`d` coefficient arrays, each of size :math:`K_i` for :math:`i=1, \dots, d`. Or a single coefficient array of size :math:`K`, meaning all dimensions get the same coefficient array. Or a 2d array of size :math:`(K,d)`. Defaults to None, which is populating coefficients with all zeros.
        """
        assert(pdim>0)
        super().__init__(pdim)
        self.function = None

        assert(sdim>0)
        if mi is None:
            mi = np.zeros((1, sdim), dtype=int)

        self.setMiCfs(mi, cfs=cfs)

        if isinstance(pctype, str):
            self.pctypes = [pctype] * self.sdim
        elif isinstance(pctype, list):
            self.pctypes = pctype
            # if len(self.indz)>0:
            #     del self.pctypes[self.indz]
            #     self.indz=[]
        else:
            print(f"PC type {pctype} type is not recognized. Exiting.")
            sys.exit()
        assert(len(self.pctypes)==self.sdim)

        self.getParamIndices() # TODO: will we ever need the inverse of getParamIndices()? if so, either do list search or better write a special function
        self.getMaxOrders()

        self.PC1ds = [PC1d(pt) for pt in self.pctypes]


    def __repr__(self):
        return f"{self.pctypes} PC Random Variable(pdim={self.pdim}, sdim={self.sdim})"

    def setMiCfs(self, mi, cfs=None):
        r"""Sets the multiindex and coefficients together.

        Args:
            mi (list or np.ndarray): List of :math:`d` multiindex arrays, each of size :math:`(K_i,s)` for :math:`i=1, \dots, d`. Or a single multiindex array of size :math:`(K,s)`, meaning all dimensions get the same multiindex.
            cfs (list or np.ndarray, optional): List of :math:`d` coefficient arrays, each of size :math:`K_i` for :math:`i=1, \dots, d`. Or a single coefficient array of size :math:`K`, meaning all dimensions get the same coefficient array. Or a 2d array of size :math:`(d, K)`. Defaults to None, which is populating coefficients with all zeros.
        """
        self.setMi(mi)
        self.setCfs(cfs=cfs)
        self.checkMiCfsizes()

    def setMi(self, mi):
        r"""Sets the multiindex.

        Args:
            mi (list or np.ndarray): List of :math:`d` multiindex arrays, each of size :math:`(K_i,s)` for :math:`i=1, \dots, d`. Or a single multiindex array of size :math:`(K,s)`, meaning all dimensions get the same multiindex.

        Note:
            Dangerous to use externally, as it may conflict with other attributes/sizes. Prefer to use setMiCfs() externally.
        """
        if isinstance(mi, np.ndarray):
            assert(len(mi.shape) == 2)
            self.mindices = [mi] * self.pdim
        elif isinstance(mi, list):
            assert(len(mi) == self.pdim)
            self.mindices = mi.copy()
        else:
            print(f"Multiindex {mi} type is not recognized. Exiting.")
            sys.exit()

        self.sdim = self.mindices[0].shape[1]
        self.detind = []
        self.rndind = []
        ss = np.zeros((self.sdim,), dtype=int)
        for i in range(self.pdim):
            morders = np.sum(self.mindices[i],axis=0)
            ss += morders
            if np.sum(morders)==0:
                self.detind.append(i)
            else:
                self.rndind.append(i)

        return

    def setCfs(self, cfs=None):
        r"""Sets the coefficients. Dangerous to use externally, as it may conflict with other attributes/sizes. Prefer to use setMiCfs() externally.

        Args:
            cfs (list or np.ndarray): List of :math:`d` coefficient arrays, each of size :math:`K_i` for :math:`i=1, \dots, d`. Or a single coefficient array of size :math:`K`, meaning all dimensions get the same coefficient array. Or a 2d array of size :math:`(d, K)`. Defaults to None, which is populating coefficients with all zeros.

        Note:
           Dangerous to use externally, as it may conflict with other attributes. Prefer to use setMiCfs() externally.
        """
        if cfs is None:
            self.coefs = [np.zeros((mind.shape[0],)) for mind in self.mindices]
        elif isinstance(cfs, np.ndarray):
            if len(cfs.shape)==1:
                self.coefs = [cfs] * self.pdim
            elif len(cfs.shape)==2:
                assert(cfs.shape[0] == self.pdim)
                self.coefs = [cfs[j, :] for j in range(self.pdim)]
            else:
                print(f"Wrong shape {cfs.shape} of coefs array. Exiting.")
                sys.exit()
        elif isinstance(cfs, list):
            assert(len(cfs) == self.pdim)
            self.coefs = cfs.copy()
        else:
            print(f"Coefs {cfs} type is not recognized. Exiting.")
            sys.exit()


        return

    def checkMiCfsizes(self):
        """Checks the multiindex and coeffient list for any size incompatibility.
        """
        assert(len(self.mindices) == len(self.coefs))
        # Size checks
        for i in range(self.pdim):
            assert(isinstance(self.mindices[i], np.ndarray))
            assert(isinstance(self.coefs[i], np.ndarray))
            assert(len(self.mindices[i].shape) == 2)
            assert(len(self.coefs[i].shape) == 1)

            assert(self.mindices[i].shape[1] == self.sdim)
            #print(i, self.mindices[i].shape, self.coefs[i].shape)
            assert(self.mindices[i].shape[0] == self.coefs[i].shape[0])

        return


    def setRandomCfs(self):
        r"""Sets coefficients randomly, sampling unniformly in :math:`[0,1]`.
        """
        cfs_list = []
        for i in range(self.pdim):
            cfs_list.append(np.random.rand(self.mindices[i].shape[0],))

        self.setCfs(cfs=cfs_list)

        return

    def getMaxOrders(self):
        """Computes and populates an internal array maxOrd storing maximal order per physical dimension.
        """
        maxorders = np.empty((self.pdim, self.sdim), dtype=int)
        for j in range(self.pdim):
            maxorders[j, :] = np.max(self.mindices[j], axis=0)
        self.maxOrd = np.max(maxorders, axis=0)

        return

    def getParamIndices(self):
        """Populates the bookkeeping list of index pairs pind.
        """
        self.pind = []
        for i in range(self.pdim):
            for ipc in range(self.coefs[i].shape[0]):
                self.pind.append((i, ipc))

        return

    def printInfo(self):
        """Core dump of multiindices and coefficients.
        """
        for i in range(self.pdim):
            print(f"{self.mindices[i]} {self.coefs[i]}")

    def computeMean(self):
        r"""Computes the mean of the random variable.

        Returns:
            np.ndarray: A 1d array of size :math:`d` for means per physical dimension.
        """
        mean = np.zeros((self.pdim))
        for i in range(self.pdim):
            totords = np.sum(self.mindices[i], axis=1)
            indOfZeroMI = np.where(totords == 0)[0]
            for jj in indOfZeroMI:
                mean[i] += self.coefs[i][jj]

        return mean

    def computeVar(self):
        r"""Computes the variance of the random variable.

        Returns:
            np.ndarray: A 1d array of size :math:`d` for variance per physical dimension.
        """
        var = np.zeros((self.pdim))
        for i in range(self.pdim):
            normsq = self.evalBasesNormsSq(i)
            totords = np.sum(self.mindices[i], axis=1)
            indOfNonZeroMI = np.where(totords > 0)[0]
            for jj in indOfNonZeroMI:
                var[i] += self.coefs[i][jj]**2 * normsq[jj]

        return var

    def computeSens(self):
        r"""Computes main Sobol sensitivity indices of the PC with respect to stochastic dimensions.

        Returns:
            np.ndarray: A 2d array of main sensitivities of size :math:`(d,s)`.
        """
        var = self.computeVar()

        mainsens = np.zeros((self.pdim, self.sdim))
        for i in range(self.pdim):
            if var[i] == 0.0:
                var[i] += 1.e-12

            normsq = self.evalBasesNormsSq(i)
            for j in range(self.sdim):
                indOfOnlyj = np.where(np.sum(np.delete(self.mindices[i], j, axis=1), axis=1)==0)[0]

                for jj in indOfOnlyj:
                    if self.mindices[i][jj, j]>0:
                        mainsens[i,j] += self.coefs[i][jj]**2 * normsq[jj]

                mainsens[i,j] /= var[i]

        return mainsens

    def computeTotSens(self):
        r"""Computes total Sobol sensitivity indices of the PC with respect to stochastic dimensions.

        Returns:
            np.ndarray: A 2d array of total sensitivities of size :math:`(d,s)`.
        """
        var = self.computeVar()

        totsens = np.zeros((self.pdim, self.sdim))
        for i in range(self.pdim):
            if var[i] == 0.0:
                var[i] += 1.e-12

            normsq = self.evalBasesNormsSq(i)
            for j in range(self.sdim):
                for jj in range(self.mindices[i].shape[0]):
                    if self.mindices[i][jj, j]>0:
                        totsens[i,j] += self.coefs[i][jj]**2 * normsq[jj]

                totsens[i,j] /= var[i]


        return totsens

    def computeJointSens(self):
        r"""Computes joint Sobol sensitivity indices of the PC with respect to stochastic dimension pairs.

        Returns:
            np.ndarray: A 3d array of joint sensitivities of size :math:`(d,s,s)`.
        """
        var = self.computeVar()

        jointsens = np.zeros((self.pdim, self.sdim, self.sdim))
        for i in range(self.pdim):
            if var[i] == 0.0:
                var[i] += 1.e-12

            normsq = self.evalBasesNormsSq(i)
            for j in range(self.sdim):
                for k in range(j+1, self.sdim):
                    for jj in range(self.mindices[i].shape[0]):
                        if self.mindices[i][jj, j]*self.mindices[i][jj, k]>0:
                            jointsens[i,j,k] += self.coefs[i][jj]**2 * normsq[jj]

                    jointsens[i,j,k] /= var[i]
                    jointsens[i,k,j] = jointsens[i,j,k]+0.0


        return jointsens

    def computeGroupSens(self, paramIndices):
        r"""Computes group sensitivities of a subset of parameters.

        Args:
            paramIndices (list): List of indices to group. Each element should be between :math:`0` and :math:`s-1`.

        Returns:
            np.ndarray: An 1d array of size `d` for sensitivities of this group for all :math:`d` dimensions.
        """
        var = self.computeVar()
        nind = len(paramIndices)
        groupsens = np.zeros((self.pdim, ))


        for i in range(self.pdim):
            if var[i] == 0.0:
                var[i] += 1.e-12

            normsq = self.evalBasesNormsSq(i)

            indOfOnlyGroup = np.where(np.sum(np.delete(self.mindices[i], paramIndices, axis=1), axis=1)==0)[0]
            for jj in indOfOnlyGroup:
                if np.sum(self.mindices[i][jj, :])>0:
                    groupsens[i] += self.coefs[i][jj]**2 * normsq[jj]

            groupsens[i] /= var[i]

        return groupsens
    
    def sampleGerm(self, nsam=1):
        r"""Sample PC germ vector.

        Args:
            nsam (int, optional): Number of samples requested. Defaults to :math:`M=1`.

        Returns:
            np.ndarray: A 2d array of size :math:`(M,s)`.
        """
        germSam = np.empty((nsam, self.sdim))
        for i in range(self.sdim):
            germSam[:, i] = self.PC1ds[i].sample(nsam)

        return germSam

    def quadGerm(self, pts=None):
        r"""Generates quadrature samples of PC germ vector.

        Args:
            pts (np.ndarray, optional): An integer 1d array of size :math:`s` indicating how many points per each stochastic dimension, :math:`q_i` for :math:`i=1, \dots, s`. Default is None, which means 2 points per stochastic dimension.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair of arrays: a 2d array of quadrature points of size :math:`(Q,s)` and corresponding 1d array of weights of size :math:`Q`, where :math:`Q=q_1 q_2 \cdots q_s` is the total number of points.

        Note:
            This is full tensor product quadrature. Sparse quadrature is not implemented.
        """
        if pts is None:
            pts = [2] * self.sdim

        assert(len(pts)==self.sdim)

        quad1ds = []
        wght1ds = []
        for i in range(self.sdim):
            qdpts, wghts = self.PC1ds[i].quad(pts[i])
            quad1ds.append(qdpts)
            wght1ds.append(wghts)

        wQuad = functools.reduce(np.outer, wght1ds).ravel()

        germQuad = np.array(cartes_list(quad1ds))
        # if self.sdim==1:
        #     # somehow cartes_list needs a transpose for 1d
        #     germQuad = germQuad.T

        return germQuad, wQuad


    def evalBases(self, xi, jdim):
        r"""Evaluation of PC bases at given input germ values for a given physical dimension.

        Args:
            xi (np.ndarray): A 2d array of size :math:`(M,s)` for the input.
            jdim (int): The index of :math:`i` of the PC random variable/vector. Should be between :math:`0` and :math:`d-1`.

        Returns:
            np.ndarray: A 2d output array of size :math:`(M, K_i)` where :math:`K_i` is the number of PC bases for the :math:`i`-th dimension.
        """
        nxi, xidim = xi.shape
        assert(xidim==self.sdim)
        assert(jdim>=0 and jdim<self.pdim)
        mindex = self.mindices[jdim]
        npc, sdim_ = mindex.shape
        assert(sdim_==self.sdim)
        ybases = np.empty((nxi, npc))
        # prerun and save
        pcs_list=[]
        for i in range(self.sdim):
            pcs_list.append(self.PC1ds[i](xi[:, i], self.maxOrd[i]))

        for k in range(npc):
            prd = 1.0
            for i in range(self.sdim):
                prd *= pcs_list[i][mindex[k,i]]

            ybases[:, k] = prd

        return ybases

    def evalBasesNormsSq(self, jdim):
        r"""Evaluates bases norms-squared for a given physical dimension.

        Args:
            jdim (int): The index of :math:`i` of the PC random variable/vector. Should be between :math:`0` and :math:`d-1`.

        Returns:
            np.ndarray: An 1d array of size :math:`K_i`, the number of bases for the :math:`i`-th dimension.
        """
        assert(jdim>=0 and jdim<self.pdim)
        mindex = self.mindices[jdim]
        npc, sdim_ = mindex.shape
        assert(sdim_==self.sdim)

        norms = np.empty((npc))
        for k in range(npc):
            prd = 1.0
            for i in range(self.sdim):
                prd *= self.PC1ds[i].normsq(mindex[k,i])

            norms[k] = prd

        return norms


    def evalPC(self, x):
        r"""Evaluate PC expansion for a given set of inputs.

        Args:
            x (np.ndarray): A 2d array of size :math:`(M,s)` for the input.

        Returns:
            np.ndarray: A 2d array of size :math:`(M,d)` for the output.
        """
        assert(self.sdim==x.shape[1])
        nsam = x.shape[0]
        y = np.zeros((nsam, self.pdim))
        # prerun and save
        pcs_list=[]
        for i in range(self.sdim):
            pcs_list.append(self.PC1ds[i](x[:, i], self.maxOrd[i]))

        for j in range(self.pdim):
            mi = self.mindices[j]
            cf = self.coefs[j]
            npc = mi.shape[0]
            val = np.zeros(nsam)
            for k in range(npc):
                prd = cf[k]
                for i in range(self.sdim):
                    prd *= pcs_list[i][mi[k,i]]
                val += prd

            y[:, j] =  val
        return y

    def setFunction(self):
        """Set the PC evaluator as an internal function, an object of a class Function, with all the useful features of the class.
        """
        self.function = Function(name='PC Function')
        domain = np.empty((self.sdim, 2))
        for i in range(self.sdim):
            domain[i, :] = self.PC1ds[i].domain
        self.function.setDimDom(domain=np.clip(domain, -5.0, 5.0))
        self.function.setCall(self.evalPC)

    def sample(self, nsam):
        r"""Sample from the PC random variable. Basically chaining sampling the germ and evaluating the PC.

        Args:
            nsam (int): Number of samples requested, :math:`M`.

        Returns:
            np.ndarray: A 2d array of size :math:`(M,d)` for the output.
        """
        x = self.sampleGerm(nsam)
        return self.evalPC(x)

    def cfsFlatten(self):
        r"""Flatten all the PC coefficients.

        Returns:
            np.ndarray: An 1d array of size :math:`K_1+\dots +K_d`, the total number of PC coefficients for all dimensions.
        """
        cfs_flat = np.concatenate(self.coefs)

        return cfs_flat

    def cfsUnflatten(self, cfs_flat):
        r"""Reverse the flattening operation, given a long flat array, this sets the coefficient array list appropriately.

        Args:
            cfs_flat (np.ndarray):  An 1d array of size :math:`K_1+\cdots +K_d`, the total number of PC coefficients for all dimensions.
        """
        self.coefs = []
        k = 0
        for i in range(self.pdim):
            npc = self.mindices[i].shape[0]
            self.coefs.append(cfs_flat[k:k+npc])
            k += npc

        return

    def compressPC(self):
        """A method to produce a new, compressed PCRV object in case some stochastic dimensions are irrelevant (i.e. 0 order across all physical dimensions)

        Returns:
            PCRV: New PCRV object that has fewer stochastic dimensions.
        """
        indz = np.where(self.maxOrd==0)[0] #indices to remove

        mindices_new = []
        for i in range(self.pdim):
            mm = np.delete(self.mindices[i], indz, axis=1)
            mindices_new.append(mm)

        pctype_new = [j for i,j in enumerate(self.pctypes) if i not in indz]
        pcrv_new = PCRV(self.pdim, self.sdim-len(indz), pctype_new, mi=mindices_new, cfs=self.coefs)

        return pcrv_new

    def compressMI(self):
        """A method to compress potentially identical multiindex rows, by adding their corresponding coefficients.
        """

        for i in range(self.pdim):
            mi = self.mindices[i]
            #np.unique(mi, )

            mm, inds = np.unique(mi, return_inverse=True, axis=0)
            newnpc = mm.shape[0]
            cc = np.zeros((newnpc,))
            for j in range(mi.shape[0]):
                cc[inds[j]] += self.coefs[i][j]

            self.mindices[i] = mm.copy()
            self.coefs[i] = cc.copy()

        return


    def slicePC(self,fixind=None,nominal=None):
        if nominal is None:
            nominal = np.zeros((self.sdim))
        if fixind is None:
            fixind = [] #[i for i in range(self.sdim)]
        assert(len(fixind)<self.sdim)

        newdims = list(set(range(self.sdim))-set(fixind))
        fixdim=len(fixind)
        sdim_new = self.sdim-fixdim
        pctypes_new = [self.pctypes[i] for i in newdims]
        mi_new = [mi[:,newdims] for mi in self.mindices]

        pcs_list=[]
        for i in fixind:
            assert(i<self.sdim)
            pcs_list.append(self.PC1ds[i](nominal[i].reshape(1,1), self.maxOrd[i]))

        cfs_new=[]
        for j in range(self.pdim):
            mi = self.mindices[j]
            cf = self.coefs[j]
            npc = mi.shape[0]
            cf_new = np.zeros((npc,))
            for k in range(npc):
                prd = cf[k]
                for ifix in range(fixdim):
                    prd *= pcs_list[ifix][mi[k,fixind[ifix]]]
                cf_new[k]=prd

            cfs_new.append(cf_new)

        pcrv_new = PCRV(self.pdim, sdim_new, pctypes_new, mi=mi_new, cfs=cfs_new)

        pcrv_new.compressMI()

        return pcrv_new

############################################################
############################################################
############################################################

class PCRV_iid(PCRV):
    """A PC random variable/vector with a special structure of one germ per dimension. As a consequence, the number of stochastic and physical dimensions coincide. It inherits all the attributes of the parent PCRV class.
    """
    def __init__(self, pdim, pctype, orders=None, cfs=None):
        r"""Initialization of the IID PC random variable.

        Args:

            pdim (int): The number of dimensions, :math:`d`. Same as stochastic dimensions, `s`.
            pctype (str or list): PC type. Either a list of :math:`s` strings (one per stochastic dimension), or a single string for all dimensions.
            orders (np.ndarray, optional): An integer array of size :math:`d` indicating the PC order :math:`p_i` for each physical dimension for :math:`i=1,\dots,d`. Defaults to None, which sets order :math:`p_i=1` for all dimensions.
            cfs (list or np.ndarray, optional): List of :math:`d` coefficient arrays, each of size :math:`p_i+1` for :math:`i=1,\dots,d`. Or a single coefficient array of size :math:`p+1`, meaning all dimensions get the same coefficient array (assuming all orders are the same). Or a 2d array of size :math:`(p+1,d)` (again, assuming all orders are the same). Defaults to None, which is populating coefficients with all zeros.
        """
        if orders is None:
            orders = np.ones((pdim,), dtype=int)

        mindices = []
        for ii in range(pdim):
            mindex = np.zeros((orders[ii]+1, pdim), dtype=int)
            mindex[:, ii] = np.arange(orders[ii]+1)
            mindices.append(mindex)
        super(PCRV_iid, self).__init__(pdim, pdim, pctype, mi=mindices, cfs=cfs)

        return

############################################################
############################################################
############################################################

class PCRV_mvn(PCRV):
    """A PC random variable/vector that is a multivariate normal. As a consequence, the number of stochastic and physical dimensions coincide and PC type is Gauss-Hermite for all dimensions. It inherits all the attributes of the parent PCRV class.

    Attributes:
        cov (np.ndarray): Covariance array.
        mean (np.ndarray): Mean array.
    """
    def __init__(self, pdim, rndind=None, mean=None, cov=None):
        r"""Initialization of MVN PC random variable.

        Args:
            pdim (int): The number of dimensions, :math:`d`. Same as stochastic dimensions, :math:`s`.
            rndind (None, optional): List of :math:`r` indices indicating the ones that are random. Each element must be between :math:`0` and :math:`d-1`. Defaults to None, which means all dimensions are random.
            mean (np.ndarray, optional): An array for the mean of size :math:`d`. Defaults to None, which means all zeros.
            cov (np.ndarray, optional): A 2d array of size :math:`(r, r)`. Defaults to None, which is means identity covariance.
        """
        if rndind is None:
            rndind = range(pdim)

        if len(rndind)>0:
            assert(len(rndind)==len(set(rndind)))
            assert(max(rndind)<pdim)
            assert((np.array(rndind)>=0).all())
            assert((np.array(rndind)<pdim).all())
            assert(len(rndind)<=pdim)

        if mean is None:
            self.mean = np.zeros(pdim,)
        else:
            self.mean = mean

        if cov is None:
            self.cov = np.eye(len(rndind))
        else:
            self.cov = cov

        assert(self.mean.shape[0]==pdim)
        assert(self.cov.shape[0]==len(rndind))
        assert(self.cov.shape[1]==self.cov.shape[0])


        lower = safe_cholesky(self.cov)

        mindices = []
        cfs = []
        i=0
        for ii in range(pdim):
            if ii in rndind:
                mindex = np.zeros((i+2, pdim), dtype=int)
                mindex[1:i+2, :i+1] = np.eye(i+1)
                cf = np.zeros((i+2,))
                cf[0] = self.mean[ii]
                cf[1:i+2] = lower[i,:i+1]
                i+=1
            else:
                mindex = np.zeros((1, pdim), dtype=int)
                cf = self.mean[ii]*np.ones((1,))
            mindices.append(mindex)
            cfs.append(cf)

        super(PCRV_mvn, self).__init__(pdim, pdim, "HG", mi=mindices, cfs=cfs)

        return

    def pdf(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

############################################################
############################################################
############################################################

class PC1d():
    r"""1-dimensional PC random variable.

    Attributes:
        a (callable): int->float function :math:`a_n` in the recurrent relation
        b (callable): int->float function :math:`b_n` in the recurrent relation
        domain (np.ndarray): A 1d array of size :math:`2` indicating the domain of definition.
        p0 (callable): The 0th order basis evaluator from 1d np.ndarray to 1d np.ndarray.
        p1 (callable): The 1rd order basis evaluator from 1d np.ndarray to 1d np.ndarray.
        pctype (str): The PC type. Only 'LU' and 'HG' are implemented.
        sample (callable): int->float sampling function, where the input is a number of samples requested, and the output is an 1d array of the corresponding size.
    """

    def __init__(self, pctype='LU'):
        """Initialization.

        Args:
            pctype (str): The PC type. Only 'LU' and 'HG' are implemented. Defaults to 'LU'.
        """
        self.pctype = pctype
        if self.pctype == 'LU':
            self.p0 = lambda x: np.ones_like(x)
            self.p1 = lambda x: x
            self.a = lambda n: (2. * n + 1.) / (n + 1.)
            self.b = lambda n: -n / (n + 1.)
            self.sample = lambda m: 2.*np.random.rand(m)-1.
            self.domain = np.array([-1., 1.])
        elif self.pctype == 'HG':
            self.p0 = lambda x: np.ones_like(x)
            self.p1 = lambda x: x
            self.a = lambda n: 1.
            self.b = lambda n: -n
            self.sample = lambda m: np.random.randn(m)
            self.domain = np.array([-np.inf, np.inf])
        else:
            print(f'PC1d type {self.pctype} is not recognized. Exiting.')
            sys.exit()

    def __call__(self, x, order):
        r"""The function call of the 1d PC object.

        Args:
            x (np.ndarray): A size :math:`M` 1d array of inputs at which PC is evaluated.
            order (int): The requested order :math:`p`.

        Returns:
            list[np.ndarray]: A list of size :math:`p+1` containing 1d arrays of the PC bases evaluated at the requested points.
        """
        if order == 0:
            return [self.p0(x)]
        elif order == 1:
            return [self.p0(x), self.p1(x)]
        else:
            pcvals = [self.p0(x), self.p1(x)]
            for iord in range(2, order + 1):
                pcval = self.a(iord - 1) * x * pcvals[-1] + self.b(iord - 1) * pcvals[-2]
                pcvals.append(pcval)

        return pcvals


    def germCdf(self, x):
        r"""Evaluate the germ cumulative distribution functions (CDFs).

        Args:
            x (np.ndarray): A size :math:`M` 1d array of inputs at which CDF is evaluated.

        Returns:
            np.ndarray: A size :math:`M` 1d array of outputs containing CDF evaluations.
        """
        if self.pctype == 'LU':
            cdf = (x+1.)/2.
            cdf[cdf>1]=1.0
            cdf[cdf<=-1]=0.0
        elif self.pctype == 'HG':
            cdf = (1.0 + erf(x / np.sqrt(2.0))) / 2.0
        else:
            print(f'PC1d type {self.pctype} is not recognized. Exiting.')
            sys.exit()

        return cdf

    def germSample(self, nsam):
        r"""Samples the germ.

        Args:
            nsam (int): Input number of samples requested, :math:`M`.

        Returns:
            np.ndarray: A 1d array of size :math:`M` containing the germ samples.
        """
        if self.pctype == 'LU':
            germ_sam = np.random.rand(nsam)*2.0-1.0
        elif self.pctype == 'HG':
            germ_sam = np.random.randn(nsam)
        else:
            print(f'PC1d type {self.pctype} is not recognized. Exiting.')
            sys.exit()

        return germ_sam

    def normsq(self, ord):
        r"""Computes norm-squared of a basis with a given order.

        Args:
            ord (int): Requested order :math:`p`.

        Returns:
            float: The norm-squared of the basis of order :math:`p`.
        """
        if self.pctype == 'LU':
            val = 1./(2.*float(ord)+1.)
        elif self.pctype == 'HG':
            if ord == 0:
                val = 1.0
            else:
                val = float(np.prod(np.arange(1, ord+1)))
        else:
            print(f'PC1d type {self.pctype} is not recognized. Exiting.')
            sys.exit()

        return val

    def quad(self, k):
        """One-dimensional quadrature point/weight generation.

        Args:
            k (int): The level of the quadrature.

        Returns:
            tuple: A pair of 1d arrays of the same size, one for the quadrature points, and the other for the corresponding weights.

        Note:
            Utilizes the Golub-Welsch method, see :cite:t:`Golub:1969` or https://www.ams.org/journals/mcom/1969-23-106/S0025-5718-69-99647-1/S0025-5718-69-99647-1.pdf.
        """
        gw = np.zeros((k, k))
        for i in range(k-1):
            gw[i+1, i] = np.sqrt(-self.b(i+1)/(self.a(i)*self.a(i+1)))

        eig, evec = np.linalg.eigh(gw)

        qdpts = eig
        wghts = evec[0,:]**2

        return qdpts, wghts
