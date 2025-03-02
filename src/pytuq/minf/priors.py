#!/usr/bin/env python
"""Module for prior classes"""

import sys
import numpy as np


class Prior(object):
    """Base class for prior evaluation.

    Attributes:
        infer (infer.Infer): Inference object containing necessary info for prior evaluation.
        prtype (str): Prior name. Currently, 'uniform' or 'normal' are implemented.
        mean (np.ndarray): Prior mean array.
    """

    def __init__(self, infer):
        """Instantiation of a Prior object.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the prior.
        """
        self.infer = infer
        self.prtype = None
        self.mean = None

    def _setMean(self, mean):
        """Set the mean array.

        Args:
            mean (np.ndarray): Mean array.
        """
        self.mean = mean

    def eval(self, pp):
        """Evaluation of the prior. Not implemented in the base class.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Raises:
            NotImplementedError: This should be implemented in children classes.
        """
        raise NotImplementedError


    def computeDataPrior(self, pp):
        """Computes the part of the prior for the parameter relevant to data noise, in case the chain includes it.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            float: Returns log-prior corresponding to parameter controlling data noise.
        """
        if self.infer.dvtype == 'std_infer':
            logprior_baseline = -np.log(abs(pp[-1]))
        elif self.infer.dvtype == 'std_infer_log':
            logprior_baseline = 0.0
        elif self.infer.dvtype == 'var_fromdata_infer':
            logprior_baseline = - np.log(pp[-1]**2)
        elif self.infer.dvtype == 'log_var_fromdata_infer':
            logprior_baseline = 0.0
        else:
            logprior_baseline = 0.0

        return logprior_baseline

###############################################################
###############################################################
###############################################################

class Prior_uniform(Prior):
    r"""Uniform prior class.

    Attributes:
        domain (np.ndarray): A 2d array of size `(d,2)`. Can be None, for uninformative priors with no range constraint.
        factor (float): A standard deviation factor outside of which the prior returns very low value. For now, we use :math:`f=3` for HG, and :math:`f=\sqrt{3}` for LU. This is somewhat empiric and should be used with care.

    Note:
        Note that unlike normal prior, this is *not* a uniform prior on all parameters. This is somewhat empirical rejection mechanism if embedded parameter values lead to values outside given ranges.
    """

    def __init__(self, infer, domain=None):
        """Instantiates a uniform prior object.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the prior.
            domain (np.ndarray, optional): A 2d array of size `(d,2)`. Default is None, for uninformative priors with no range constraint.
        """
        super().__init__(infer)
        self.prtype = 'uniform'

        self.domain = domain
        if self.domain is not None:
            self._setMean(0.5 * (self.domain[:, 0] + self.domain[:, 1]))
        else:
            self._setMean(np.zeros(self.infer.inpcrv.pdim))



        if self.infer.pc_type == 'HG':
            self.factor = 3.0
        elif self.infer.pc_type == 'LU':
            self.factor = np.sqrt(3.0)
        else:
            print(f"Embedded PC type {self.infer.pc_type} is unknown. Exiting.")
            sys.exit()

    def __repr__(self):
        """Print representation of a uniform prior object.

        Returns:
            str: String representation.
        """
        if self.domain is None:
            stri = f"Prior: Uniform with no range constraint"
        else:
            stri = f"Prior: Uniform with range {self.domain}"
        return stri

    def eval(self, pp):
        """Evaluate the log-prior.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            float: Value of the log-prior.
        """
        logprior = self.computeDataPrior(pp)

        pcf_flat = np.array(self.infer.pcflat)
        pcf_flat[pcf_flat==None]=pp[:len(pp)-self.infer.extrainferparams]
        self.infer.inpcrv.cfsUnflatten(pcf_flat)

        pmeans, pstdevs = self.infer.inpcrv.computeMean(), np.sqrt(self.infer.inpcrv.computeVar())

        # for idim in range(self.infer.inpcrv.pdim):
        #     cfs = self.infer.inpcrv.coefs[idim]

            # # For independent embedding, the 1st order coefficient (standard deviation) is positive
            # if self.infer.inpdf_type=='pci' and len(cfs)>1 and cfs[1]<0.0:
            #     return -1.e+80

            # # For triangular embedding, all last row of coefficients are positive
            # if self.infer.inpdf_type=='pct' and idim==self.infer.rndind[-1] and len(cfs)>1 and not np.prod(cfs[1:]>0.0):
            #     return -1.e+80

        # For independent embedding, the 1st order coefficient (standard deviation) is positive
        if self.infer.inpdf_type=='pci':
            for i, chind in enumerate(self.infer.chind):
                if chind[1]==1 and pp[i]<=0.0:
                    return -1.e+80

        # For triangular embedding, all last row of coefficients are positive
        elif self.infer.inpdf_type=='pct':
            if (pp[len(pp)-self.infer.extrainferparams-self.infer.sdim:len(pp)-self.infer.extrainferparams]<=0.0).any():
                    return -1.e+80


        if self.domain is not None:
            for i, chind in enumerate(self.infer.chind):
                if chind[1]==0:
                    idim = chind[0]
                    if pmeans[idim]+self.factor * pstdevs[idim]>self.domain[idim, 1] or pmeans[idim]-self.factor * pstdevs[idim]<self.domain[idim, 0]:
                        return -1.e+80


            #logprior -= np.log(abs(cfs[1])) this was giving some unexpected behavior (stuck at sigma=0 values)
            #print(pmeans, logprior)

        return logprior


###############################################################
###############################################################
###############################################################

class Prior_normal(Prior):
    r"""Normal prior class.

    Attributes:
        mean (np.ndarray): A 1d array of size `p-p_d` for prior mean, where `p` is dimensionality of the chain, and `p_d` is the number of extra inferred parameters.
        var (np.ndarray): A 1d array of size `p-p_d` for prior variance, where `p` is dimensionality of the chain, and `p_d` is the number of extra inferred parameters.
    """

    def __init__(self, infer, mean=None, var=None):
        """Instantiates a normal prior object.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the prior.
            mean (np.ndarray): A 1d array of size `d` for prior mean for physical parameters. If None, sets the mean to all zeros.
            var (np.ndarray): A 1d array of size `d` for prior variance for physical parameters. If None, effectively sets the variance to all infinite, i.e. no prior.
        """
        super().__init__(infer)
        self.prtype = 'normal'

        if mean is not None:
            self._setMean(mean)
        else:
            self._setMean(np.zeros(self.infer.inpcrv.pdim))

        self.var = var

        assert(self.mean.shape[0]==self.infer.inpcrv.pdim)
        if self.var is not None:
            assert(self.var.shape[0]==self.infer.inpcrv.pdim)


    def __repr__(self):
        """Print representation of a normal prior object.

        Returns:
            str: String representation.
        """
        stri = f"Prior: Independent normal with mean {self.mean} and variance {self.var}"

        return stri

    def eval(self, pp):
        """Evaluate the log-prior.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            float: Value of the log-prior.
        """
        logprior = self.computeDataPrior(pp)
        if self.var is None:
            return logprior


        for i, chind in enumerate(self.infer.chind):
            if chind[1]==0:
                logprior -= 0.5*np.log(2.*np.pi)
                logprior -= 0.5*np.log(self.var)
                logprior -= 0.5*(pp[i]-self.mean[chind[0]])**2/self.var[chind[0]]

        return logprior
