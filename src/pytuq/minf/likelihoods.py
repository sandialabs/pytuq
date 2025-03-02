#!/usr/bin/env python
"""Module for various likelihood options."""

import numpy as np

class Likelihood(object):
    """Base class for likelihood evaluation.

    Attributes:
        infer (infer.Infer): Inference object containing necessary info for likelihood evaluation.
    """
    def __init__(self, infer):
        """Instantiation of a Likelihood object.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the likelihood.
        """
        self.infer = infer

    def eval(self, pp):
        """Evaluation of the likelihood. Not implemented in the base class.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Raises:
            NotImplementedError: This should be implemented in children classes.
        """
        raise NotImplementedError

#############################################################################
#############################################################################
#############################################################################

class Likelihood_dummy(Likelihood):
    """Dummy likelihood class. Useful for testing."""

    def __init__(self, infer):
        """Instantiation.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the likelihood.
        """
        super().__init__(infer)

    def __repr__(self):
        """Print representation of the object.

        Returns:
            str: String representation.
        """
        return f"Likelihood: Dummy"

    def eval(self, pp):

        loglik = 0.0

        return loglik

#############################################################################
#############################################################################
#############################################################################

class Likelihood_classical(Likelihood):
    """Classical likelihood class."""

    def __init__(self, infer):
        """Instantiation.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the likelihood.
        """
        super().__init__(infer)


    def __repr__(self):
        """Print representation of the object.

        Returns:
            str: String representation.
        """
        return f"Likelihood: Classical"

    def eval(self, pp):
        """Evaluates the log-likelihood.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            float: Value of the log-likelihood.
        """
        model_means, model_vars = self.infer.getModelMoments_NISP(pp)
        datavar = self.infer.getDataVar(pp)
        loglik = 0.0
        for idata in range(self.infer.ndata):
            modelval = model_means[idata]
            for ie in range(self.infer.neachs[idata]):
                err = self.infer.md_transform(self.infer.ydata[idata][ie]) - modelval
                loglik -= 0.5 * err**2 / datavar[idata]
                loglik -= 0.5 * np.log(2.*np.pi)
                loglik -= 0.5 * np.log(datavar[idata])

        return loglik

#############################################################################
#############################################################################
#############################################################################

class Likelihood_logclassical(Likelihood):
    """Log-classical likelihood."""

    def __init__(self, infer):
        """Instantiation.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the likelihood.
        """
        super().__init__(infer)

    def __repr__(self):
        """Print representation of the object.

        Returns:
            str: String representation.
        """
        return f"Likelihood: Log-Classical"

    def eval(self, pp):
        """Evaluates the log-likelihood.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            float: Value of the log-likelihood.
        """
        model_means, model_vars = self.infer.getModelMoments_NISP(pp)
        datavar = self.infer.getDataVar(pp)
        loglik = 0.0
        for idata in range(self.infer.ndata):
            modelval = model_means[idata]
            for ie in range(self.infer.neachs[idata]):
                err = np.log(self.infer.md_transform(self.infer.ydata[idata][ie])) - np.log(modelval)
                loglik -= 0.5 * err**2 / datavar[idata]
                loglik -= 0.5 * np.log(2.*np.pi)
                loglik -= 0.5 * np.log(datavar[idata])

        return loglik

#############################################################################
#############################################################################
#############################################################################

class Likelihood_abc(Likelihood):
    r"""ABC likelihood.

    Attributes:
        abceps (float): ABC tolerance parameter :math:`\epsilon`.
        abcalpha (float): ABC stdev/residual factor :math:`\alpha`.
    """

    def __init__(self, infer, abceps=0.01, abcalpha=1.0):
        r"""Instantiation.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the likelihood.
            abceps (float, optional): ABC tolerance parameter :math:`\epsilon`. Defaults to 0.01.
            abcalpha (float, optional): ABC stdev/residual factor :math:`\alpha`. Defaults to 1.0.
        """
        super().__init__(infer)
        self.abceps = abceps
        self.abcalpha = abcalpha

    def __repr__(self):
        """Print representation of the object.

        Returns:
            str: String representation.
        """
        return f"Likelihood: Model Error ABC (eps={self.abceps}, alpha={self.abcalpha})"

    def eval(self, pp):
        """Evaluates the log-likelihood.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            float: Value of the log-likelihood.
        """
        model_means, model_vars = self.infer.getModelMoments_NISP(pp)
        datavar = self.infer.getDataVar(pp)

        loglik = 0.0
        for idata in range(self.infer.ndata):
            for ie in range(self.infer.neachs[idata]):
                err = np.abs(self.infer.md_transform(self.infer.ydata[idata][ie]) - model_means[idata])
                loglik -= 0.5 * err**2 / self.abceps**2
                loglik -= 0.5 * (self.abcalpha*err-np.sqrt(model_vars[idata]+datavar[idata]))**2 / self.abceps**2
        loglik -= 0.5 * np.log(2.*np.pi)
        loglik -= np.log(self.abceps)

        return loglik

#############################################################################
#############################################################################
#############################################################################

class Likelihood_gausmarg(Likelihood):
    """IID Likelihood."""

    def __init__(self, infer):
        """Instantiation.

        Args:
            infer (infer.Infer): Inference object containing necessary info for the likelihood.
        """
        super().__init__(infer)


    def __repr__(self):
        """Print representation of the object.

        Returns:
            str: String representation.
        """
        return f"Likelihood: Model Error Gaussian iid"

    def eval(self, pp):
        """Evaluates the log-likelihood.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            float: Value of the log-likelihood.
        """
        model_means, model_vars = self.infer.getModelMoments_NISP(pp)
        datavar = self.infer.getDataVar(pp)

        loglik = 0.0
        for idata in range(self.infer.ndata):
            for ie in range(self.infer.neachs[idata]):
                err = np.abs(self.infer.md_transform(self.infer.ydata[idata][ie]) - model_means[idata])
                loglik -= 0.5 * err**2 / (model_vars[idata]+datavar[idata])
                loglik -= 0.5 * np.log(model_vars[idata]+datavar[idata])
                loglik -= 0.5 * np.log(2.*np.pi)


        return loglik
