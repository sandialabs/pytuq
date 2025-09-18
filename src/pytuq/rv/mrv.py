#!/usr/bin/env python
"""Classes for various multivariate random variables."""


import sys
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal
from scipy.special import erf

from ..minf.mcmc import AMCMC
from ..func.func import Function


############################################################
############################################################
############################################################

class MRV():
    """Base class for multivariate random variables.

    Attributes:
        pdim (int): The dimensionality of the random variable.
    """

    def __init__(self, pdim):
        r"""Initializing function.

        Args:
            pdim (int): The dimensionality :math:`d` of the multivariate random variable/vector.
        """
        self.pdim = pdim

    def __repr__(self):
        return f"Multivariate Random Variable(dim={self.pdim})"

    def sample(self, nsam):
        raise NotImplementedError("Sampling is not implemented in the base class.")

    def pdf(self, x):
        raise NotImplementedError("PDF function is not implemented in the base class.")

    def logpdf(self, x):
        return np.log(self.pdf(x))


    def cdf(self, x):
        raise NotImplementedError("CDF function is not implemented in the base class.")

    def sample_indomain(self, nsam, domain, itry_max=1000):
        r"""Sampling according to the distribution truncated within a given domain.

        Args:
            nsam (int): Number of samples requested, :math:`N`.
            domain (np.ndarray): A domain given as a 2d array of size :math:`(d,2)`.
            itry_max (int): maximum number of chunks of :math:`N` sampled. Effectively this is the rejection ratio. If too high, end without having :math:`N` samples in the domain and provide a warning.

        Returns:
            np.ndarray: A 2d array of samples of size `(N,d)`.

        Note:
            This is based on rejection sampling, and if the domain has small intersection with the volume of sampling, it may take long time.
        """
        sam = np.empty((0, self.pdim))
        itry = 0
        while sam.shape[0]<nsam:
            itry += 1
            sam_check = self.sample(nsam)
            ind = (sam_check>domain[:,0]).all(axis=1)*(sam_check<domain[:,1]).all(axis=1)
            sam = np.vstack((sam, sam_check[ind]))
            if itry > itry_max:
                print(f"WARNING: Rejection rate is too high, sampled only N={sam.shape[0]} points instead of requested {nsam}")
                return sam

        return sam[:nsam]

    def volume_indomain(self, domain):
        r"""Compute volume of the PDF of this random variable truncated within a given domain.

        Args:
            domain (np.ndarray): A domain given as a 2d array of size :math:`(d,2)`.

        Returns:
            float: Volume of the PDF inside the given domain.

        Note:
            see :cite:t:`Kan:2017` or https://www.cesarerobotti.com/wp-content/uploads/2019/04/JCGS-KR.pdf for the math.
        """
        ii = np.array([i for i in product(range(2), repeat=self.pdim)])
        volume = 0.0
        for i in ii:
            corner = domain[:, 0].copy()
            corner[i==1] = domain[i==1, 1]
            volume += (-1)**(self.pdim-np.sum(i)) * self.cdf(corner)

        return volume

##################################################################
##################################################################
##################################################################

class GMM(MRV):
    r"""Gaussian mixture model random variable.

    Attributes:
        means (list[np.ndarray]): List of :math:`K` means, each a 1d array of size :math:`d`.
        covs (list[np.ndarray]): List of :math:`K` covariances, each a 2d array of size :math:`(d, d)`.
        weights (np.ndarray): An 1d array of size :math:`K` for the mixture weights.
    """
    def __init__(self, means, covs=None, weights=None):
        r"""Initialization.

        Args:
            means (list[np.ndarray]): List of :math:`K` means, each a 1d array of size :math:`d`.
            covs (list[np.ndarray], optional): List of :math:`K` covariances, each a 2d array of size :math:`(d, d)`. Default is None, which means identity covariance for all mixtures.
            weights (np.ndarray, optional): An 1d array of size :math:`K` for the mixture weights. Default is None, which means equal weights to all mixtures.
        """
        super().__init__(len(means[0]))

        self.means = means
        ncl = len(self.means)

        if covs is None:
            self.covs = [np.eye(mean.shape[0]) for mean in self.means]
        else:
            self.covs=covs
        if weights is None or np.sum(weights)==0.0:
            self.weights = np.ones((ncl,))/ncl
        else:
            self.weights = weights / np.sum(weights)
        self.size_checks()

    def size_checks(self):
        """Size checks to make sure everything is consistent.
        """
        assert(len(self.weights)==len(self.means))
        assert(len(self.weights)==len(self.covs))
        for mean, cov in zip(self.means, self.covs):
            assert(len(mean)==len(self.means[0]))
            assert(len(mean)==cov.shape[0])
            assert(len(mean)==cov.shape[1])


    def sample(self, nsam):
        r"""Sampling function for this random variable.

        Args:
            nsam (int): Number of requested samples :math:`N`.

        Returns:
            np.ndarray: A 2d array of size :math:`(N,d)`.
        """
        nmcs = np.random.multinomial(nsam, self.weights, size=1)[0]
        assert(np.sum(nmcs) == nsam)
        sam = np.empty((0, self.pdim))
        for mean, cov, nmc_this in zip(self.means, self.covs, nmcs):
            sam_this = np.random.multivariate_normal(mean,
                                                     cov,
                                                     size=(int(nmc_this), ))
            sam = np.vstack((sam, sam_this))

        return sam

    def pdf(self, xdata):
        r"""PDF evaluated at given points.

        Args:
            xdata (np.ndarray): A 2d array of size :math:`(M,d)` for :math:`M` points at which PDF is evaluated.

        Returns:
            np.ndarray: A 1d array of size :math:`M` for PDF evaluation at the given points.
        """
        gmm_pdf = 0.0
        for mean, cov, weight in zip(self.means, self.covs, self.weights):
            gmm_pdf += weight * multivariate_normal.pdf(xdata, mean=mean, cov=cov)

        return gmm_pdf

    def cdf(self, xdata):
        r"""CDF evaluated at given points.

        Args:
            xdata (np.ndarray): A 2d array of size :math:`(M,d)` for :math:`M` points at which CDF is evaluated.

        Returns:
            np.ndarray: A 1d array of size :math:`M` for CDF evaluation at the given points.
        """
        gmm_cdf = 0.0
        for mean, cov, weight in zip(self.means, self.covs, self.weights):
            dist = multivariate_normal(mean=mean, cov=cov)
            gmm_cdf += weight * dist.cdf(np.array(xdata))

        return gmm_cdf

############################################################
############################################################
############################################################

class Mixture(MRV):
    """A class for a mixture of random variables.

    Attributes:
        rv_list (list[rv.mrv.MRV]): List of Multivariate Random Variable objects.
        weights (np.ndarray): Mixture weights, an 1d array.
    """

    def __init__(self, rv_list, weights=None):
        """Summary

        Args:
            rv_list (list[rv.mrv.MRV]): List of Multivariate Random Variable objects.
            weights (None, optional): Weighte of the mixture. Defaults to None, meaning equal weights.
        """
        super().__init__(rv_list[0].pdim)
        for rv_ in rv_list:
            assert(rv_.pdim==self.pdim)

        self.rv_list = rv_list

        nmix = len(self.rv_list)
        if weights is None:
            self.weights = np.ones((nmix,))/nmix
        else:
            self.weights = np.array(weights)

        assert(len(self.weights)==nmix)
        assert np.isclose(np.sum(self.weights),1.0), f'{np.sum(self.weights)}'

    def __repr__(self):
        """String representation.

        Returns:
            str: String description.
        """
        return f"Mixture of random variables {[i for i in self.rv_list]}"

    def sample(self, nsam):
        """Sampling routine.

        Args:
            nsam (int): Number of samples requested.

        Returns:
            np.ndarray: A 1d array of samples.
        """
        nmix = len(self.rv_list)
        imix = np.random.choice(range(nmix), size=nsam, p=self.weights)

        samples = np.empty((nsam,))
        for i in range(nsam):
            samples[i] = self.rv_list[imix[i]].sample(1)

        return samples

    def pdf(self, x):
        """Evaluate the PDF.

        Args:
            x (np.ndarray): 1d array at which PDF is evaluated.

        Returns:
            np.ndarray: 1d array of PDF values.
        """
        value = 0.0
        for i, rv in enumerate(self.rv_list):
            value+=self.weights[i]*rv.pdf(x)

        return value


    def cdf(self, x):
        """Evaluate the CDF.

        Args:
            x (np.ndarray): 1d array at which CDF is evaluated.

        Returns:
            np.ndarray: 1d array of CDF values.
        """
        value = 0.0
        for i, rv in enumerate(self.rv_list):
            value+=self.weights[i]*rv.cdf(x)

        return value

############################################################
############################################################
############################################################

class Inverse(MRV):
    """Class for an inverse of a random variable.

    Attributes:
        rv (rv.mrv.MRV): Underlying original Multivariate Random Variable object.
    """
    def __init__(self, rv):
        """Initialization.

        Args:
            rv (rv.mrv.MRV): Multivariate Random Variable object.
        """
        super().__init__(rv.pdim)
        self.rv = rv

    def __repr__(self):
        """String representation.

        Returns:
            str: String description.
        """
        return f"Inverse of random variable {self.rv}"

    def sample(self, nsam):
        """Sampling routine.

        Args:
            nsam (int): Number of samples requested.

        Returns:
            np.ndarray: A 1d array of samples.
        """
        return 1./self.rv.sample(nsam)


    def pdf(self, x):
        """Evaluate the PDF.

        Args:
            x (np.ndarray): 1d array at which PDF is evaluated.

        Returns:
            np.ndarray: 1d array of PDF values.
        """
        return self.rv.pdf(1.0/x)/x**2

    def logpdf(self, x):
        """Evaluate the log-PDF.

        Args:
            x (np.ndarray): 1d array at which log-PDF is evaluated.

        Returns:
            np.ndarray: 1d array of log-PDF values.
        """
        return self.rv.logpdf(1.0/x) - 2.0*np.log(x)

    def cdf(self, x):
        """Evaluate the CDF.

        Args:
            x (np.ndarray): 1d array at which CDF is evaluated.

        Returns:
            np.ndarray: 1d array of CDF values.
        """
        return 1.-self.rv.cdf(1./x)


############################################################

class Pareto_1d(MRV):
    """A class for univariate Pareto distribution.

    Attributes:
        b (float): Power parameter.
    """

    def __init__(self, alpha=1.0, xm=1.0):
        """Initialization.

        Args:
            alpha (float): Power parameter.
            xm (float): The domain minimum.
        """
        super().__init__(1) #univariate
        assert alpha>0.0, f'{alpha}'
        assert xm>0.0, f'{xm}'

        self.alpha = alpha
        self.xm = xm
        self.params = [self.alpha, self.xm]

    def __repr__(self):
        """String representation.

        Returns:
            str: String description.
        """
        return f"Univariate Pareto Random Variable"

    def sample(self, nsam):
        """Sampling routine.

        Args:
            nsam (int): Number of samples requested.

        Returns:
            np.ndarray: A 1d array of samples.
        """
        return self.xm*np.random.rand(nsam)**(-1./self.alpha)

    def pdf(self, x):
        """Evaluate the PDF.

        Args:
            x (np.ndarray): 1d array at which PDF is evaluated.

        Returns:
            np.ndarray: 1d array of PDF values.
        """
        if np.any(x<self.xm):
            return np.zeros_like(x)

        return self.alpha*self.xm**self.alpha / x**(self.alpha+1.0)

    def logpdf(self, x):
        """Evaluate the log-PDF.

        Args:
            x (np.ndarray): 1d array at which log-PDF is evaluated.

        Returns:
            np.ndarray: 1d array of log-PDF values.
        """
        if np.any(x<self.xm):
            return 0.0 # actually should be -inf

        return np.log(self.alpha)+self.alpha*np.log(self.xm)-(self.alpha+1.0)*np.log(x)

    def cdf(self, x):
        """Evaluate the CDF.

        Args:
            x (np.ndarray): 1d array at which CDF is evaluated.

        Returns:
            np.ndarray: 1d array of CDF values.
        """
        cumdf = np.zeros_like(x)

        cumdf[x>self.xm] = 1.0 - (self.xm/x[x>self.xm])**self.alpha

        return cumdf



############################################################

class Normal_1d(MRV):
    """A class for univariate Normal distribution.

    Attributes:
        mu (float): Location parameter.
        sigma (float): Scale parameter.
    """

    def __init__(self, mu=0.0, sigma=1.0):
        """Initialization.

        Args:
            mu (float): Location parameter.
            sigma (float): Scale parameter.
        """
        super().__init__(1)  # univariate
        assert sigma > 0.0, f'{sigma}'
        self.mu = mu
        self.sigma = sigma

        self.params = [self.mu, self.sigma]

    def __repr__(self):
        """String representation.

        Returns:
            str: String description.
        """
        return f"Univariate Normal Random Variable"

    def sample(self, nsam):
        """Sampling routine.

        Args:
            nsam (int): Number of samples requested.

        Returns:
            np.ndarray: A 1d array of samples.
        """
        return self.sigma * np.random.randn(nsam) + self.mu

    def pdf(self, x):
        """Evaluate the PDF.

        Args:
            x (np.ndarray): 1d array at which PDF is evaluated.

        Returns:
            np.ndarray: 1d array of PDF values.
        """
        return np.exp(-0.5 * ((x - self.mu) / self.sigma)**2) / (self.sigma * np.sqrt(2. * np.pi))

    def logpdf(self, x):
        """Evaluate the log-PDF.

        Args:
            x (np.ndarray): 1d array at which log-PDF is evaluated.

        Returns:
            np.ndarray: 1d array of log-PDF values.
        """
        return -np.log(self.sigma) - 0.5 * np.log(2. * np.pi) - 0.5 * ((x - self.mu) / self.sigma)**2

    def cdf(self, x):
        """Evaluate the CDF.

        Args:
            x (np.ndarray): 1d array at which CDF is evaluated.

        Returns:
            np.ndarray: 1d array of CDF values.
        """
        cumdf = np.zeros_like(x)

        cumdf[x > 0] = 0.5 + 0.5 * erf((x - self.mu) / (np.sqrt(2.0) * self.sigma))

        return cumdf

############################################################

class Lognormal_1d(MRV):
    """A class for univariate Lognormal distribution.

    Attributes:
        mu (float): Location parameter.
        sigma (float): Scale parameter.
    """

    def __init__(self, mu=0.0, sigma=1.0):
        """Initialization.

        Args:
            mu (float): Location parameter.
            sigma (float): Scale parameter.
        """
        super().__init__(1) #univariate
        assert sigma>0.0, f'{sigma}'
        self.mu = mu
        self.sigma = sigma

        self.params = [self.mu, self.sigma]


    def __repr__(self):
        """String representation.

        Returns:
            str: String description.
        """
        return f"Univariate Lognormal Random Variable"

    def sample(self, nsam):
        """Sampling routine.

        Args:
            nsam (int): Number of samples requested.

        Returns:
            np.ndarray: A 1d array of samples.
        """
        return np.exp(self.sigma*np.random.randn(nsam)+self.mu)

    def pdf(self, x):
        """Evaluate the PDF.

        Args:
            x (np.ndarray): 1d array at which PDF is evaluated.

        Returns:
            np.ndarray: 1d array of PDF values.
        """
        assert ~np.any(x<=0.0)
        return np.exp(-0.5*(np.log(x)-self.mu)/ self.sigma)**2 / (x*self.sigma*np.sqrt(2.*np.pi))


    def logpdf(self, x):
        """Evaluate the log-PDF.

        Args:
            x (np.ndarray): 1d array at which log-PDF is evaluated.

        Returns:
            np.ndarray: 1d array of log-PDF values.
        """
        assert ~np.any(x<=0.0)
        return -np.log(x)-np.log(self.sigma)-0.5*np.log(2.*np.pi)- 0.5*((np.log(x)-self.mu)/ self.sigma)**2

    def cdf(self, x):
        """Evaluate the CDF.

        Args:
            x (np.ndarray): 1d array at which CDF is evaluated.

        Returns:
            np.ndarray: 1d array of CDF values.
        """
        cumdf = np.zeros_like(x)

        cumdf[x>0] = 0.5+0.5*erf((np.log(x[x>0])-self.mu)/ (np.sqrt(2.0)*self.sigma))

        return cumdf


############################################################

class Weibull_1d(MRV):
    """A class for univariate Weibull distribution.

    Attributes:
        k (float): Shape parameter.
        lam (float): Scale parameter.
    """

    def __init__(self, lam, k=1.0):
        """Initialization.

        Args:
            lam (float): Scale parameter.
            k (float): Shape parameter.
        """
        super().__init__(1) #univariate
        assert lam>0, f'{lam}'
        assert k>0, f'{k}'
        self.lam = lam
        self.k = k

        self.params = [self.lam, self.k]


    def __repr__(self):
        """String representation.

        Returns:
            str: String description.
        """
        return f"Univariate Weibull Random Variable"

    def sample(self, nsam):
        """Sampling routine.

        Args:
            nsam (int): Number of samples requested.

        Returns:
            np.ndarray: A 1d array of samples.
        """
        return self.lam*np.power(-np.log(np.random.rand(nsam)), 1./self.k)

    def pdf(self, x):
        """Evaluate the PDF.

        Args:
            x (np.ndarray): 1d array at which PDF is evaluated.

        Returns:
            np.ndarray: 1d array of PDF values.
        """
        xc = x.copy()
        xc[xc<=0.0]=0.0
        return (self.k / self.lam) * (xc / self.lam)**(self.k - 1) * np.exp(-(xc / self.lam)**self.k)

    def logpdf(self, x):
        """Evaluate the log-PDF.

        Args:
            x (np.ndarray): 1d array at which log-PDF is evaluated.

        Returns:
            np.ndarray: 1d array of log-PDF values.
        """
        xc = x.copy()
        xc[xc<=0.0]=0.0
        return np.log(self.k)- np.log(self.lam) + (self.k - 1) * (np.log(xc)-np.log(self.lam)) -(xc / self.lam)**self.k

    def cdf(self, x):
        """Evaluate the CDF.

        Args:
            x (np.ndarray): 1d array at which PDF is evaluated.

        Returns:
            np.ndarray: 1d array of PDF values.
        """
        xc = x.copy()
        xc[xc<=0.0]=0.0
        return 1.0 - np.exp(-(xc / self.lam)**self.k)



##################################################################
##################################################################
##################################################################

class MCMCRV(MRV):
    r"""A class for MCMC-based random variable.

    Attributes:
        logpost (callable): Function evaluator np.ndarray->float for log-posterior. Can take optional keyword arguments as well.
        nmcmc (int): Number of MCMC steps
        param_ini (np.ndarray): Initial condition of the chain, a 1d array of size :math:`d`.
    """

    def __init__(self, pdim, logpost, param_ini=None, nmcmc=10000):
        r"""Initialization.

        Args:
            pdim (int): The dimensionality of the random variable.
            logpost (callable): Function evaluator :math:`(N,)`->scalar for log-posterior.
            param_ini (None, optional): Initial condition of the chain, a 1d array of size :math:`d`. Defaults to None, which populates the array with hardwired 0.1 values.
            nmcmc (int, optional): Number of MCMC steps, defaults to 10000.
        """
        super().__init__(pdim)
        self.logpost = logpost
        if param_ini is not None:
            self.param_ini = param_ini
        else:
            self.param_ini = 0.1*np.ones((self.pdim))
        self.nmcmc = nmcmc


    def sample(self, nsam, **post_info):
        r"""Sampling function.

        Args:
            nsam (int): Number of requested samples :math:`N`.
            post_info (dict): Dictionary keyword arguments for the logpost function.

        Returns:
            np.ndarray: A 2d array of size :math:`(N,d)`.
        """
        if nsam>self.nmcmc//2:
            print(f"Can not sample more than half of the original chain. {nsam}>{self.nmcmc//2}. Exiting.")
            sys.exit()

        cov_ini = np.diag(0.1+0.1*np.abs(self.param_ini))

        calib_params={'cov_ini': cov_ini,
                      't0': 100, 'tadapt' : 100,
                      'gamma' : 0.1}

        calib = AMCMC(**calib_params)

        calib.setLogPost(self.logpost, None, **post_info)
        calib_results = calib.run(self.nmcmc, self.param_ini)
        #calib_results = {'chain' : samples, 'mapparams' : cmode,
        #'maxpost' : pmode, 'accrate' : acc_rate, 'logpost' : logposts, 'alphas' : alphas}

        every = (self.nmcmc//2)//nsam
        sam = calib_results['chain'][::-1][:every*nsam][::every]
        assert(sam.shape[0]==nsam)
        return sam

    def pdf_unscaled(self, x):
        r"""PDF evaluation without scaling.

        Args:
            x (np.ndarray): 1d array of size :math:`M` where PDF is evaluated.

        Returns:
            np.ndarray: PDF evaluated at requested points, an array of size :math:`M`.

        Note:
            PDF is unscaled, i.e. only evaluates exponential of the log-posterior.
        """
        return np.exp(np.array([self.logpost(xx) for xx in x]))

    def pdf(self, x):
        raise NotImplementedError("CDF function is not implemented in this class.")


    def cdf(self, x):
        raise NotImplementedError("CDF function is not implemented in this class.")

##################################################################
##################################################################
##################################################################

