#!/usr/bin/env python
"""Module for various MCMC flavors."""

import numpy as np
from .calib import MCMCBase

class AMCMC(MCMCBase):
    r"""Adaptive MCMC class. Based on :cite:t:`Haario:2001`.

    Attributes:
        cov_ini (np.ndarray): Initial covariance array of size `(p,p)`.
        gamma (float): Proposal jump size factor :math:`\gamma`.
        propcov (np.ndarray): Proposal covariance, working array.
        t0 (int): Step where adaptivity begins.
        tadapt (int): Adapt/update covariance every `tadapt` steps.
    """
    def __init__(self, cov_ini=None, gamma=0.1, t0=100, tadapt=1000):
        r"""Initialization.

        Args:
            cov_ini (None, optional): Initial covariance. Defaults to None, in which case it is set to be proportional to the current state.
            gamma (float, optional): $\gamma$ parameter, effectively step size. Defaults to 0.1.
            t0 (int, optional): When adaptivity starts. Defaults to 100.
            tadapt (int, optional): Adapting frequency. Defaults to 1000.
        """
        super().__init__()

        self.cov_ini = cov_ini
        self.t0 = t0
        self.tadapt = tadapt
        self.gamma = gamma

    def sampler(self, current, imcmc):
        """Sampler function of a single step.

        Args:
            current (np.ndarray): The current chain state.
            imcmc (int): Current step.

        Returns:
            tuple(float, float, float): Current proposal, and two auxiliary ratios.
        """
        current_proposal = current.copy()
        cdim = len(current)

        # Compute covariance matrix
        if imcmc == 0:
            self._Xm = current.copy()
            self._cov = np.zeros((cdim, cdim))
        else:
            self._Xm = (imcmc * self._Xm + current) / (imcmc + 1.0)
            rt = (imcmc - 1.0) / imcmc
            st = (imcmc + 1.0) / imcmc**2
            self._cov = rt * self._cov + st * np.dot(np.reshape(current - self._Xm, (cdim, 1)), np.reshape(current - self._Xm, (1, cdim)))

        if imcmc == 0:
            if self.cov_ini is not None:
                self.propcov = self.cov_ini
            else:
                self.propcov = 0.01 + np.diag(0.09*np.abs(current))
        elif (imcmc > self.t0) and (imcmc % self.tadapt == 0):
                self.propcov = (self.gamma * 2.4**2 / cdim) * (self._cov + 10**(-8) * np.eye(cdim))

        # Generate proposal candidate
        current_proposal += np.random.multivariate_normal(np.zeros(cdim,), self.propcov)
        proposed_K = 0.0
        current_K = 0.0

        return current_proposal, current_K, proposed_K



###############################################################
###############################################################
###############################################################


class HMC(MCMCBase):
    # Implementation based on Neal, 2011. https://arxiv.org/pdf/1206.1901.pdf
    def __init__(self, epsilon=0.05, L=3):
        r"""Initialization.

        Args:
            epsilon (float, optional): $\epsilon$ discretization time-step.
            L (int, optional): L, step count.
        """
        super().__init__()
        self.epsilon = epsilon
        self.L = L

    def sampler(self, current, imcmc):
        """Sampler function of a single step.

        Args:
            current (np.ndarray): The current chain state.
            imcmc (int): Current step.

        Returns:
            tuple(float, float, float): Current proposal, and two auxiliary ratios.
        """
        assert(self.logPostGrad is not None)

        current_proposal = current.copy()
        cdim = len(current)


        p = np.random.randn(cdim)
        current_K = np.sum(np.square(p)) / 2

        # Make a half step for momentum at the beginning (Leapfrog Method step starts here)

        p += self.epsilon * self.logPostGrad(current_proposal, **self.postInfo) / 2

        for jj in range(self.L):
            # Make a full step for the position
            current_proposal += self.epsilon * p

            # Make a full step for the momentum, expecpt at the end of the trajectory

            if jj != self.L - 1:
                p += self.epsilon * self.logPostGrad(current_proposal, **self.postInfo)

        # Make a half step for momentum at the end (Leapfrog Method step ends here)
        p += self.epsilon* self.logPostGrad(current_proposal, **self.postInfo) / 2


        # Negate momentum to make proposal symmetric
        p = -p # This isn't really necessary but implemented per original paper

        # Evaluate kinetic and potential energies
        proposed_K = np.sum(np.square(p)) / 2

        return current_proposal, current_K, proposed_K




class MALA(MCMCBase):
    # Note: MALA is actually exactly HMC with L=1.
    # See Girolami paper https://statmodeling.stat.columbia.edu/wp-content/uploads/2010/04/RMHMC_MG_BC_SC_REV_08_04_10.pdf
    def __init__(self, epsilon=0.05):
        r"""Initialization.

        Args:
            epsilon (float, optional): $\epsilon$ discretization time-step.
        """
        super().__init__()
        self.epsilon = epsilon

    def sampler(self, current, imcmc):
        """Sampler function of a single step.

        Args:
            current (np.ndarray): The current chain state.
            imcmc (int): Current step.

        Returns:
            tuple(float, float, float): Current proposal, and two auxiliary ratios.
        """
        assert(self.logPostGrad is not None)

        current_proposal = current.copy()
        cdim = len(current)


        p = np.random.randn(cdim)

        grad_current = self.logPostGrad(current, **self.postInfo)
        current_proposal += 0.5*self.epsilon**2 * grad_current + self.epsilon * p

        grad_prop = self.logPostGrad(current_proposal, **self.postInfo)
        current_K = np.sum(np.square(p)) / 2

        p += self.epsilon * (grad_current+grad_prop)/ 2
        proposed_K = np.sum(np.square(p)) / 2

        return current_proposal, current_K, proposed_K
