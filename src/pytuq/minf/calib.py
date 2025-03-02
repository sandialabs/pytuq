#!/usr/bin/env python
"""Metropolis-Hastings calibration module."""
import numpy as np

class MCMCBase(object):
    """Base class for calibration."""

    def __init__(self):
        """Dummy instantiation."""
        self.logPost = None
        self.logPostGrad = None
        self.postInfo = {}


    def setLogPost(self, logPost, logPostGrad, **postInfo):
        """Setting LogPost and optionally its Gradient.

        Args:
            logPost (callable): Log-posterior evaluator function.
            logPostGrad (callable): Log-posterior gradient evaluator function. Can be None.
            **postInfo: Dictionary arguments for the log-Posterior and its gradient.
        """
        self.logPost = logPost
        self.logPostGrad = logPostGrad
        self.postInfo = postInfo


    def sampler(self, current, imcmc):
        """Sampler function of a single step.

        Args:
            current (np.ndarray): The current chain state.
            imcmc (int): Current step.

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError("Sampler is not implemented in the parent class.")

    def run(self, nmcmc, param_ini):
        """Metropolis-Hastings run of Markov chain Monte Carlo.

        Args:
            nmcmc (int): Number of MCMC steps.
            param_ini (np.ndarray): Initial chain position, an 1d array.

        Returns:
            dict: Dictionary of results. Keys are 'chain' (chain samples array), 'mapparams' (MAP parameters array), 'maxpost' (maximal log-post value), 'accrate' (acceptance rate), 'logpost' (log-post values throughout the chain), 'alphas' (acceptance probabilities throughout the chain).
        """
        assert(self.logPost is not None)
        cdim = len(param_ini)            # chain dimensionality
        samples = []  # MCMC samples
        alphas = [] # Store alphas (posterior ratios)
        logposts = []  # Log-posterior values]
        na = 0                        # counter for accepted steps

        current = param_ini.copy()                # first step
        current_U = -self.logPost(current, **self.postInfo)  # NEGATIVE logposterior
        pmode = -current_U  # record MCMC 'mode', which is the current MAP value (maximum posterior)
        cmode = current  # MAP sample

        samples.append(current)
        logposts.append(-current_U)
        alphas.append(0.0)

        # Loop over MCMC steps
        for imcmc in range(nmcmc):
            current_proposal, current_K, proposed_K = self.sampler(current, imcmc)

            proposed_U = -self.logPost(current_proposal, **self.postInfo)
            proposed_H = proposed_U + proposed_K
            current_H = current_U + current_K

            mh_prob = np.exp(current_H - proposed_H)

            # Accept...
            if np.random.random_sample() < mh_prob:
                na += 1  # Acceptance counter
                current = current_proposal+0.0
                current_U = proposed_U+0.0
                if -current_U >= pmode:
                    pmode = -current_U
                    cmode = current+0.0

            samples.append(current)
            alphas.append(mh_prob)
            logposts.append(-current_U)

            acc_rate = float(na) / (imcmc+1)

            if((imcmc + 2) % (nmcmc / 10) == 0) or imcmc == nmcmc - 2:
                print('%d / %d completed, acceptance rate %lg' % (imcmc + 2, nmcmc, acc_rate))

        results = {
            'chain' : np.array(samples),
            'mapparams' : cmode,
            'maxpost' : pmode,
            'accrate' : acc_rate,
            'logpost' : np.array(logposts),
            'alphas' : np.array(alphas)
            }

        return results

