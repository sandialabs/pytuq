#!/usr/bin/env python
"""Particle Swarm Optimization module. Relies on pyswarms library.
"""

import sys
import numpy as np
from .optim import OptBase

try:
    import pyswarms as ps
except ImportError:
    print("Warning: pyswarms not installed. Particle swarm optimization will not work.")
    sys.exit()



class PSO(OptBase):
    r""""Particle swarm optimization class.
    """

    def __init__(self, dim):
        """Initialization.

        """
        super().__init__()
        self.dim = dim

    def run(self, nsteps, param_ini, nparticles=100, bounds=None, psoptions=None):
        """A PSO optimization run.

        Args:
            nsteps (int): Number of optimization steps.
            param_ini (None): Is irrelevant, and only given to match the signature of the parent.
            nparticles (int, optional): Number of particles. Defaults to 100.
            bounds (np.ndarray, optional): Optimization bounds array.
            psoptions (None, optional): Optional keyword arguments to pass to pyswarms.

        Returns:
            dict: Dictionary of results. Keys are 'samples' (history array of parameters), 'objvalues' (history of objective values), 'best' (best parameters array), 'bestobj' (best objective value).
        """
        assert(self.Objective is not None)

        if bounds is None:
            min_bound = -10.
            max_bound = 10.
            bounds = ([min_bound]*self.dim, [max_bound]*self.dim)

        if psoptions is None:
            psoptions = {
            'c1': 0.5,  # Cognitive parameter
            'c2': 0.3,  # Social parameter
            'w': 0.9,   # Inertia weight
            'k': 3,     # Number of neighbors for each particle
            'p': 2      # Minkowski p-norm for distance calculation (e.g., 2 for Euclidean)
            }

        def obj_func(x):
            out = np.zeros([x.shape[0]])
            for i in range(x.shape[0]):
                out[i] = self.Objective(np.squeeze(x[i,:]))
            return out

        optimizer = ps.single.GlobalBestPSO(n_particles=nparticles, dimensions=self.dim, bounds = bounds, options=psoptions)
        cost, opt = optimizer.optimize(obj_func, iters=nsteps)
        print('Optimal values via PSO: ', opt)
        objvalues = np.array(optimizer.cost_history)

        samples = np.mean(optimizer.pos_history, axis=1)

        results = {
            'samples': samples,
            'objvalues': objvalues,
            'best': opt,
            'bestobj': cost
        }

        return results
