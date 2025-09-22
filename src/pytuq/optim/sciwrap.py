#!/usr/bin/env python
"""LBFGS Optimization Module.
"""

import numpy as np

from .optim import OptBase

from scipy.optimize import minimize

class ScipyWrapper(OptBase):
    r""""Optimization class wrapper to scipy options. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
    """

    def __init__(self, method='BFGS', bounds=None, options=None):
        """Initialization.

        Args:
            method (str, optional): Method string. Defaults to BFGS.
            bounds (tuple, optional): Tuple of min-max pairs for each dimension. Defaults to None.
            options (None, optional): Optional keyword arguments to scipy minimize function.
        """
        super().__init__()
        self.method = method
        self.bounds = bounds
        self.options = options

    def run(self, nsteps, param_ini):
        """An optimization run.

        Args:
            param_ini (np.ndarray): Initial position, an 1d array.

        Returns:
            dict: Dictionary of results. Keys are 'samples' (history array of parameters), 'objvalues' (history of objective values), 'best' (best parameters array), 'bestobj' (best objective value).
        """
        assert(self.Objective is not None)

        self.history = []

        res = minimize(self.Objective, param_ini,
                       args=(),
                       method=self.method,
                       jac = self.ObjectiveGrad,
                       hess = self.ObjectiveHess,
                       bounds = self.bounds,
                       options=self.options,
                       callback=self.store_history)
        print('Opt:', res.x)
        #print(res)

        results = {
            'samples': np.array([hist['x'] for hist in self.history]),
            'objvalues': np.array([hist['fun_val'] for hist in self.history]),
            'best': res.x,
            'bestobj': res.fun
        }

        return results

    # Define the callback function
    def store_history(self, x, *args):
        """Store the current parameter vector and the objective function value.

        Args:
            x (np.ndarray): Current state.
            *args: Positional arguments, if any.
        """
        #
        self.history.append({'x': np.copy(x), 'fun_val': self.Objective(x)})


