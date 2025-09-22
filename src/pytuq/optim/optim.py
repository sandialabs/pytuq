#!/usr/bin/env python
"""Optimization base module."""
import numpy as np

class OptBase(object):
    """Base class for optimization."""

    def __init__(self):
        """Dummy instantiation."""
        self.Objective = None
        self.ObjectiveGrad = None
        self.ObjectiveHess = None
        self.ObjectiveInfo = {}


    def setObjective(self, Objective, ObjectiveGrad=None, ObjectiveHess=None, **ObjectiveInfo):
        """Setting Objective and optionally its Gradient and Hessian.

        Args:
            Objective (callable): Objective evaluator function.
            ObjectiveGrad (callable, optional): Objective gradient evaluator function. Defaults to None.
            ObjectiveHess (callable, optional): Objective hessian evaluator function. Defaults to None.
            **ObjectiveInfo: Dictionary arguments for the Objective and its gradient and hessian.
        """
        self.Objective = Objective
        self.ObjectiveGrad = ObjectiveGrad
        self.ObjectiveHess = ObjectiveHess
        self.ObjectiveInfo = ObjectiveInfo


    def stepper(self, current, istep):
        """Sampler function of a single step.

        Args:
            current (np.ndarray): The current state.
            istep (int): Current step.

        Raises:
            NotImplementedError: Not Implemented in the parent class.
        """
        raise NotImplementedError("Stepper is not implemented in the parent class.")

    def run(self, nsteps, param_ini):
        """An optimization run.

        Args:
            nsteps (int): Number of optimization steps.
            param_ini (np.ndarray): Initial position, an 1d array.

        Returns:
            dict: Dictionary of results. Keys are 'samples' (history array of parameters), 'objvalues' (history of objective values), 'best' (best parameters array), 'bestobj' (best objective value).
        """
        assert(self.Objective is not None)
        samples = []  # Parameter samples
        objvalues = []  # Objective values

        current = param_ini.copy()                # first step
        current_obj = self.Objective(current, **self.ObjectiveInfo)
        pmode = current_obj + 0.0  # record current best value
        cmode = current + 0.0  # best parameter sample

        samples.append(current)
        objvalues.append(current_obj)

        # Loop over Optimization steps
        for istep in range(nsteps):
            current = self.stepper(current)

            current_obj = self.Objective(current, **self.ObjectiveInfo)

            if current_obj <= pmode:
                pmode = current_obj
                cmode = current + 0.0

            samples.append(current)
            objvalues.append(current_obj)

            if((istep + 2) % (nsteps / 10) == 0) or istep == nsteps - 2:
                print('%d / %d completed, obj. value %lg' % (istep + 2, nsteps, pmode))

        results = {
            'samples' : np.array(samples),
            'objvalues' : np.array(objvalues),
            'best' : cmode,
            'bestobj' : pmode
            }

        return results

