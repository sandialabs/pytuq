#!/usr/bin/env python
"""Gradient descent optimization module."""

import numpy as np
from .optim import OptBase

class GD(OptBase):
    r""""Gradient descent optimization class.
    """
    def __init__(self, step_size=0.01):
        r"""Initialization.

        Args:
            step_size (float, optional): Step size parameter. Defaults to 0.01.
        """
        super().__init__()

        self.step_size = step_size

    def stepper(self, current, istep):
        """Sampler function of a single step.

        Args:
            current (np.ndarray): The current state.
            istep (int): Current step.

        Returns:
            tuple(float, float, float): New state.
        """
        assert(self.ObjectiveGrad is not None)

        newstate = current.copy()


        newstate -= self.step_size * self.ObjectiveGrad(current, **self.ObjectiveInfo)



        return newstate


