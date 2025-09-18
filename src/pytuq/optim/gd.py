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

    def stepper(self, current):
        """Stepper function of a single step.

        Args:
            current (np.ndarray): The current state.

        Returns:
            np.ndarray: New state.
        """
        assert(self.ObjectiveGrad is not None)

        newstate = current.copy()


        newstate -= self.step_size * self.ObjectiveGrad(current, **self.ObjectiveInfo)



        return newstate


# TODO: this needs to be more tested and perhaps made general for any algorithm.
class SGD(OptBase):
    r""""Stochastic gradient descent optimization class.
    """
    def __init__(self, step_size=0.01, batch_size=1):
        """Initialization.

        Args:
            step_size (float, optional): Step size parameter. Defaults to 0.01.
            batch_size (int, optional): Batch size. Defaults to 1.
        """
        super().__init__()

        self.step_size = step_size
        self.batch_size = batch_size


    def stepper(self, current):
        """Stepper function of a single step.

        Args:
            current (np.ndarray): The current state.

        Returns:
            np.ndarray: New state.
        """
        assert(self.ObjectiveGrad is not None)

        newstate = current.copy()

        assert(self.ObjectiveInfo is not None)
        assert('xy_data' in self.ObjectiveInfo)
        xydata_save = self.ObjectiveInfo['xy_data']
        n = xydata_save[0].shape[0]

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            xdata = self.ObjectiveInfo['xy_data'][0][start:end]
            ydata = self.ObjectiveInfo['xy_data'][1][start:end]
            del self.ObjectiveInfo['xy_data']
            newstate -= self.step_size * self.ObjectiveGrad(current, xy_data=(xdata, ydata), **self.ObjectiveInfo)

        self.ObjectiveInfo['xy_data'] = xydata_save

        return newstate


class Adam(OptBase):
    """Simple Adam optimizer that performs gradient descent.
    """
    def __init__(self, dim, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.dim = dim
        self.m = np.zeros(self.dim, dtype=float)
        self.v = np.zeros(self.dim, dtype=float)
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def stepper(self, current):
        """Stepper function of a single step.

        Args:
            current (np.ndarray): The current state.

        Returns:
            np.ndarray: New state.
        """
        assert(self.ObjectiveGrad is not None)
        newstate = current.copy()
        grad = self.ObjectiveGrad(current, **self.ObjectiveInfo)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        newstate -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


        return newstate
