#!/usr/bin/env python
"""
1d benchmark functions module.

Most of the functions are taken from https://github.com/Vahanosi4ek/pytuq_funcs.
"""
import sys
import numpy as np

from .func import Function

################################################################################
################################################################################
################################################################################

class TFData(Function):
    r"""TensorFlow function

    Data generating toy model inspired by https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb#scrollTo=5zCEYpzu7bDX .

    .. math::
        f(x)=w_0 x (1 + \sin(x)) + b_0

    Default constant values are :math:`w_0 = 0.125`, :math:`b_0 = 5.0`, :math:`a = -20.0`, :math:`b = 60.0`.
    """

    def __init__(self, name='tfdata'):
        super().__init__()
        self.name = name

        self.dim = 1
        self.outdim = 1

        self.w0 = 0.125
        self.b0 = 5.
        self.a = -20.
        self.b = 60.

        self.setDimDom(domain=np.array([[self.a, self.b]]))

        return

    def __call__(self, x):

        y = (self.w0 * x * (1. + np.sin(x)) + self.b0)

        return y

    def grad(self, x):

        dy = self.w0 * (1. + np.sin(x) + x * np.cos(x))

        return dy.reshape(-1, self.outdim, self.dim)

################################################################################
################################################################################
################################################################################

class SineSum(Function):
    r"""Simple sum of sines

    Problem 02 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem02]

    .. math::
        f(x)=\sin(c_1x)+\sin(c_2x)


    Default constant values are :math:`c = (1., 10./3.)`.

    """
    def __init__(self, c1=1., c2=10./3., name="SineSum"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([2.7, 7.5]))

    def __call__(self, x):
        return np.sin(self.c1 * x) + np.sin(self.c2 * x)

    def grad(self, x):
        return (self.c1 * np.cos(self.c1 * x) + self.c2 * np.cos(self.c2 * x))[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class SineSum2(Function):
    r"""A more complex sum of sines

    Problem 03 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem03]

    .. math::
        f(x)=-\sum_{k=1}^{c_1}k\sin((k+1)x+k)


    Default constant value is :math:`c = 6`.

    """
    def __init__(self, c1=6, name="SineSum2"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

    def __call__(self, x):
        summation = np.zeros((x.shape[0], self.c1))
        k = np.broadcast_to(np.arange(1, self.c1 + 1), (x.shape[0], self.c1))
        summation += k * np.sin((k + 1) * x + k)

        return -np.sum(summation, axis=1, keepdims=True)

    def grad(self, x):
        _ = self.__call__(x)
        summation = np.zeros((x.shape[0], self.c1))
        k = np.broadcast_to(np.arange(1, self.c1 + 1), (x.shape[0], self.c1))
        summation += k * (k + 1) * np.cos((k + 1) * x + k)

        return -np.sum(summation, axis=1, keepdims=True)[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class QuadxExp(Function):
    r"""Product of quadratic and exponent

    Problem 04 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem04]

    .. math::
        f(x)=-(c_1x^2+c_2x+c_3)e^{-x}


    Default constant values are :math:`c = (16., -24., 5.)`.

    """
    def __init__(self, c1=16., c2=-24., c3=5., name="QuadxExp"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([1.9, 3.9]))

    def __call__(self, x):
        self._quad = self.c1 * x ** 2 + self.c2 * x + self.c3
        self._exp = np.exp(-x)
        return (-self._quad * self._exp)

    def grad(self, x):
        _ = self.__call__(x)
        return -(self._quad * -self._exp + self._exp * (2 * self.c1 * x + self.c2))[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class LinxSin(Function):
    r"""Product of linear and sine functions

    Problem 05 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem05]

    .. math::
        f(x)=-(c_1-c_2x)sin(c_3x)


    Default constant values are :math:`c = (1.4, -3., 18.)`.

    """
    def __init__(self, c1=1.4, c2=-3., c3=18., name="LinxSin"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.2]))

    def __call__(self, x):
        self._linear = (self.c1 + self.c2 * x)
        self._sine = np.sin(self.c3 * x)
        return -self._linear * self._sine

    def grad(self, x):
        _ = self.__call__(x)
        return -(self._linear * self.c3 * np.cos(self.c3 * x) + self._sine * self.c2)[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class SinexExp(Function):
    r"""Product of sine and exp functions

    Problem06 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem06]

    .. math::
        f(x)=-(x+\sin(x))e^{-x^2}

    """
    def __init__(self, name="SinexExp"):
        super().__init__(name=name)

        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

    def __call__(self, x):
        self._sine = (x + np.sin(x))
        self._exp = np.exp(-x ** 2)
        return -self._sine * self._exp

    def grad(self, x):
        _ = self.__call__(x)
        return -(self._sine * (-self._exp * 2 * x) + self._exp * (np.cos(x) + 1))[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class SineLogSum(Function):
    r"""Sum of sine and log functions

    Problem07 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem07]

    .. math::
        f(x)=\sin(c_1x) + \sin(c_2x) + \log_{c_3}(x) + c_4x + c_5


    Default constant values are :math:`c = (1., 10/3, e, -0.84, 3.)`.

    """
    def __init__(self, c1=1., c2=10/3, c3=np.exp(1), c4=-.84, c5=3., name="SineLogSum"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([2.7, 7.5]))

    def __call__(self, x):
        return np.sin(self.c1 * x) + np.sin(self.c2 * x) + np.emath.logn(self.c3, x) + self.c4 * x + self.c5

    def grad(self, x):
        return (self.c1 * np.cos(self.c1 * x) + self.c2 * np.cos(self.c2 * x) + 1 / (x * np.log(self.c3)) + self.c4)[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class CosineSum(Function):
    r"""Simple sum of cosines

    Problem 08 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem08]

    .. math::
        f(x)=-\sum_{k=1}^{c_1}k\cos((k+1)x+k)


    Default constant value is :math:`c = 6`.

    """
    def __init__(self, c1=6, name="CosineSum"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

    def __call__(self, x):
        summation = np.zeros((x.shape[0], self.c1))
        k = np.broadcast_to(np.arange(1, self.c1 + 1), (x.shape[0], self.c1))
        summation += k * np.cos((k + 1) * x + k)

        return -np.sum(summation, axis=1, keepdims=True)

    def grad(self, x):
        summation = np.zeros((x.shape[0], self.c1))
        k = np.broadcast_to(np.arange(1, self.c1 + 1), (x.shape[0], self.c1))
        summation -= k * (k + 1) * np.sin((k + 1) * x + k)

        return -np.sum(summation, axis=1, keepdims=True)[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class Sinex(Function):
    r"""Product of x and sine function

    Problem10 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem10]

    .. math::
        f(x)=-x\sin(x)

    """
    def __init__(self, name="Sinex"):
        super().__init__(name=name)

        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

    def __call__(self, x):
        return -x * np.sin(x)

    def grad(self, x):
        return (-x * np.cos(x) - np.sin(x))[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class CosineSum2(Function):
    r"""Simple sum of cosines

    Problem11 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem11]

    .. math::
        f(x)=c_1\cos(x) + \cos(c_2x)


    Default constant values are :math:`c = (2., 2.)`.

    """
    def __init__(self, c1=2., c2=2., name="CosineSum2"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-np.pi / 2, np.pi * 2]))

    def __call__(self, x):
        return self.c1 * np.cos(x) + np.cos(self.c2 * x)

    def grad(self, x):
        return (-self.c1 * np.sin(x) - self.c2 * np.sin(self.c2 * x))[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class Sinusoidal(Function):
    r"""Simple 1d sine function

    Sinusoidal [https://www.sfu.ca/~ssurjano/curretal88sin.html]

    .. math::
        f(x)=\sin(c_1\pi(x-c_2))

    Default constant values are :math:`c = (2., 0.1)`.

    """
    def __init__(self, c1=2., c2=0.1, name="Sinusoidal"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):
        return np.sin(self.c1 * np.pi * (x - self.c2))

    def grad(self, x):
        return (np.cos(self.c1 * np.pi * (x - self.c2)) * self.c1 * np.pi)[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class Forrester(Function):
    r"""Forrester function

    Forrester [https://www.sfu.ca/~ssurjano/forretal08.html]

    .. math::
        f(x)=(c_1x-c_2)^2\sin(c_3x-c_4)

    Default constant values are :math:`c = (6., 2., 12., 4)`.

    """
    def __init__(self, c1=6., c2=2., c3=12., c4=4., name="Forrester"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):
        return (self.c1 * x - self.c2) ** 2 * np.sin(self.c3 * x - self.c4)

    def grad(self, x):
        return (2 * self.c1 * (self.c1 * x - self.c2) * np.sin(self.c3 * x - self.c4) + (self.c1 * x - self.c2) ** 2 * self.c3 * np.cos(self.c3 * x - self.c4))[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class GramacyLee2(Function):
    r"""Complicated oscillatory 1d function

    Gramacy and Lee (2012) [https://www.sfu.ca/~ssurjano/grlee12.html]

    .. math::
        f(x)=\frac{\sin(c_1\pi x)}{c_2x}+(x-c_3)^{c_4}

    Default constant values are :math:`c = (10., 2., 1., 4.)`.

    """
    def __init__(self, c1=10., c2=2., c3=1., c4=4., name="GramacyLee2"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0.5, 2.5]))

    def __call__(self, x):
        return np.sin(self.c1 * np.pi * x) / (self.c2 * x) + (x - self.c3) ** self.c4

    def grad(self, x):
        return (((self.c1 * np.pi * np.cos(self.c1 * np.pi * x) * self.c2 * x) - self.c2 * np.sin(self.c1 * np.pi * x)) / (self.c2 * x) ** 2 + self.c4 * (x - self.c3) ** (self.c4 - 1))[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class Higdon(Function):
    r"""Higdon function

    Higdon (2002) [https://www.sfu.ca/~ssurjano/hig02.html]

    .. math::
        f(x)=\sin(2\pi x/c_1) + c_2\sin(2\pi x/c_3)

    Default constant values are :math:`c = (10., 0.2, 2.5)`.

    """
    def __init__(self, c1=10., c2=0.2, c3=2.5, name="Higdon"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

    def __call__(self, x):
        return np.sin(2 * np.pi * x / self.c1) + self.c2 * np.sin(2 * np.pi * x / self.c3)

    def grad(self, x):
        return (2 * np.pi / self.c1 * np.cos(2 * np.pi * x / self.c1) + 2 * np.pi * self.c2 / self.c3 * np.cos(2 * np.pi * x / self.c3))[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class Holsclaw(Function):
    r"""Holsclaw function

    Holsclaw et al. [https://www.sfu.ca/~ssurjano/holsetal13sin.html]

    .. math::
        f(x)=\frac{x\sin(x)}{c_1}

    Default constant value is :math:`c = 10.0`.

    """
    def __init__(self, c1=10., name="Holsclaw"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 1
        self.outdim = 1


        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

    def __call__(self, x):
        return x * np.sin(x) / self.c1

    def grad(self, x):
        return ((np.sin(x) + x * np.cos(x)) / self.c1)[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class DampedCosine(Function):
    r"""A simple 1d cosine function

    Damped Cosine [https://www.sfu.ca/~ssurjano/santetal03dc.html]

    .. math::
        f(x)=e^{c_1x}\cos(c_2\pi x)

    Default constant values are :math:`c = (-1.4, 3.5)`.

    """
    def __init__(self, c1=-1.4, c2=3.5, name="DampedCosine"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 1
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):
        return np.exp(self.c1 * x) * np.cos(self.c2 * np.pi * x)

    def grad(self, x):
        return (self.c1 * self.__call__(x) - self.c2 * np.pi * np.exp(self.c1 * x) * np.sin(self.c2 * np.pi * x))[:, np.newaxis, :]
