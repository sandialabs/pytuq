#!/usr/bin/env python
"""
Chemistry benchmark potential functions module.
"""
import numpy as np

from .func import Function


class LennardJones(Function):
    r"""Lennard Jones Potential

    Reference: [https://en.wikipedia.org/wiki/Lennard-Jones_potential]

    .. math::
        f(r) = \epsilon \left( \left(\frac{r_0}{r}\right)^{2n} - 2 \left(\frac{r_0}{r}\right)^n \right)

    Default parameters are :math:`\epsilon = 1.0`, :math:`r_0 = 1.0`, and :math:`n=6`.
    """
    def __init__(self, name='Lennard Jones', eps=1.0, r0=1.0, n=6):
        super().__init__()
        self.name = name

        self.eps = eps
        self.r0 = r0
        self.n = n

        self.setDimDom(domain=np.array([[self.r0 * (np.power(1. + np.sqrt(1.+2./self.eps), -1./self.n)), 2.0 * self.r0]]))
        self.outdim = 1


        return

    def __call__(self, x):


        self.checkDim(x)

        yy = self.eps* (np.power(self.r0 / x, 2.0 * self.n) - 2.0 * np.power(self.r0 / x, self.n))

        return yy.reshape(-1,1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        grad[:, 0, :] = - self.r0 * self.eps* (2.0 * self.n * np.power(self.r0 / x, 2.0 * self.n - 1) - 2.0 * self.n * np.power(self.r0 / x, self.n - 1.0)) * np.power(x, -2.0)

        return grad


class MullerBrown(Function):
    r"""Muller Brown Potential.

    Reference: [https://hunterheidenreich.com/notes/computational-chemistry/benchmark-problems/muller-brown-1979/]

    .. math::
        f(x,y) = \sum_{i=1}^{4} A_i \exp\left(a_i(x - x_{0i})^2 + b_i(x - x_{0i})(y - y_{0i}) + c_i(y - y_{0i})^2\right)

    """
    def __init__(self, name='Muller-Brown'):
        super().__init__()
        self.setDimDom(domain=np.array([[-1.5, 0.5], [0.0, 2.0]]))
        self.name = name
        self.outdim = 1


        self.factor = np.array([-200.0, -100.0, -170.0, 15.0])
        self.a = np.array([-1.0, -1.0, -6.5, 0.7])
        self.b = np.array([0.0, 0.0, 11.0, 0.6])
        self.c = np.array([-10.0, -10.0, -6.5, 0.7])
        self.x0 = np.array([1.0, 0.0, -0.5, -1.0])
        self.y0 = np.array([0.0, 0.5, 1.5, 1.0])

        return

    def __call__(self, x):

        #: Ensure x is of the right dimensionality
        self.checkDim(x)


        yy = np.zeros((x.shape[0], self.outdim))
        for i in range(4):
            yy[:,0] = yy[:,0] + self.factor[i] * np.exp(self.a[i] * (x[:, 0] - self.x0[i])**2 +
                                         self.b[i] * (x[:, 0] - self.x0[i]) * (x[:, 1] - self.y0[i]) +
                                         self.c[i] * (x[:, 1] - self.y0[i])**2)

        return yy

    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        for i in range(4):
            grad[:, 0, 0] = grad[:, 0, 0] + \
                            self.factor[i] * np.exp(self.a[i] * (x[:, 0] - self.x0[i])**2 +
                                                    self.b[i] * (x[:, 0] - self.x0[i]) *
                                                    (x[:, 1] - self.y0[i]) +
                                                    self.c[i] * (x[:, 1] - self.y0[i])**2) * \
                                                    (2. * self.a[i] * (x[:, 0] - self.x0[i]) +
                                                    self.b[i] * (x[:, 1] - self.y0[i]))
            grad[:, 0, 1] = grad[:, 0, 1] + \
                            self.factor[i] * np.exp(self.a[i] * (x[:, 0] - self.x0[i])**2 +
                                                    self.b[i] * (x[:, 0] - self.x0[i]) *
                                                    (x[:, 1] - self.y0[i]) +
                                                    self.c[i] * (x[:, 1] - self.y0[i])**2) * \
                                                    (2. * self.c[i] * (x[:, 1] - self.y0[i]) +
                                                    self.b[i] * (x[:, 0] - self.x0[i]))

        return grad


