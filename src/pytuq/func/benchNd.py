#!/usr/bin/env python
"""
Nd benchmark functions module.

Most of the functions are taken from https://github.com/Vahanosi4ek/pytuq_funcs that autogenerates the codes given function's latex strings.
"""
import sys
import numpy as np

from scipy.special import factorial
from scipy.stats import multivariate_normal

from .func import Function


################################################################################
################################################################################
################################################################################

class Ackley(Function):
    r"""
    Ackley function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Ackley]

    Complex cosine function with many local minima

    .. math::
        f(x)=-c_1e^{-c_2\sqrt{\sum_{i=1}^{d}x_1^2}}-e^{\frac{1}{d}\sum_{i=1}^{d}\cos(c_3x_i)}+c_4+e


    Default constant values are :math:`c = (20., 0.2, 2\pi)` and :math:`d = 2`.
    """
    def __init__(self, c1=20., c2=0.2, c3=2*np.pi, c4=20., d=2, name="Ackley"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.d = c1, c2, c3, c4, d
        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-32, 32]))

    def __call__(self, x):
        t1 = np.exp(-self.c2 * np.sqrt(np.sum(x ** 2, axis=1, keepdims=True) / self.d))
        t2 = np.exp(np.sum(np.cos(self.c3 * x), axis=1, keepdims=True) / self.d)
        return -self.c1 * t1 - t2 + self.c4 + np.exp(1.)

    def grad(self, x):
        t1 = -np.exp(-self.c2 * np.sqrt(np.sum(x ** 2, axis=1, keepdims=True) / self.d)) * self.c2 * x / (self.d * np.sqrt(np.sum(x ** 2, axis=1, keepdims=True) / self.d))
        t2 = -np.exp(np.sum(np.cos(self.c3 * x), axis=1, keepdims=True) / self.d) * self.c3 * np.sin(self.c3 * x) / self.d

        return (-self.c1 * t1 - t2)[:, np.newaxis, :]


################################################################################
################################################################################
################################################################################

class Alpine01(Function):
    r"""
    Alpine01 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Alpine01]

    A N-d multimodal function

    .. math::
        f(x)=\sum_{i=1}^n |x_i\sin(x_i)+c_1x_2|


    Default constant values are :math:`c_1 = 0.1`.
    """
    def __init__(self, c1=0.1, d=2, name="Alpine01"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

    def __call__(self, x):
        return np.sum(np.abs(x * np.sin(x) + self.c1 * x), axis=1, keepdims=True)

    def grad(self, x):
        return (np.sign(x * np.sin(x) + self.c1 * x) * (x * np.cos(x) + np.sin(x) + self.c1))[:, np.newaxis, :]


################################################################################
################################################################################
################################################################################

class Alpine02(Function):
    r"""
    Alpine02 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Alpine02]

    A N-d multimodal function

    .. math::
        f(x)=\prod_{i=1}^{n}\sqrt{x_i}\sin(x_i)
    """
    def __init__(self, d=2, name="Alpine02"):
        super().__init__(name=name)

        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

    def __call__(self, x):
        return np.prod(np.sqrt(x) * np.sin(x), axis=1, keepdims=True)

    def grad(self, x):
        inner = np.sqrt(x) * np.sin(x)
        inner_grad = np.sqrt(x) * np.cos(x) + 1 / 2 * x ** (-1 / 2) * np.sin(x)
        return (self.__call__(x) / inner * inner_grad)[:, np.newaxis, :]


################################################################################
################################################################################
################################################################################

class AMGM(Function):
    r"""
    AMGM function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.AMGM]

    Difference-squared between AM and GM

    .. math::
        f(x)=\left ( \frac{1}{n} \sum_{i=1}^{n} x_i - \sqrt[n]{ \prod_{i=1}^{n} x_i} \right )^2
    """
    def __init__(self, d=2, name="AMGM"):
        super().__init__(name=name)

        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

    def __call__(self, x):
        self._am = 1 / self.dim * np.sum(x, axis=1, keepdims=True)
        self._gm = np.power(np.prod(x, axis=1, keepdims=True), 1 / self.dim)
        return (self._am - self._gm) ** 2

    def grad(self, x):
        _ = self.__call__(x)
        return (2 * (self._am - self._gm) * (1 / self.dim - 1 / self.dim * np.power((np.prod(x, axis=1, keepdims=True)), -1 / self.dim) * np.prod(x, axis=1, keepdims=True) / x))[:, np.newaxis, :]



################################################################################
################################################################################
################################################################################

class Bohachevsky(Function):
    r"""
    Bohachevsky function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bohachevsky]

    A N-d multimodal function

    .. math::
        f(x)=\sum_{i=1}^{n-1}\left[x_i^2 + c_1x_{i+1}^2 - c_2\cos(c_3\pi x_i) - c_4\cos(c_5\pi x_{i+1}) + c_6\right]


    Default constant values are :math:`c = (2., 0.3, 3., 0.4, 4., 0.7)`.
    """
    def __init__(self, c1=2., c2=0.3, c3=3., c4=0.4, c5=4., c6=0.7, d=2, name="Bohachevsky"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.d = c1, c2, c3, c4, c5, c6, d
        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-15., 15.]))

    def __call__(self, x):
        x_shr = np.concatenate((x[:, 1:], x[:, :1]), axis=1)
        inner = x ** 2 + self.c1 * x_shr ** 2 - self.c2 * np.cos(self.c3 * np.pi * x) - self.c4 * np.cos(self.c5 * np.pi * x_shr) + self.c6
        return np.sum(inner[:, :-1], axis=1, keepdims=True)

    def grad(self, x):
        # Uses a trick where the first and last column cancel where we need them to.
        x1 = np.concatenate((x[:, :-1], np.zeros((x.shape[0], 1))), axis=1)
        x2 = np.concatenate((np.zeros((x.shape[0], 1)), x[:, 1:]), axis=1)
        return (2 * x1 + self.c2 * self.c3 * np.pi * np.sin(self.c3 * np.pi * x1) + self.c1 * 2 * x2 + self.c4 * self.c5 * np.pi * np.sin(self.c5 * np.pi * x2))[:, np.newaxis, :]




################################################################################
################################################################################
################################################################################

class Cigar(Function):
    r"""
    Cigar function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Cigar]

    A N-d multimodal function

    .. math::
        f(x)=x_1^2 + c_1 \sum_{i=2}^{n} x_i^2


    Default constant values are :math:`c = 10^3`
    """
    def __init__(self, c1=10 ** 3, d=4, name="Cigar"):
        super().__init__(name=name)

        self.c1, self.d = c1, d
        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-100., 100.]))

    def __call__(self, x):
        return (x[:, 0].reshape(-1, 1)**2+self.c1 * np.sum(x[:, 1:]**2, axis=1, keepdims=True))

    def grad(self, x):
        x_modified = np.concatenate((x[:, 0][:, np.newaxis], self.c1 * x[:, 1:]), axis=1)
        grad = 2 * x_modified

        return grad[:, np.newaxis, :]



################################################################################
################################################################################
################################################################################

class CosineMixture(Function):
    r"""
    CosineMixture function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture]

    A N-d multimodal function

    .. math::
        f(x)=-c_1 \sum_{i=1}^n \cos(c_2 \pi x_i) - \sum_{i=1}^n x_i^2


    Default constant values are :math:`c = (0.1, 5.0)
    """
    def __init__(self, c1=0.1, c2=5.0, d=2, name="CosineMixture"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-1., 1.]))

    def __call__(self, x):
        return (-self.c1*np.sum(np.cos(self.c2*np.pi*x), axis=1)-np.sum(x**2.0, axis=1)).reshape(-1, 1)

    def grad(self, x):
        grad = self.c1*self.c2*np.pi*np.sin(self.c2*np.pi*x)-2*x

        return grad[:, np.newaxis, :]


################################################################################
################################################################################
################################################################################

class Griewank(Function):
    r"""
    Griewank function

    Reference: [https://www.sfu.ca/~ssurjano/griewank.html]

    .. math::
        f(x)=\sum_{i=1}^{d}\frac{x_i^2}{c_1}-\prod_{i=1}^{d}\cos(\frac{x_i}{\sqrt{i}}) + c_2


    Default constant values are :math:`c = (4000., 1.)` and :math:`d = 2`.
    """
    def __init__(self, c1=4000., c2=1., d=2, name="Griewank"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-600., 600.]))

    def __call__(self, x):
        self._term1 = np.sum(x ** 2 / self.c1, axis=1, keepdims=True)
        self._term2 = 1.
        for i in range(1, self.dim + 1):
            self._term2 *= np.cos(x[:, i - 1] / np.sqrt(i))
        self._term2 = self._term2

        return self._term1 - self._term2[:, np.newaxis] + self.c2

    def grad(self, x):
        _ = self.__call__(x)
        term1_grad = 2 * x / self.c1
        term2_grad = np.zeros((x.shape[0], self.dim))
        for i in range(1, self.dim + 1):
            term2_grad[:, i - 1] = -self._term2 / np.cos(x[:, i - 1] / np.sqrt(i)) * np.sin(x[:, i - 1] / np.sqrt(i)) / np.sqrt(i)

        return (term1_grad - term2_grad)[:, np.newaxis, :]



################################################################################
################################################################################
################################################################################

class Mishra07(Function):
    r"""
    Mishra07 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra07]

    A multimodal minimization function

    .. math::
        f(x)=\left(\prod_{i=1}^d x_i-d!\right)^2
    """
    def __init__(self, d=2, name="Mishra07"):
        super().__init__(name=name)

        self.dim = d
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-1., 1.]))

    def __call__(self, x):
        return ((np.prod(x,axis=1)-factorial(self.dim))**2).reshape(-1,1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        for i in range(self.dim):
            grad[:, 0, i] = 2.0 * (np.prod(x,axis=1)-factorial(self.dim)) * np.prod(x,axis=1)/x[:, i]

        return grad

################################################################################
################################################################################
################################################################################

class MVN(Function):
    r"""Multivariate Normal function.

    Reference: [https://en.wikipedia.org/wiki/Multivariate_normal_distribution]

    .. math::
        f(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)
    """
    def __init__(self, mean=[0., 0.], cov=[[1., 0.], [0., 1.]], name='MVN'):
        super().__init__()

        self.mean = np.array(mean, dtype=float)
        self.cov = np.array(cov, dtype=float)

        f = 4.0
        domain = np.tile(self.mean.reshape(-1,1), (1,2))
        stds = np.sqrt(np.diag(self.cov))
        domain[:,0] -= f * stds
        domain[:,1] += f * stds

        self.setDimDom(domain=domain)
        self.name = name

        self.outdim = 1

        return

    def __call__(self, x):
        self.checkDim(x)

        yy = multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

        return yy.reshape(-1,1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        invcov = np.linalg.inv(self.cov)

        mvn_vals = self.__call__(x)[:,0]

        for j in range(self.dim):
            diff = (x[:, j] - self.mean[j])
            grad[:, 0, j] = - mvn_vals * np.dot(invcov[j, :], (x - self.mean).T)

        return grad


################################################################################
################################################################################
################################################################################


class NegAlpineN2(Function):
    r"""Negative Alpine function.

    ... math::
        f(x) = - \sqrt{\prod_{i=1}^{d} x_i \cdot \sin(x_i)}
    """
    def __init__(self, name='Alpine N2', dim=2):
        super().__init__()
        self.setDimDom(domain=np.array(np.tile([0.001, 10.], (dim, 1))))
        self.name = name
        self.outdim = 1

        return

    def __call__(self, x):

        self.checkDim(x)

        yy = - np.sqrt(np.prod(x, axis=1)) * np.prod(np.sin(x), axis=1)

        return yy.reshape(-1, 1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        for j in range(self.dim):
            grad[:, 0, j] = (0.5 / x[:, j] + 1./np.tan(x[:, j])) * self.__call__(x)[:,0]

        return grad

################################################################################
################################################################################
################################################################################

class Sobol(Function):
    r"""Sobol function

    Reference: [https://www.sfu.ca/~ssurjano/gfunc.html]

    .. math::
        f(x) = \prod_{i=1}^{d} \frac{|4x_i -2| + a_i}{1 + a_i}
    """
    def __init__(self, name='Sobol', dim=5):
        super().__init__()
        self.name = name
        self.dim = dim
        self.outdim = 1

        self.setDimDom(domain = np.ones((self.dim, 1)) * np.array([0., 1.]))
        self.a = np.array([(i-2.)/2. for i in range(1,self.dim+1)])


    def __call__(self, x):
        sam = x.shape[0]
        self.checkDim(x)

        ydata=np.empty((sam, self.outdim))
        for j in range(sam):
            val=1.
            for k in range(self.dim):
                val *= ( (abs(4.*x[j,k]-2.)+self.a[k])/(1.+self.a[k]) )
            ydata[j,0]=val

        return ydata

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        for i in range(self.dim):
            partial = 1. / (1. + self.a[i]) * 4. * np.sign(4. * x[:, i] - 2.)
            for j in range(self.dim):
                if j != i:
                    partial *= ( (abs(4.*x[:,j]-2.)+self.a[j])/(1.+self.a[j]) )
            grad[:, 0, i] = partial

        return grad



################################################################################
################################################################################
################################################################################


class SumSquares(Function):
    r"""SumSquares function.

    .. math::
        f(x) = \sum_{i=1}^{d} (1 + i) x_i^2

    """
    def __init__(self, name='SumSquares', dim=5):
        super().__init__()
        self.setDimDom(domain=np.array(np.tile([-10.0, 10.0], (dim, 1))))
        self.name = name
        self.outdim = 1

        return

    def __call__(self, x):

        self.checkDim(x)

        yy = np.zeros((x.shape[0], self.outdim))
        for j in range(self.dim):
            yy[:,0] += (1. + j) * x[:, j]**2

        return yy.reshape(-1,1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        for j in range(self.dim):
            grad[:, 0, j] = 2.0 * (1. + j) * x[:, j]

        return grad



