#!/usr/bin/env python
"""
2d benchmark functions module.

Most of the functions are taken from https://github.com/Vahanosi4ek/pytuq_funcs that autogenerates the codes given function's latex strings.
"""
import sys
import numpy as np
from scipy.stats import multivariate_normal

from .func import Function


################################################################################
################################################################################
################################################################################

class Adjiman(Function):
    r"""Adjiman function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Adjiman]

    .. math::
        f(x)=\cos(x_1)\sin(x_2)-\frac{x_1}{x_2^2+c_1}

    Default constant value is :math:`c_1 = (1.0)`.

    """
    def __init__(self, c1=1., name="Adjiman"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-1., 2.,], [-1., 1.]]))

    def __call__(self, x):
        return (np.cos(x[:, 0]) * np.sin(x[:, 1]) - x[:, 0] / (x[:, 1] ** 2 + self.c1))[:, np.newaxis]

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        x1, x2 = x[:, 0], x[:, 1]

        grad[:, 0, 0] = np.sin(x2) * -np.sin(x1) - 1 / (x2 ** 2 + self.c1)
        grad[:, 0, 1] = np.cos(x1) * np.cos(x2) - x1 * -1 / (x2 ** 2 + self.c1) ** 2 * 2 * x2

        return grad


################################################################################
################################################################################
################################################################################

class BartelsConn(Function):
    r"""BartelsConn function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.BartelsConn]

    .. math::
        f(x)=|x_1^2+x_2^2+x_1x_2|+|\sin(x_1)|+|\cos(x_2)|

    """
    def __init__(self, name="BartelsConn"):
        super().__init__(name=name)

        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-50., 50.]))

    def __call__(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        self._t1 = x1 ** 2 + x2 ** 2 + x1 * x2
        self._t2 = np.sin(x1)
        self._t3 = np.cos(x2)
        return (np.abs(self._t1) + np.abs(self._t2) + np.abs(self._t3))[:, np.newaxis]

    def grad(self, x):
        _ = self.__call__(x)
        x1, x2 = x[:, 0], x[:, 1]
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = np.sign(self._t1) * (2 * x1 + x2) + np.sign(self._t2) * np.cos(x1)
        grad[:, 0, 1] = np.sign(self._t1) * (2 * x2 + x1) + np.sign(self._t3) * -np.sin(x2)
        return grad

################################################################################
################################################################################
################################################################################

class Bird(Function):
    r"""Bird function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bird]

    .. math::
        f(x)=(x_1-x_2)^2+e^{(c_1-\sin(x_1))^2}\cos(x_2)+e^{(c_2-\cos(x_2))^2}\sin(x_1)

    Default constant values are :math:`c = (1., 1.)`.

    """
    def __init__(self, c1=1., c2=1., name="Bird"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-2 * np.pi, 2 * np.pi]))

    def __call__(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        self._t1 = x1 - x2
        self._t2_1 = np.exp((self.c1 - np.sin(x1)) ** 2)
        self._t2_2 = np.cos(x2)
        self._t3_1 = np.exp((self.c2 - np.cos(x2)) ** 2)
        self._t3_2 = np.sin(x1)
        return (self._t1 ** 2 + self._t2_1 * self._t2_2 + self._t3_1 * self._t3_2)[:, np.newaxis]

    def grad(self, x):
        _ = self.__call__(x)
        x1, x2 = x[:, 0], x[:, 1]
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        _t2_1_grad = self._t2_1 * 2 * (self.c1 - np.sin(x1)) * -np.cos(x1)
        _t2_2_grad = -np.sin(x2)
        _t3_1_grad = self._t3_1 * 2 * (self.c2 - np.cos(x2)) * np.sin(x2)
        _t3_2_grad = np.cos(x1)
        grad[:, 0, 0] = 2 * self._t1 + self._t2_2 * _t2_1_grad + self._t3_1 * _t3_2_grad
        grad[:, 0, 1] = -2 * self._t1 + self._t2_1 * _t2_2_grad + self._t3_2 * _t3_1_grad
        return grad


################################################################################
################################################################################
################################################################################

class Branin(Function):
    """Branin function

    Reference: [https://www.sfu.ca/~ssurjano/branin.html]

    """
    def __init__(self, name='Branin'):
        super().__init__()
        self.setDimDom(domain=np.array([[-5., 10.], [0., 15.]]))
        self.name = name
        self.outdim = 1

        self.a_ = 1.
        self.b_ = 5.1/(4*np.pi**2)
        self.c_ = 5./np.pi
        self.r_ = 6.
        self.s_ = 10.
        self.t_ = 1./(8.*np.pi)

        return

    def __call__(self, x):
        self.checkDim(x)

        yy = self.a_ * (x[:, 1] - self.b_ * x[:, 0]**2 + self.c_ * x[:, 0] - self.r_)**2
        yy += self.s_*(1.-self.t_)*np.cos(x[:, 0])
        yy += self.s_

        return yy.reshape(-1,1)


    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = 2.0 * self.a_ * \
            (x[:, 1] - self.b_ * x[:, 0]**2 + self.c_ * x[:, 0] - self.r_) * \
            (-2.0 * self.b_ * x[:, 0] + self.c_) - \
            self.s_ * (1. - self.t_) * np.sin(x[:, 0])
        grad[:, 0, 1] = 2.0 * self.a_ * \
            (x[:, 1] - self.b_ * x[:, 0]**2 + self.c_ * x[:, 0] - self.r_)


        return grad

################################################################################
################################################################################
################################################################################


class Branin01(Function):
    r"""Branin01 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Branin01]

    .. math::
        f(x)=\left(-c_1 \frac{x_1^{2}}{\pi^{2}} + c_2 \frac{x_1}{\pi} + x_2 - c_3\right)^{2} + \left(c_4 - \frac{c_5}{c_6 \pi} \right) \cos\left(x_1\right) + c_7

    Default constant values are :math:`c = (1.275, 5., 6., 10., 5., 4., 10.)`.

    """
    def __init__(self, c1=1.275, c2=5., c3=6., c4=10., c5=5., c6=4., c7=10., name="Branin01"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7 = c1, c2, c3, c4, c5, c6, c7
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-5., 10.], [0., 15.]]))

    def __call__(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        self._t1 = -self.c1 * x1 ** 2 / np.pi ** 2 + self.c2 * x1 / np.pi + x2 - self.c3
        self._t2 = self.c4 - self.c5 / (self.c6 * np.pi)
        return (self._t1 ** 2 + self._t2 * np.cos(x1) + self.c7)[:, np.newaxis]

    def grad(self, x):
        _ = self.__call__(x)
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        x1, x2 = x[:, 0], x[:, 1]
        grad[:, 0, 0] = 2 * self._t1 * (-self.c1 * 2 / np.pi ** 2 * x1 + self.c2 / np.pi) - self._t2 * np.sin(x1)
        grad[:, 0, 1] = 2 * self._t1
        return grad

################################################################################
################################################################################
################################################################################

class Branin02(Function):
    r"""Branin02 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Branin02]

        .. math::
            f(x)=\left(- c_1 \frac{x_1^{2}}{\pi^{2}} + c_2 \frac{x_1}{\pi} + x_2 -c_3\right)^{2} + \left(c_4 - \frac{c_5}{c_6 \pi} \right) \cos\left(x_1\right) \cos\left(x_2\right) + \log(x_1^2+x_2^2 +c_7) + c_8


        Default constant values are :math:`c = (1.275, 5., 6., 10., 5., 4., 1., 10.)`.

        """
    def __init__(self, c1=1.275, c2=5., c3=6., c4=10., c5=5., c6=4., c7=1., c8=10., name="Branin02"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8 = c1, c2, c3, c4, c5, c6, c7, c8
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-5., 15.]))

    def __call__(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        self._t1 = -self.c1 * x1 ** 2 / np.pi ** 2 + self.c2 * x1 / np.pi + x2 - self.c3
        self._t2 = self.c4 - self.c5 / (self.c6 * np.pi)
        self._t3 = np.log(x1 ** 2 + x2 ** 2 + self.c7)
        return (self._t1 ** 2 + self._t2 * np.cos(x1) * np.cos(x2) + self._t3 + self.c8)[:, np.newaxis]

    def grad(self, x):
        _ = self.__call__(x)
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        x1, x2 = x[:, 0], x[:, 1]
        grad[:, 0, 0] = 2 * self._t1 * (-self.c1 * 2 / np.pi ** 2 * x1 + self.c2 / np.pi) - self._t2 * np.cos(x2) * np.sin(x1) + 1 / (x1 ** 2 + x2 ** 2 + self.c7) * 2 * x1
        grad[:, 0, 1] = 2 * self._t1 - self._t2 * np.cos(x1) * np.sin(x2) + 1 / (x1 ** 2 + x2 ** 2 + self.c7) * 2 * x2
        return grad

################################################################################
################################################################################
################################################################################

class Brent(Function):
    r"""Brent function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Brent]

        .. math::
            f(x)=(x_1 + 10)^2 + (x_2 + 10)^2 + e^{(-x_1^2-x_2^2)}


        Default constant values are :math:`c = (10., 10.)`.

        """
    def __init__(self, c1=10., c2=10., name="Brent"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

    def __call__(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return ((x1 + self.c1) ** 2 + (x2 + self.c2) ** 2 + np.exp(-x1 ** 2 - x2 ** 2))[:, np.newaxis]

    def grad(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return (2 * (x + self.c1) + np.exp(-x1 ** 2 - x2 ** 2)[:, np.newaxis] * -2 * x)[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class Bukin02(Function):
    r"""Bukin02 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin02]

        .. math::
            f(x)=c_1 (x_2 - c_2 x_1^2 + c_3) + c_4 (x_1 + c_5)^2


        Default constant values are :math:`c = (100.0, 0.01, 1.0, 0.01, 10.0)`

        """
    def __init__(self, c1=100.0, c2=0.01, c3=1.0, c4=0.01, c5=10.0, name="Bukin02"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-15., -5.], [-3., 3.]]))

    def __call__(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return (self.c1*(x[:, 1]-self.c2*x[:, 0]**2.0+self.c3)+self.c4*(x[:, 0]+self.c5)**2.0).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = -2.0*self.c2*self.c1*x[:, 0]+2.0*self.c4*(x[:, 0]+self.c5)
        grad[:, 0, 1] = self.c1

        return grad

################################################################################
################################################################################
################################################################################

class Bukin04(Function):
    r"""Bukin04 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin04]

        .. math::
            f(x)=c_1 x_2^{2} + c_2 |x_1 + c_3|


        Default constant values are :math:`c = (100.0, 0.01, 10.0)`

        """
    def __init__(self, c1=100.0, c2=0.01, c3=10.0, name="Bukin04"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-15.0, -5.0], [-3.0, 3.0]]))

    def __call__(self, x):
        return (self.c1*x[:, 1]**(2.0)+self.c2*np.abs(x[:, 0]+self.c3)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = self.c2*np.sign(x[:, 0]+self.c3)
        grad[:, 0, 1] = 2.0*self.c1*x[:, 1]

        return grad


################################################################################
################################################################################
################################################################################

class Bukin6(Function):
    r"""Bukin6 function

        .. math::
            f(x)=c_1\sqrt{|x_2-c_2 x_1^2|}+c_3|x_1+c_4|


        Default constant values are :math:`c = (100.0, 0.01, 0.01, 10.0)`

        """
    def __init__(self, c1=100.0, c2=0.01, c3=0.01, c4=10.0, name="Bukin6"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-15., -5.], [-3., 3.]]))

    def __call__(self, x):
        return (self.c1*np.sqrt(np.abs(x[:, 1]-self.c2*x[:, 0]**2.0))+self.c3*np.abs(x[:, 0]+self.c4)).reshape(-1, 1)

    def grad(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        x1t1 = self.c1 * -np.sign(x2 - self.c2 * x1 ** 2) / np.sqrt(np.abs(x2 - self.c2 * x1 ** 2)) * self.c2 * x1
        x1t2 = self.c3 * np.sign(x1 + self.c4)

        x2t1 = self.c1 * np.sign(x2 - self.c2 * x1 ** 2) / (2 * np.sqrt(np.abs(x2 - self.c2 * x1 ** 2)))

        grad[:, 0, 0] = x1t1 + x1t2
        grad[:, 0, 1] = x2t1

        return grad

################################################################################
################################################################################
################################################################################

class CarromTable(Function):
    r"""CarromTable function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CarromTable]

        .. math::
            f(x)=- c_1 \exp(c_2 |c_3 - \frac{\sqrt{x_1^{2} + x_2^{2}}}{\pi}|) \cos(x_1)^2 \cos(x_2)^2


        Default constant values are :math:`c = (1/30, 2.0, 1.0)`

        """
    def __init__(self, c1=1/30, c2=2.0, c3=1.0, name="CarromTable"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10.0, 10.0], [-10.0, 10.0]]))

    def __call__(self, x):
        return (-self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 0]**(2.0)+x[:, 1]**(2.0)))/(np.pi)))*np.cos(x[:, 0])**2.0*np.cos(x[:, 1])**2.0).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 0]**(2.0)+x[:, 1]**(2.0)))/(np.pi)))*self.c2*np.sign(self.c3-(np.sqrt(x[:, 0]**(2.0)+x[:, 1]**(2.0)))/(np.pi))*(1.0)/(2.0*np.pi*np.sqrt(x[:, 0]**2.0+x[:, 1]**2.0))*2.0*x[:, 0]*np.cos(x[:, 0])**2.0*np.cos(x[:, 1])**2.0+self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 0]**(2.0)+x[:, 1]**(2.0)))/(np.pi)))*2.0*np.cos(x[:, 0])*np.sin(x[:, 0])*np.cos(x[:, 1])**2.0
        grad[:, 0, 1] = self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 1]**(2.0)+x[:, 0]**(2.0)))/(np.pi)))*self.c2*np.sign(self.c3-(np.sqrt(x[:, 1]**(2.0)+x[:, 0]**(2.0)))/(np.pi))*(1.0)/(2.0*np.pi*np.sqrt(x[:, 1]**2.0+x[:, 0]**2.0))*2.0*x[:, 1]*np.cos(x[:, 1])**2.0*np.cos(x[:, 0])**2.0+self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 1]**(2.0)+x[:, 0]**(2.0)))/(np.pi)))*2.0*np.cos(x[:, 1])*np.sin(x[:, 1])*np.cos(x[:, 0])**2.0

        return grad


################################################################################
################################################################################
################################################################################

class ChengSandu(Function):
    r"""Cheng and Sandu 2d function

        .. math::
            f(x)=\cos(x_1+x_2)e^{x_1x_2}

        """
    def __init__(self, name="ChengSandu"):
        super().__init__(name=name)

        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):
        return (np.cos(x[:, 0] + x[:, 1]) * np.exp(x[:, 0] * x[:, 1]))[:, np.newaxis]

    def grad(self, x):
        exp = np.exp(x[:, 0] * x[:, 1])[:, np.newaxis]
        sin = np.sin(x[:, 0] + x[:, 1])[:, np.newaxis]
        cos = np.cos(x[:, 0] + x[:, 1])[:, np.newaxis]

        return ((x[:, ::-1] * cos - sin) * exp)[:, np.newaxis, :]

################################################################################
################################################################################
################################################################################

class Chichinadze(Function):
    r"""Chichinadze function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Chichinadze]

        .. math::
            f(x)= x_1^{2} - c_1 x_1 + c_2 \sin(c_3 \pi x_1) + c_4 \cos(c_5 \pi x_1) + c_6 - \frac{c_7}{\exp(c_8 (x_2 -c_9)^{2})}


        Default constant values are :math:`c = (12.0, 8.0, 2.5, 10.0, 0.5, 11.0, 0.2 * \sqrt{5}, 0.5, 0.5)`

        """
    def __init__(self, c1=12.0, c2=8.0, c3=2.5, c4=10.0, c5=0.5, c6=11.0, c7=0.2 * np.sqrt(5), c8=0.5, c9=0.5, name="Chichinadze"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9 = c1, c2, c3, c4, c5, c6, c7, c8, c9
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-30.0, 30.0], [-30.0, 30.0]]))

    def __call__(self, x):
        return (x[:, 0]**(2.0)-self.c1*x[:, 0]+self.c2*np.sin(self.c3*np.pi*x[:, 0])+self.c4*np.cos(self.c5*np.pi*x[:, 0])+self.c6-(self.c7)/(np.exp(self.c8*(x[:, 1]-self.c9)**(2.0)))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = 2.0*x[:, 0]-self.c1+self.c2*self.c3*np.pi*np.cos(self.c3*np.pi*x[:, 0])-self.c4*self.c5*np.pi*np.sin(self.c5*np.pi*x[:, 0])
        grad[:, 0, 1] = (self.c7)/(np.exp(self.c8*(x[:, 1]-self.c9)**(2.0))**2.0)*np.exp(self.c8*(x[:, 1]-self.c9)**(2.0))*self.c8*2.0*(x[:, 1]-self.c9)

        return grad


################################################################################
################################################################################
################################################################################

class CrossInTray(Function):
    r"""Cross-In-Tray function

        .. math::
            f(x)=-c_1(|\sin(x_1)\sin(x_2)|e^{|c_2-\frac{\sqrt{x_1^2+x_2^2}}{\pi}|}+c_3)^{c_4}


        Default constant values are :math:`c = (0.0001, 100., 1., 0.1)`.

        """
    def __init__(self, c1=0.0001, c2=100., c3=1., c4=0.1, name="CrossInTray"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

    def __call__(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        inner1 = np.sin(x1) * np.sin(x2)
        inner2 = np.exp(np.abs(self.c2 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))

        return (-self.c1 * (np.abs(inner1 * inner2) + self.c3) ** self.c4)[:, np.newaxis]

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        x1, x2 = x[:, 0], x[:, 1]

        dist = np.sqrt(x1 ** 2 + x2 ** 2)
        inner1x1 = np.cos(x1) * np.sin(x2)
        inner1x2 = np.cos(x2) * np.sin(x1)
        inner2x1 = -np.exp(np.abs(self.c2 - dist / np.pi)) * np.sign(self.c2 - dist / np.pi) * x1 / (np.pi * dist)
        inner2x2 = -np.exp(np.abs(self.c2 - dist / np.pi)) * np.sign(self.c2 - dist / np.pi) * x2 / (np.pi * dist)
        innerx1 = inner1x1 * np.exp(np.abs(self.c2 - dist / np.pi)) + np.sin(x1) * np.sin(x2) * inner2x1
        innerx2 = inner1x2 * np.exp(np.abs(self.c2 - dist / np.pi)) + np.sin(x1) * np.sin(x2) * inner2x2
        inner_absx1 = np.sign(np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(self.c2 - dist / np.pi)))) * innerx1
        inner_absx2 = np.sign(np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(self.c2 - dist / np.pi)))) * innerx2

        grad[:, 0, 0] = -self.c1 * self.c4 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(self.c2 - dist / np.pi))) + self.c3) ** (self.c4 - 1) * inner_absx1
        grad[:, 0, 1] = -self.c1 * self.c4 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(self.c2 - dist / np.pi))) + self.c3) ** (self.c4 - 1) * inner_absx2

        return grad

################################################################################
################################################################################
################################################################################

class Damavandi(Function):
    r"""Damavandi function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Damavandi]

        .. math::
            f(x)=\left[ c_1 - |\frac{\sin[\pi(x_1-c_2)]\sin[\pi(x_2-c_3)]}{\pi^2(x_1-c_4)(x_2-c_5)}|^{c_6} \right] \left[c_7 + (x_1-c_8)^2 + c_9(x_2-c_{10})^2 \right]


        Default constant values are :math:`c = (1.0, 2.0, 2.0, 2.0, 2.0, 5.0, 2.0, 7.0, 2.0, 7.0)`.

        """
    def __init__(self, c1=1.0, c2=2.0, c3=2.0, c4=2.0, c5=2.0, c6=5.0, c7=2.0, c8=7.0, c9=2.0, c10=7.0, name="Damavandi"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10 = c1, c2, c3, c4, c5, c6, c7, c8, c9, c10
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[0.0, 14.0], [0.0, 14.0]]))

    def __call__(self, x):
        return (((self.c1)-((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6)))*(((self.c7)+(((x[:, 0])-(self.c8))**2))+((self.c9)*(((x[:, 1])-(self.c10))**2)))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((-(((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6))*((self.c6)*((1/(np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5))))))*((np.sign(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))*((((((np.cos(np.pi*((x[:, 0])-(self.c2))))*(np.pi*1))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))*(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5))))-(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))*(((np.pi**2)*1)*((x[:, 1])-(self.c5)))))/((((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))**2)))))))*(((self.c7)+(((x[:, 0])-(self.c8))**2))+((self.c9)*(((x[:, 1])-(self.c10))**2))))+(((self.c1)-((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6)))*((((x[:, 0])-(self.c8))**2)*(2*((1/((x[:, 0])-(self.c8)))*1))))
        grad[:, 0, 1] = ((-(((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6))*((self.c6)*((1/(np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5))))))*((np.sign(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))*(((((np.sin(np.pi*((x[:, 0])-(self.c2))))*((np.cos(np.pi*((x[:, 1])-(self.c3))))*(np.pi*1)))*(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5))))-(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))*(((np.pi**2)*((x[:, 0])-(self.c4)))*1)))/((((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))**2)))))))*(((self.c7)+(((x[:, 0])-(self.c8))**2))+((self.c9)*(((x[:, 1])-(self.c10))**2))))+(((self.c1)-((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6)))*((self.c9)*((((x[:, 1])-(self.c10))**2)*(2*((1/((x[:, 1])-(self.c10)))*1)))))

        return grad

################################################################################
################################################################################
################################################################################

class DeckkersAarts(Function):
    r"""DeckkersAarts function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.DeckkersAarts]

        .. math::
            f(x)=c_1x_1^2 + x_2^2 - (x_1^2 + x_2^2)^2 + c_2(x_1^2 + x_2^2)^4


        Default constant values are :math:`c = (1000, 0.001)`.

        """
    def __init__(self, c1=1000, c2=0.001, name="DeckkersAarts"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-20.0, 20.0], [-20.0, 20.0]]))

    def __call__(self, x):
        return (((((self.c1)*(x[:, 0]**2))+(x[:, 1]**2))-(((x[:, 0]**2)+(x[:, 1]**2))**2))+((self.c2)*(((x[:, 0]**2)+(x[:, 1]**2))**4))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((self.c1)*(((x[:, 0])**2)*(2*((1/(x[:, 0]))*1))))-(((((x[:, 0])**2)+((x[:, 1])**2))**2)*(2*((1/(((x[:, 0])**2)+((x[:, 1])**2)))*(((x[:, 0])**2)*(2*((1/(x[:, 0]))*1)))))))+((self.c2)*(((((x[:, 0])**2)+((x[:, 1])**2))**4)*(4*((1/(((x[:, 0])**2)+((x[:, 1])**2)))*(((x[:, 0])**2)*(2*((1/(x[:, 0]))*1)))))))
        grad[:, 0, 1] = ((((x[:, 1])**2)*(2*((1/(x[:, 1]))*1)))-(((((x[:, 0])**2)+((x[:, 1])**2))**2)*(2*((1/(((x[:, 0])**2)+((x[:, 1])**2)))*(((x[:, 1])**2)*(2*((1/(x[:, 1]))*1)))))))+((self.c2)*(((((x[:, 0])**2)+((x[:, 1])**2))**4)*(4*((1/(((x[:, 0])**2)+((x[:, 1])**2)))*(((x[:, 1])**2)*(2*((1/(x[:, 1]))*1)))))))

        return grad

################################################################################
################################################################################
################################################################################

class EggCrate(Function):
    r"""EggCrate function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.EggCrate]

        .. math::
            f(x)=x_1^2 + x_2^2 + c_1 \left[ \sin^2(x_1) + \sin^2(x_2) \right]


        Default constant values are :math:`c = 25.0`.

        """
    def __init__(self, c1=25.0, name="EggCrate"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

    def __call__(self, x):
        return ((x[:, 0]**2+x[:, 1]**2)+(self.c1*(np.sin(x[:, 0])**2+np.sin(x[:, 1])**2))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (x[:, 0]**2*(2*(1/x[:, 0])))+(self.c1*(np.sin(x[:, 0])**2*(2*((1/np.sin(x[:, 0]))*np.cos(x[:, 0])))))
        grad[:, 0, 1] = (x[:, 1]**2*(2*(1/x[:, 1])))+(self.c1*(np.sin(x[:, 1])**2*(2*((1/np.sin(x[:, 1]))*np.cos(x[:, 1])))))

        return grad



################################################################################
################################################################################
################################################################################

class EggHolder(Function):
    r"""Egg Holder function

        Reference: [https://www.sfu.ca/~ssurjano/egg.html]

        .. math::
            f(x)=-(x_2+c_1)\sin(\sqrt{|x_2+\frac{x_1}{2}+c_2|})-x_1\sin(x_1-(x_2+c_3))


        Default constant values are :math:`c = (47., 47., 47.)`.

        Note: this is a modified version: the original's last term has square-root and absolute value in sine argument.
        """
    def __init__(self, c1=47., c2=47., c3=47., name='EggHolder'):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-512., 512.]))

    def __call__(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        self._term1_1 = x2 + self.c1
        self._term1_2 = np.sin(np.sqrt(np.abs(x2 + x1 / 2 + self.c2)))
        self._term2 = x1 * np.sin(x1 - (x2 + self.c3))
        return (-self._term1_1 * self._term1_2 - self._term2)[:, np.newaxis]

    def grad(self, x):
        _ = self.__call__(x)
        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        x1, x2 = x[:, 0], x[:, 1]
        x1_grad1 = self._term1_1 * np.cos(np.sqrt(np.abs(x2 + x1 / 2 + self.c2))) * np.sign(x2 + x1 / 2 + self.c2) / (4 * np.sqrt(np.abs(x2 + x1 / 2 + self.c2)))
        x1_grad2 = np.sin(x1 - (x2 + self.c3)) + x1 * np.cos(x1 - (x2 + self.c3))
        x2_grad1 = x1_grad1 * 2 + self._term1_2
        x2_grad2 = -x1 * np.cos(x1 - x2 - self.c3)
        grad[:, 0, 0] = -x1_grad1 - x1_grad2
        grad[:, 0, 1] = -x2_grad1 - x2_grad2

        return grad

################################################################################
################################################################################
################################################################################

class ElAttarVidyasagarDutta(Function):
    r"""ElAttarVidyasagarDutta function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.ElAttarVidyasagarDutta]

        .. math::
            f(x)=(x_1^2 + x_2 - c_1)^2 + (x_1 + x_2^2 - c_2)^2 + (x_1^2 + x_2^3 - c_3)^2


        Default constant values are :math:`c = (10.0, 7.0, 1.0)`.

        """
    def __init__(self, c1=10.0, c2=7.0, c3=1.0, name="ElAttarVidyasagarDutta"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-100, 100], [-100, 100]]))

    def __call__(self, x):
        return ((((x[:, 0]**2+x[:, 1])-self.c1)**2+((x[:, 0]+x[:, 1]**2)-self.c2)**2)+((x[:, 0]**2+x[:, 1]**3)-self.c3)**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((((x[:, 0]**2+x[:, 1])-self.c1)**2*(2*((1/((x[:, 0]**2+x[:, 1])-self.c1))*(x[:, 0]**2*(2*(1/x[:, 0]))))))+(((x[:, 0]+x[:, 1]**2)-self.c2)**2*(2*(1/((x[:, 0]+x[:, 1]**2)-self.c2)))))+(((x[:, 0]**2+x[:, 1]**3)-self.c3)**2*(2*((1/((x[:, 0]**2+x[:, 1]**3)-self.c3))*(x[:, 0]**2*(2*(1/x[:, 0]))))))
        grad[:, 0, 1] = ((((x[:, 0]**2+x[:, 1])-self.c1)**2*(2*(1/((x[:, 0]**2+x[:, 1])-self.c1))))+(((x[:, 0]+x[:, 1]**2)-self.c2)**2*(2*((1/((x[:, 0]+x[:, 1]**2)-self.c2))*(x[:, 1]**2*(2*(1/x[:, 1])))))))+(((x[:, 0]**2+x[:, 1]**3)-self.c3)**2*(2*((1/((x[:, 0]**2+x[:, 1]**3)-self.c3))*(x[:, 1]**3*(3*(1/x[:, 1]))))))

        return grad

################################################################################
################################################################################
################################################################################

class FreudensteinRoth(Function):
    r"""FreudensteinRoth function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_F.html#go_benchmark.FreudensteinRoth]

        .. math::
            f(x)=\left(x_1 - c_1 + \left[(c_2 - x_2)x_2 - c_3 \right] x_2 \right)^2 + \left (x_1 - c_4 + \left[(x_2 + c_5)x_2 - c_6 \right] x_2 \right)^2


        Default constant values are :math:`c = (13.0, 5.0, 2.0, 29.0, 1.0, 14.0)`.

        """
    def __init__(self, c1=13.0, c2=5.0, c3=2.0, c4=29.0, c5=1.0, c6=14.0, name="FreudensteinRoth"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return (((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1]))**2+((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1]))**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1]))**2*(2*(1/((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1])))))+(((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1]))**2*(2*(1/((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1])))))
        grad[:, 0, 1] = (((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1]))**2*(2*((1/((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1])))*((((-x[:, 1])+(self.c2-x[:, 1]))*x[:, 1])+(((self.c2-x[:, 1])*x[:, 1])-self.c3)))))+(((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1]))**2*(2*((1/((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1])))*(((x[:, 1]+(x[:, 1]+self.c5))*x[:, 1])+(((x[:, 1]+self.c5)*x[:, 1])-self.c6)))))

        return grad


################################################################################
################################################################################
################################################################################

class Franke(Function):
    r"""Franke function

    .. math::
        f(x) = 0.75 e^{-\frac{(9x_1-2)^2 + (9x_2-2)^2}{4}} + 0.75 e^{-\frac{(9x_1+1)^2}{49} - \frac{9x_2+1}{10}} + 0.5 e^{-\frac{(9x_1-7)^2 + (9x_2-3)^2}{4}} - 0.2 e^{-(9x_1-4)^2 - (9x_2-7)^2}

    """
    def __init__(self, name='Franke'):
        super().__init__()
        self.name = name
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain = np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):


        tt1 = 0.75*np.exp(-((9*x[:,0] - 2)**2 + (9*x[:,1] - 2)**2)/4.)
        tt2 = 0.75*np.exp(-((9*x[:,0] + 1)**2)/49 - (9*x[:,1] + 1)/10.)
        tt3 = 0.5*np.exp(-((9*x[:,0] - 7)**2 + (9*x[:,1] - 3)**2)/4.)
        tt4 = -0.2*np.exp(-(9*x[:,0] - 4)**2 - (9*x[:,1] - 7)**2)



        return (tt1 + tt2 + tt3 + tt4).reshape(-1,self.outdim)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        tt1 = 0.75*np.exp(-((9*x[:,0] - 2)**2 + (9*x[:,1] - 2)**2)/4.)
        tt2 = 0.75*np.exp(-((9*x[:,0] + 1)**2)/49 - (9*x[:,1] + 1)/10.)
        tt3 = 0.5*np.exp(-((9*x[:,0] - 7)**2 + (9*x[:,1] - 3)**2)/4.)
        tt4 = -0.2*np.exp(-(9*x[:,0] - 4)**2 - (9*x[:,1] - 7)**2)

        grad[:,0,0] = -2*(9*x[:,0] - 2)*9/4 * tt1 - 2*(9*x[:,0] + 1)*9/49 * tt2 + \
              -2*(9*x[:,0] - 7)*9/4 * tt3 - 2*(9*x[:,0] - 4)*9 * tt4
        grad[:,0,1] = -2*(9*x[:,1] - 2)*9/4 * tt1 - 9./10. * tt2 + \
              -2*(9*x[:,1] - 3)*9/4 * tt3 - 2*(9*x[:,1] - 7)*9 * tt4

        return grad



################################################################################
################################################################################
################################################################################

class GoldsteinPrice(Function):
    r"""GoldsteinPrice function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_G.html#go_benchmark.GoldsteinPrice]

        .. math::
            f(x)=\left[ 1+(x_1+x_2+1)^2(19-14x_1+3x_1^2-14x_2+6x_1x_2+3x_2^2) \right] \left[ 30+(2x_1-3x_2)^2(18-32x_1+12x_1^2+48x_2-36x_1x_2+27x_2^2) \right]

        """
    def __init__(self, name="GoldsteinPrice"):
        super().__init__(name=name)


        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-2, 2], [-2, 2]]))

    def __call__(self, x):
        return ((1+(((x[:, 0]+x[:, 1])+1)**2*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2))))*(30+(((2*x[:, 0])-(3*x[:, 1]))**2*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2))))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((((((x[:, 0]+x[:, 1])+1)**2*(2*(1/((x[:, 0]+x[:, 1])+1))))*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2)))+(((x[:, 0]+x[:, 1])+1)**2*(((-14)+(3*(x[:, 0]**2*(2*(1/x[:, 0])))))+(6*x[:, 1]))))*(30+(((2*x[:, 0])-(3*x[:, 1]))**2*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2)))))+((1+(((x[:, 0]+x[:, 1])+1)**2*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2))))*(((((2*x[:, 0])-(3*x[:, 1]))**2*(2*((1/((2*x[:, 0])-(3*x[:, 1])))*2)))*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2)))+(((2*x[:, 0])-(3*x[:, 1]))**2*(((-32)+(12*(x[:, 0]**2*(2*(1/x[:, 0])))))-(36*x[:, 1])))))
        grad[:, 0, 1] = ((((((x[:, 0]+x[:, 1])+1)**2*(2*(1/((x[:, 0]+x[:, 1])+1))))*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2)))+(((x[:, 0]+x[:, 1])+1)**2*(((-14)+(6*x[:, 0]))+(3*(x[:, 1]**2*(2*(1/x[:, 1])))))))*(30+(((2*x[:, 0])-(3*x[:, 1]))**2*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2)))))+((1+(((x[:, 0]+x[:, 1])+1)**2*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2))))*(((((2*x[:, 0])-(3*x[:, 1]))**2*(2*((1/((2*x[:, 0])-(3*x[:, 1])))*(-3))))*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2)))+(((2*x[:, 0])-(3*x[:, 1]))**2*((48-(36*x[:, 0]))+(27*(x[:, 1]**2*(2*(1/x[:, 1]))))))))

        return grad

################################################################################
################################################################################
################################################################################

class HimmelBlau(Function):
    r"""HimmelBlau function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_H.html#go_benchmark.HimmelBlau]

        .. math::
            f(x)=(x_1^2 + x_2 - c_1)^2 + (x_1 + x_2^2 - c_2)^2


        Default constant values are :math:`c = (11.0, 7.0)`.

        """
    def __init__(self, c1=11.0, c2=7.0, name="HimmelBlau"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-6, 6], [-6, 6]]))

    def __call__(self, x):
        return (((x[:, 0]**2+x[:, 1])-self.c1)**2+((x[:, 0]+x[:, 1]**2)-self.c2)**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((x[:, 0]**2+x[:, 1])-self.c1)**2*(2*((1/((x[:, 0]**2+x[:, 1])-self.c1))*(x[:, 0]**2*(2*(1/x[:, 0]))))))+(((x[:, 0]+x[:, 1]**2)-self.c2)**2*(2*(1/((x[:, 0]+x[:, 1]**2)-self.c2))))
        grad[:, 0, 1] = (((x[:, 0]**2+x[:, 1])-self.c1)**2*(2*(1/((x[:, 0]**2+x[:, 1])-self.c1))))+(((x[:, 0]+x[:, 1]**2)-self.c2)**2*(2*((1/((x[:, 0]+x[:, 1]**2)-self.c2))*(x[:, 1]**2*(2*(1/x[:, 1]))))))

        return grad

################################################################################
################################################################################
################################################################################

class Hosaki(Function):
    r"""Hosaki function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_H.html#go_benchmark.Hosaki]

        .. math::
            f(x)=\left ( c_1 - c_2x_1 + c_3x_1^2 - c_4x_1^3 + c_5x_1^4 \right )x_2^2e^{-x_1}


        Default constant values are :math:`c = (1.0, 8.0, 7.0, 8./3., 0.25)`.

        """
    def __init__(self, c1=1.0, c2=8.0, c3=7.0, c4=8./3., c5=0.25, name="Hosaki"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[0, 10], [0, 10]]))

    def __call__(self, x):
        return ((((((self.c1-(self.c2*x[:, 0]))+(self.c3*x[:, 0]**2))-(self.c4*x[:, 0]**3))+(self.c5*x[:, 0]**4))*x[:, 1]**2)*np.exp(-x[:, 0])).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((((((-self.c2)+(self.c3*(x[:, 0]**2*(2*(1/x[:, 0])))))-(self.c4*(x[:, 0]**3*(3*(1/x[:, 0])))))+(self.c5*(x[:, 0]**4*(4*(1/x[:, 0])))))*x[:, 1]**2)*np.exp(-x[:, 0]))+((((((self.c1-(self.c2*x[:, 0]))+(self.c3*x[:, 0]**2))-(self.c4*x[:, 0]**3))+(self.c5*x[:, 0]**4))*x[:, 1]**2)*(-np.exp(-x[:, 0])))
        grad[:, 0, 1] = (((((self.c1-(self.c2*x[:, 0]))+(self.c3*x[:, 0]**2))-(self.c4*x[:, 0]**3))+(self.c5*x[:, 0]**4))*(x[:, 1]**2*(2*(1/x[:, 1]))))*np.exp(-x[:, 0])

        return grad


################################################################################
################################################################################
################################################################################


class Keane(Function):
    r"""
    Keane function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_K.html#go_benchmark.Keane]

    A multimodal minimization function

    .. math::
        f(x)=\frac{\sin^2(x_1 - x_2)\sin^2(x_1 + x_2)}{\sqrt{x_1^2 + x_2^2}}

    """
    def __init__(self, name="Keane"):
        super().__init__(name=name)


        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[1.e-8, 10], [1.e-8, 10]]))

    def __call__(self, x):
        return ((np.sin(x[:, 0]-x[:, 1])**2*np.sin(x[:, 0]+x[:, 1])**2)/np.sqrt(x[:, 0]**2+x[:, 1]**2)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = 2.*np.sin(x[:, 0]-x[:, 1])*np.sin(x[:, 0]+x[:, 1])*np.sin(2.*x[:, 0])/np.sqrt(x[:, 0]**2+x[:, 1]**2) - self.__call__(x)[:, 0]*x[:, 0]/(x[:, 0]**2 + x[:, 1]**2)
        grad[:, 0, 1] = - 2.*np.sin(x[:, 0]-x[:, 1])*np.sin(x[:, 0]+x[:, 1])*np.sin(2.*x[:,1 ])/np.sqrt(x[:, 0]**2+x[:, 1]**2) - self.__call__(x)[:, 0]*x[:, 1]/(x[:, 0]**2 + x[:, 1]**2)

        return grad

################################################################################
################################################################################
################################################################################

class Leon(Function):
    r"""Leon function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_L.html#go_benchmark.Leon]

        .. math::
            f(x)= \left(1 - x_{1}\right)^{2} + c_1 \left(x_{2} - x_{1}^{2} \right)^{2}


        Default constant values are :math:`c = 100`.

        """
    def __init__(self, c1=100, name="Leon"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-1.2, 1.2], [-1.2, 1.2]]))

    def __call__(self, x):
        return ((1-x[:, 0])**2+(self.c1*(x[:, 1]-x[:, 0]**2)**2)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((1-x[:, 0])**2*(2*(-(1/(1-x[:, 0])))))+(self.c1*((x[:, 1]-x[:, 0]**2)**2*(2*((1/(x[:, 1]-x[:, 0]**2))*(-(x[:, 0]**2*(2*(1/x[:, 0]))))))))
        grad[:, 0, 1] = self.c1*((x[:, 1]-x[:, 0]**2)**2*(2*(1/(x[:, 1]-x[:, 0]**2))))

        return grad

################################################################################
################################################################################
################################################################################

class Levy13(Function):
    r"""Levy13 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_L.html#go_benchmark.Levy13]

        .. math::
            f(x)=\left(x_{1} -c_1\right)^{2} \left[\sin^{2}\left(c_2 \pi x_{2}\right) + c_3\right] + \left(x_{2} -c_4\right)^{2} \left[\sin^{2}\left(c_5 \pi x_{2}\right) + c_6\right] + \sin^{2}\left(c_7 \pi x_{1}\right)


        Default constant values are :math:`c = (1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 3.0)`.

        """
    def __init__(self, c1=1.0, c2=3.0, c3=1.0, c4=1.0, c5=2.0, c6=1.0, c7=3.0, name="Levy13"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7 = c1, c2, c3, c4, c5, c6, c7
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return ((((x[:, 0]-self.c1)**2*(np.sin((self.c2*np.pi)*x[:, 1])**2+self.c3))+((x[:, 1]-self.c4)**2*(np.sin((self.c5*np.pi)*x[:, 1])**2+self.c6)))+np.sin((self.c7*np.pi)*x[:, 0])**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((x[:, 0]-self.c1)**2*(2*(1/(x[:, 0]-self.c1))))*(np.sin((self.c2*np.pi)*x[:, 1])**2+self.c3))+(np.sin((self.c7*np.pi)*x[:, 0])**2*(2*((1/np.sin((self.c7*np.pi)*x[:, 0]))*(np.cos((self.c7*np.pi)*x[:, 0])*(self.c7*np.pi)))))
        grad[:, 0, 1] = ((x[:, 0]-self.c1)**2*(np.sin((self.c2*np.pi)*x[:, 1])**2*(2*((1/np.sin((self.c2*np.pi)*x[:, 1]))*(np.cos((self.c2*np.pi)*x[:, 1])*(self.c2*np.pi))))))+((((x[:, 1]-self.c4)**2*(2*(1/(x[:, 1]-self.c4))))*(np.sin((self.c5*np.pi)*x[:, 1])**2+self.c6))+((x[:, 1]-self.c4)**2*(np.sin((self.c5*np.pi)*x[:, 1])**2*(2*((1/np.sin((self.c5*np.pi)*x[:, 1]))*(np.cos((self.c5*np.pi)*x[:, 1])*(self.c5*np.pi)))))))

        return grad


################################################################################
################################################################################
################################################################################

class Lim(Function):
    r"""Lim function

        Generalized nonpolynomial trigonometric 2d function

        .. math::
            f(x)=a((b+cx_1\sin(dx_1))(f+e^{gx_2})+h)


        Default constant values are :math:`c = (1/6, 30., 5., 5., 4., -5., -100.)`.

        """
    def __init__(self, c1=1/6, c2=30., c3=5., c4=5., c5=4., c6=-5., c7=-100., name="Lim"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7 = c1, c2, c3, c4, c5, c6, c7
        self.dim = 2
        self.outdim = 1


        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):
        return (self.c1 * ((self.c2 + self.c3 * x[:, 0] * np.sin(self.c4 * x[:, 0])) * (self.c5 + np.exp(self.c6 * x[:, 1])) + self.c7))[:, np.newaxis]

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        x1, x2 = x[:, 0], x[:, 1]

        grad[:, 0, 0] = self.c1 * (self.c3 * (np.sin(self.c4 * x1) + self.c4 * x1 * np.cos(self.c4 * x1)) * (self.c5 + np.exp(self.c6 * x2)))
        grad[:, 0, 1] = self.c1 * ((self.c2 + self.c3 * x1 * np.sin(self.c4 * x1)) * self.c6 * np.exp(self.c6 * x2))

        return grad

################################################################################
################################################################################
################################################################################

class Matyas(Function):
    r"""Matyas function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Matyas]

        .. math::
            f(x)=c_1(x_1^2 + x_2^2) - c_2x_1x_2


        Default constant values are :math:`c = (0.26, 0.48)`.

        """
    def __init__(self, c1=0.26, c2=0.48, name="Matyas"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return ((self.c1*(x[:, 0]**2+x[:, 1]**2))-((self.c2*x[:, 0])*x[:, 1])).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (self.c1*(x[:, 0]**2*(2*(1/x[:, 0]))))-(self.c2*x[:, 1])
        grad[:, 0, 1] = (self.c1*(x[:, 1]**2*(2*(1/x[:, 1]))))-(self.c2*x[:, 0])

        return grad


################################################################################
################################################################################
################################################################################

class Mishra03(Function):
    r"""
    Mishra03 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra03]

    A multimodal minimization function

    .. math::
        f(x)=\sqrt{|\cos{\sqrt{x_1^2 + x_2^2}}|} + c_1(x_1 + x_2)


    Default constant value is :math:`c = 0.01`.
    """
    def __init__(self, c1=0.01, name="Mishra03"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return (np.sqrt(np.abs(np.cos(np.sqrt(x[:, 0]**2+x[:, 1]**2))))+self.c1*(x[:, 0]+x[:, 1])).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (-0.5*x[:,0]/(self.__call__(x)[:,0]-self.c1*(x[:, 0]+x[:, 1])))*np.sign(np.cos(np.sqrt(x[:, 0]**2+x[:, 1]**2)))*np.sin(np.sqrt(x[:, 0]**2+x[:, 1]**2))/np.sqrt(x[:, 0]**2+x[:, 1]**2)+self.c1
        grad[:, 0, 1] = (-0.5*x[:,1]/(self.__call__(x)[:,0]-self.c1*(x[:, 0]+x[:, 1])))*np.sign(np.cos(np.sqrt(x[:, 0]**2+x[:, 1]**2)))*np.sin(np.sqrt(x[:, 0]**2+x[:, 1]**2))/np.sqrt(x[:, 0]**2+x[:, 1]**2)+self.c1

        return grad



################################################################################
################################################################################
################################################################################

class Mishra04(Function):
    r"""
    Mishra04 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra04]

    A multimodal minimization function

    .. math::
        f(x)=\sqrt{|\sin{\sqrt{x_1^2 + x_2^2}}|} + c_1(x_1 + x_2)


    Default constant value is :math:`c = 0.01`.
    """
    def __init__(self, c1=0.01, name="Mishra04"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return (np.sqrt(np.abs(np.sin(np.sqrt(x[:, 0]**2+x[:, 1]**2))))+self.c1*(x[:, 0]+x[:, 1])).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (0.5*x[:,0]/(self.__call__(x)[:,0]-self.c1*(x[:, 0]+x[:, 1])))*np.sign(np.sin(np.sqrt(x[:, 0]**2+x[:, 1]**2)))*np.cos(np.sqrt(x[:, 0]**2+x[:, 1]**2))/np.sqrt(x[:, 0]**2+x[:, 1]**2)+self.c1
        grad[:, 0, 1] = (0.5*x[:,1]/(self.__call__(x)[:,0]-self.c1*(x[:, 0]+x[:, 1])))*np.sign(np.sin(np.sqrt(x[:, 0]**2+x[:, 1]**2)))*np.cos(np.sqrt(x[:, 0]**2+x[:, 1]**2))/np.sqrt(x[:, 0]**2+x[:, 1]**2)+self.c1

        return grad

################################################################################
################################################################################
################################################################################

class Mishra05(Function):
    r"""
    Mishra05 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra05]

    A multimodal minimization function

    .. math::
        f(x)=\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) + \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2 + c_1(x_1 + x_2)


    Default constant value is :math:`c = 0.01`.

    """
    def __init__(self, c1=0.01, name="Mishra05"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return (((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2+(self.c1*(x[:, 0]+x[:, 1]))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2*(2*((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0]))*(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 0])))))))))+(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 0])))))))))+1))))+self.c1
        grad[:, 0, 1] = (((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2*(2*((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0]))*((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 1])))))))))+(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 1]))))))))))))+self.c1

        return grad


################################################################################
################################################################################
################################################################################

class Mishra06(Function):
    r"""
    Mishra06 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra06]

    A multimodal minimization function

    .. math::
        f(x)=-\log{\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) - \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2} + c_1 \left[(x_1 -c_2)^2 + (x_2 - c_3)^2 \right]


    Default constant values are :math:`c = (0.01, 1.0, 1.0)`.

    """
    def __init__(self, c1=0.01, c2=1.0, c3=1.0, name="Mishra06"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return ((-np.log(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2))+(self.c1*((x[:, 0]-self.c2)**2+(x[:, 1]-self.c3)**2))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (-((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2)*(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2*(2*((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0]))*(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 0])))))))))-(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 0])))))))))+1))))))+(self.c1*((x[:, 0]-self.c2)**2*(2*(1/(x[:, 0]-self.c2)))))
        grad[:, 0, 1] = (-((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2)*(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2*(2*((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0]))*((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 1])))))))))-(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 1]))))))))))))))+(self.c1*((x[:, 1]-self.c3)**2*(2*(1/(x[:, 1]-self.c3)))))

        return grad



################################################################################
################################################################################
################################################################################

class McCormick(Function):
    r"""McCormick function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.McCormick]

        .. math::
            f(x)=- x_{1} + c_1 x_{2} + \left(x_{1} - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + c_2


        Default constant values are :math:`c = (2.0, 1.0)`.

        """
    def __init__(self, c1=2.0, c2=1.0, name="McCormick"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-1.5, 4], [-1.5, 4]]))

    def __call__(self, x):
        return (((((-x[:, 0])+(self.c1*x[:, 1]))+(x[:, 0]-x[:, 1])**2)+np.sin(x[:, 0]+x[:, 1]))+self.c2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((-1)+((x[:, 0]-x[:, 1])**2*(2*(1/(x[:, 0]-x[:, 1])))))+np.cos(x[:, 0]+x[:, 1])
        grad[:, 0, 1] = (self.c1+((x[:, 0]-x[:, 1])**2*(2*(-(1/(x[:, 0]-x[:, 1]))))))+np.cos(x[:, 0]+x[:, 1])

        return grad



################################################################################
################################################################################
################################################################################

class NewFunction03(Function):
    r"""NewFunction03 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_N.html#go_benchmark.NewFunction03]

        .. math::
            f(x)=c_1 x_{1} + c_2 x_{2} + \left[x_{1} + \sin^{2}\left[\left(\cos\left(x_{1}\right) + \cos\left(x_{2}\right)\right)^{2}\right] + \cos^{2}\left[\left(\sin\left(x_{1}\right) + \sin\left(x_{2}\right)\right)^{2}\right]\right]^{2}


        Default constant values are :math:`c = (0.01, 0.1)`.

        """
    def __init__(self, c1=0.01, c2=0.1, name="NewFunction03"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return (((self.c1*x[:, 0])+(self.c2*x[:, 1]))+((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = self.c1+(((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)**2*(2*((1/((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2))*((1+(np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 0]))))))))))+(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 0]))))))))))))
        grad[:, 0, 1] = self.c2+(((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)**2*(2*((1/((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2))*((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 1])))))))))+(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 1]))))))))))))

        return grad

################################################################################
################################################################################
################################################################################

class Parsopoulos(Function):
    r"""Parsopoulos function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Parsopoulos]

        .. math::
            f(x)=\cos(x_1)^2 + \sin(x_2)^2

        """
    def __init__(self, name="Parsopoulos"):
        super().__init__(name=name)


        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

    def __call__(self, x):
        return (np.cos(x[:, 0])**2+np.sin(x[:, 1])**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = np.cos(x[:, 0])**2*(2*((1/np.cos(x[:, 0]))*(-np.sin(x[:, 0]))))
        grad[:, 0, 1] = np.sin(x[:, 1])**2*(2*((1/np.sin(x[:, 1]))*np.cos(x[:, 1])))

        return grad


################################################################################
################################################################################
################################################################################

class Price01(Function):
    r"""Price01 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Price01]

        .. math::
            f(x)=(\abs{ x_1 } - c_1)^2 + (\abs{ x_2 } - c_2)^2


        Default constant values are :math:`c = (5, 5)`.

        """
    def __init__(self, c1=5, c2=5, name="Price01"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-500, 500], [-500, 500]]))

    def __call__(self, x):
        return ((np.abs(x[:, 0])-self.c1)**2+(np.abs(x[:, 1])-self.c2)**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (np.abs(x[:, 0])-self.c1)**2*(2*((1/(np.abs(x[:, 0])-self.c1))*np.sign(x[:, 0])))
        grad[:, 0, 1] = (np.abs(x[:, 1])-self.c2)**2*(2*((1/(np.abs(x[:, 1])-self.c2))*np.sign(x[:, 1])))

        return grad

################################################################################
################################################################################
################################################################################

class Price02(Function):
    r"""Price02 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Price02]

        .. math::
            f(x)=c_1 + \sin^2(x_1) + \sin^2(x_2) - c_2e^{(-x_1^2 - x_2^2)}


        Default constant values are :math:`c = (1.0, 0.1)`.

        """
    def __init__(self, c1=1.0, c2=0.1, name="Price02"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return (((self.c1+np.sin(x[:, 0])**2)+np.sin(x[:, 1])**2)-(self.c2*np.exp((-x[:, 0]**2)-x[:, 1]**2))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (np.sin(x[:, 0])**2*(2*((1/np.sin(x[:, 0]))*np.cos(x[:, 0]))))-(self.c2*(np.exp((-x[:, 0]**2)-x[:, 1]**2)*(-(x[:, 0]**2*(2*(1/x[:, 0]))))))
        grad[:, 0, 1] = (np.sin(x[:, 1])**2*(2*((1/np.sin(x[:, 1]))*np.cos(x[:, 1]))))-(self.c2*(np.exp((-x[:, 0]**2)-x[:, 1]**2)*(-(x[:, 1]**2*(2*(1/x[:, 1]))))))

        return grad

################################################################################
################################################################################
################################################################################

class Price03(Function):
    r"""Price03 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Price03]

        .. math::
            f(x)=c_1(x_2 - x_1^2)^2 + \left[c_2(x_2 - c_3)^2 - x_1 - c_4 \right]^2


        Default constant values are :math:`c = (100, 6.4, 0.5, 0.6)`.

        """
    def __init__(self, c1=100, c2=6.4, c3=0.5, c4=0.6, name="Price03"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-50, 50], [-50, 50]]))

    def __call__(self, x):
        return ((self.c1*(x[:, 1]-x[:, 0]**2)**2)+(((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4)**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (self.c1*((x[:, 1]-x[:, 0]**2)**2*(2*((1/(x[:, 1]-x[:, 0]**2))*(-(x[:, 0]**2*(2*(1/x[:, 0]))))))))+((((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4)**2*(2*(-(1/(((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4)))))
        grad[:, 0, 1] = (self.c1*((x[:, 1]-x[:, 0]**2)**2*(2*(1/(x[:, 1]-x[:, 0]**2)))))+((((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4)**2*(2*((1/(((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4))*(self.c2*((x[:, 1]-self.c3)**2*(2*(1/(x[:, 1]-self.c3))))))))

        return grad

################################################################################
################################################################################
################################################################################

class Price04(Function):
    r"""Price04 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Price04]

        .. math::
            f(x)=(c_1x_1^3x_2 - x_2^3)^2 + (c_2x_1 - x_2^2 + x_2)^2


        Default constant values are :math:`c = (2, 6)`.

        """
    def __init__(self, c1=2, c2=6, name="Price04"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-50, 50], [-50, 50]]))

    def __call__(self, x):
        return ((((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3)**2+(((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1])**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3)**2*(2*((1/(((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3))*((self.c1*(x[:, 0]**3*(3*(1/x[:, 0]))))*x[:, 1]))))+((((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1])**2*(2*((1/(((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1]))*self.c2)))
        grad[:, 0, 1] = ((((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3)**2*(2*((1/(((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3))*((self.c1*x[:, 0]**3)-(x[:, 1]**3*(3*(1/x[:, 1])))))))+((((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1])**2*(2*((1/(((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1]))*((-(x[:, 1]**2*(2*(1/x[:, 1]))))+1))))

        return grad

################################################################################
################################################################################
################################################################################

class Quadratic(Function):
    r"""Quadratic function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_Q.html#go_benchmark.Quadratic]

        .. math::
            f(x)=-3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2 + 203.64x_2^2 + 182.25x_1x_2

        """
    def __init__(self, name="Quadratic"):
        super().__init__(name=name)


        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return ((((((-3803)-(138*x[:, 0]))-(232*x[:, 1]))+(128*x[:, 0]**2))+(203*x[:, 1]**2))+((182*x[:, 0])*x[:, 1])).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((-138)+(128*(x[:, 0]**2*(2*(1/x[:, 0])))))+(182*x[:, 1])
        grad[:, 0, 1] = ((-232)+(203*(x[:, 1]**2*(2*(1/x[:, 1])))))+(182*x[:, 0])

        return grad


################################################################################
################################################################################
################################################################################

class Quadratic2d(Function):
    """2d Quadratic function

    .. math::
        f(x)=0.5 (x - c)^T H (x - c)

    where :math:`H` is the Hessian matrix, and :math:`c` is the center.

    """
    def __init__(self, center=[0., 0.], hess=[[1., 0.], [0., 1.]], name='Quadratic2d'):
        super().__init__()

        self.center = np.array(center, dtype=float)
        self.hess = np.array(hess, dtype=float)
        self.cov = np.linalg.inv(self.hess)

        f = 4.0
        domain = np.tile(self.center.reshape(-1,1), (1,2))
        stds = np.sqrt(np.diag(self.cov))
        domain[:,0] -= f * stds
        domain[:,1] += f * stds


        self.setDimDom(domain=domain)
        self.name = name
        self.outdim = 1

        return

    def __call__(self, x):
        self.checkDim(x)

        nsam = x.shape[0]
        yy = np.empty(nsam,)
        for i in range(nsam):
            yy[i] = 0.5 * np.dot(x[i, :]-self.center, np.dot(self.hess, x[i, :]-self.center))


        return yy.reshape(-1,1)

    def grad(self, x):

        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        for j in range(self.dim):
            for i in range(self.dim):
                grad[:, 0, j] += self.hess[j, i] * (x[:, i] - self.center[i])

        return grad



################################################################################
################################################################################
################################################################################

class RosenbrockModified(Function):
    r"""RosenbrockModified function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_R.html#go_benchmark.RosenbrockModified]

        .. math::
            f(x)=c_1 + c_2(x_2 - x_1^2)^2 + (c_3 - x_1)^2 - c_4 e^{-\frac{(x_1+1)^2 + (x_2 + 1)^2}{c_5}}


        Default constant values are :math:`c = (74, 100, 1, 400, 0.1)`.

        """
    def __init__(self, c1=74, c2=100, c3=1, c4=400, c5=0.1, name="RosenbrockModified"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-2, 2], [-2, 2]]))

    def __call__(self, x):
        return (((self.c1+(self.c2*(x[:, 1]-x[:, 0]**2)**2))+(self.c3-x[:, 0])**2)-(self.c4*np.exp(-(((x[:, 0]+1)**2+(x[:, 1]+1)**2)/self.c5)))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((self.c2*((x[:, 1]-x[:, 0]**2)**2*(2*((1/(x[:, 1]-x[:, 0]**2))*(-(x[:, 0]**2*(2*(1/x[:, 0]))))))))+((self.c3-x[:, 0])**2*(2*(-(1/(self.c3-x[:, 0]))))))-(self.c4*(np.exp(-(((x[:, 0]+1)**2+(x[:, 1]+1)**2)/self.c5))*(-((((x[:, 0]+1)**2*(2*(1/(x[:, 0]+1))))*self.c5)/self.c5**2))))
        grad[:, 0, 1] = (self.c2*((x[:, 1]-x[:, 0]**2)**2*(2*(1/(x[:, 1]-x[:, 0]**2)))))-(self.c4*(np.exp(-(((x[:, 0]+1)**2+(x[:, 1]+1)**2)/self.c5))*(-((((x[:, 1]+1)**2*(2*(1/(x[:, 1]+1))))*self.c5)/self.c5**2))))

        return grad

################################################################################
################################################################################
################################################################################

class RotatedEllipse01(Function):
    r"""RotatedEllipse01 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_R.html#go_benchmark.RotatedEllipse01]

        .. math::
            f(x)=c_1x_1^2 - c_2 x_1x_2 + c_3x_2^2


        Default constant values are :math:`c = (7, 10.392304845413264, 13)`.

        """
    def __init__(self, c1=7, c2=10.392304845413264, c3=13, name="RotatedEllipse01"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-500, 500], [-500, 500]]))

    def __call__(self, x):
        return (((self.c1*x[:, 0]**2)-((self.c2*x[:, 0])*x[:, 1]))+(self.c3*x[:, 1]**2)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (self.c1*(x[:, 0]**2*(2*(1/x[:, 0]))))-(self.c2*x[:, 1])
        grad[:, 0, 1] = (-(self.c2*x[:, 0]))+(self.c3*(x[:, 1]**2*(2*(1/x[:, 1]))))

        return grad

################################################################################
################################################################################
################################################################################

class RotatedEllipse02(Function):
    r"""RotatedEllipse02 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_R.html#go_benchmark.RotatedEllipse02]

        .. math::
            f(x)=x_1^2 - x_1x_2 + x_2^2

        """
    def __init__(self, name="RotatedEllipse02"):
        super().__init__(name=name)


        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-500, 500], [-500, 500]]))

    def __call__(self, x):
        return ((x[:, 0]**2-(x[:, 0]*x[:, 1]))+x[:, 1]**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (x[:, 0]**2*(2*(1/x[:, 0])))-x[:, 1]
        grad[:, 0, 1] = (-x[:, 0])+(x[:, 1]**2*(2*(1/x[:, 1])))

        return grad

################################################################################
################################################################################
################################################################################

class Schaffer01(Function):
    r"""Schaffer01 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Schaffer01]

        .. math::
            f(x)=c_1 + \frac{\sin^2 (x_1^2 + x_2^2)^2 - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}


        Default constant values are :math:`c = (0.5, 0.5, 1, 0.001)`.

        """
    def __init__(self, c1=0.5, c2=0.5, c3=1, c4=0.001, name="Schaffer01"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-100, 100], [-100, 100]]))

    def __call__(self, x):
        return (self.c1+((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2-self.c2)/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2*(2*((1/np.sin(x[:, 0]**2+x[:, 1]**2)**2)*(np.sin(x[:, 0]**2+x[:, 1]**2)**2*(2*((1/np.sin(x[:, 0]**2+x[:, 1]**2))*(np.cos(x[:, 0]**2+x[:, 1]**2)*(x[:, 0]**2*(2*(1/x[:, 0]))))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 0]**2*(2*(1/x[:, 0])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2
        grad[:, 0, 1] = (((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2*(2*((1/np.sin(x[:, 0]**2+x[:, 1]**2)**2)*(np.sin(x[:, 0]**2+x[:, 1]**2)**2*(2*((1/np.sin(x[:, 0]**2+x[:, 1]**2))*(np.cos(x[:, 0]**2+x[:, 1]**2)*(x[:, 1]**2*(2*(1/x[:, 1]))))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 1]**2*(2*(1/x[:, 1])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2

        return grad

################################################################################
################################################################################
################################################################################

class Schaffer02(Function):
    r"""Schaffer02 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Schaffer02]

        .. math::
            f(x)=c_1 + \frac{\sin^2 (x_1^2 - x_2^2)^2 - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}


        Default constant values are :math:`c = (0.5, 0.5, 1, 0.001)`.

        """
    def __init__(self, c1=0.5, c2=0.5, c3=1, c4=0.001, name="Schaffer02"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-100, 100], [-100, 100]]))

    def __call__(self, x):
        return (self.c1+((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2-self.c2)/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2*(2*((1/np.sin(x[:, 0]**2-x[:, 1]**2)**2)*(np.sin(x[:, 0]**2-x[:, 1]**2)**2*(2*((1/np.sin(x[:, 0]**2-x[:, 1]**2))*(np.cos(x[:, 0]**2-x[:, 1]**2)*(x[:, 0]**2*(2*(1/x[:, 0]))))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 0]**2*(2*(1/x[:, 0])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2
        grad[:, 0, 1] = (((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2*(2*((1/np.sin(x[:, 0]**2-x[:, 1]**2)**2)*(np.sin(x[:, 0]**2-x[:, 1]**2)**2*(2*((1/np.sin(x[:, 0]**2-x[:, 1]**2))*(np.cos(x[:, 0]**2-x[:, 1]**2)*(-(x[:, 1]**2*(2*(1/x[:, 1])))))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 1]**2*(2*(1/x[:, 1])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2

        return grad

################################################################################
################################################################################
################################################################################

class Schaffer04(Function):
    r"""Schaffer04 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Schaffer04]

        .. math::
            f(x)=c_1 + \frac{\cos^2 \left( \sin(x_1^2 - x_2^2) \right ) - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}


        Default constant values are :math:`c = (0.5, 0.5, 1, 0.001)`.

        """
    def __init__(self, c1=0.5, c2=0.5, c3=1, c4=0.001, name="Schaffer04"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-100, 100], [-100, 100]]))

    def __call__(self, x):
        return (self.c1+((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2-self.c2)/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2*(2*((1/np.cos(np.sin(x[:, 0]**2-x[:, 1]**2)))*((-np.sin(np.sin(x[:, 0]**2-x[:, 1]**2)))*(np.cos(x[:, 0]**2-x[:, 1]**2)*(x[:, 0]**2*(2*(1/x[:, 0]))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 0]**2*(2*(1/x[:, 0])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2
        grad[:, 0, 1] = (((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2*(2*((1/np.cos(np.sin(x[:, 0]**2-x[:, 1]**2)))*((-np.sin(np.sin(x[:, 0]**2-x[:, 1]**2)))*(np.cos(x[:, 0]**2-x[:, 1]**2)*(-(x[:, 1]**2*(2*(1/x[:, 1])))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 1]**2*(2*(1/x[:, 1])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2

        return grad



################################################################################
################################################################################
################################################################################

class Schwefel36(Function):
    r"""Schwefel36 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Schwefel36]

        .. math::
            f(x)=-x_1x_2(c_1 - c_2x_1 - c_3x_2)


        Default constant values are :math:`c = (72, 2, 2)`.

        """
    def __init__(self, c1=72, c2=2, c3=2, name="Schwefel36"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[0, 500], [0, 500]]))

    def __call__(self, x):
        return (((-x[:, 0])*x[:, 1])*((self.c1-(self.c2*x[:, 0]))-(self.c3*x[:, 1]))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((-x[:, 1])*((self.c1-(self.c2*x[:, 0]))-(self.c3*x[:, 1])))+(((-x[:, 0])*x[:, 1])*(-self.c2))
        grad[:, 0, 1] = ((-x[:, 0])*((self.c1-(self.c2*x[:, 0]))-(self.c3*x[:, 1])))+(((-x[:, 0])*x[:, 1])*(-self.c3))

        return grad

################################################################################
################################################################################
################################################################################

class SixHumpCamel(Function):
    r"""SixHumpCamel function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.SixHumpCamel]

        .. math::
            f(x)=c_1x_1^2+x_1x_2-c_2x_2^2-c_3x_1^4+c_4x_2^4+c_5x_1^6


        Default constant values are :math:`c = (4, 4, 2.1, 4, 1./3.)`.

        """
    def __init__(self, c1=4, c2=4, c3=2.1, c4=4, c5=1./3., name="SixHumpCamel"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

    def __call__(self, x):
        return ((((((self.c1*x[:, 0]**2)+(x[:, 0]*x[:, 1]))-(self.c2*x[:, 1]**2))-(self.c3*x[:, 0]**4))+(self.c4*x[:, 1]**4))+(self.c5*x[:, 0]**6)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((self.c1*(x[:, 0]**2*(2*(1/x[:, 0]))))+x[:, 1])-(self.c3*(x[:, 0]**4*(4*(1/x[:, 0])))))+(self.c5*(x[:, 0]**6*(6*(1/x[:, 0]))))
        grad[:, 0, 1] = (x[:, 0]-(self.c2*(x[:, 1]**2*(2*(1/x[:, 1])))))+(self.c4*(x[:, 1]**4*(4*(1/x[:, 1]))))

        return grad

################################################################################
################################################################################
################################################################################

class ThreeHumpCamel(Function):
    r"""ThreeHumpCamel function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_T.html#go_benchmark.ThreeHumpCamel]

        .. math::
            f(x)=c_1x_1^2 - c_2x_1^4 + \frac{x_1^6}{c_3} + x_1x_2 + x_2^2


        Default constant values are :math:`c = (2, 1.05, 6)`.

        """
    def __init__(self, c1=2, c2=1.05, c3=6, name="ThreeHumpCamel"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

    def __call__(self, x):
        return (((((self.c1*x[:, 0]**2)-(self.c2*x[:, 0]**4))+(x[:, 0]**6/self.c3))+(x[:, 0]*x[:, 1]))+x[:, 1]**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((self.c1*(x[:, 0]**2*(2*(1/x[:, 0]))))-(self.c2*(x[:, 0]**4*(4*(1/x[:, 0])))))+(((x[:, 0]**6*(6*(1/x[:, 0])))*self.c3)/self.c3**2))+x[:, 1]
        grad[:, 0, 1] = x[:, 0]+(x[:, 1]**2*(2*(1/x[:, 1])))

        return grad

################################################################################
################################################################################
################################################################################

class Treccani(Function):
    r"""Treccani function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_T.html#go_benchmark.Treccani]

        .. math::
            f(x)=x_1^4 + c_1x_1^3 + c_2x_1^2 + x_2^2


        Default constant values are :math:`c = (4, 4)`.

        """
    def __init__(self, c1=4, c2=4, name="Treccani"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

    def __call__(self, x):
        return (((x[:, 0]**4+(self.c1*x[:, 0]**3))+(self.c2*x[:, 0]**2))+x[:, 1]**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((x[:, 0]**4*(4*(1/x[:, 0])))+(self.c1*(x[:, 0]**3*(3*(1/x[:, 0])))))+(self.c2*(x[:, 0]**2*(2*(1/x[:, 0]))))
        grad[:, 0, 1] = x[:, 1]**2*(2*(1/x[:, 1]))

        return grad

################################################################################
################################################################################
################################################################################

class Trefethen(Function):
    r"""Trefethen function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_T.html#go_benchmark.Trefethen]

        .. math::
            f(x)=0.25 x_{1}^{2} + 0.25 x_{2}^{2} + e^{\sin\left(50 x_{1}\right)} - \sin\left(10 x_{1} + 10 x_{2}\right) + \sin\left(60 e^{x_{2}}\right) + \sin\left[70 \sin\left(x_{1}\right)\right] + \sin\left[\sin\left(80 x_{2}\right)\right]

        """
    def __init__(self, name="Trefethen"):
        super().__init__(name=name)


        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return ((((np.exp(np.sin(50*x[:, 0]))-np.sin((10*x[:, 0])+(10*x[:, 1])))+np.sin(60*np.exp(x[:, 1])))+np.sin(70*np.sin(x[:, 0])))+np.sin(np.sin(80*x[:, 1]))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((np.exp(np.sin(50*x[:, 0]))*(np.cos(50*x[:, 0])*50))-(np.cos((10*x[:, 0])+(10*x[:, 1]))*10))+(np.cos(70*np.sin(x[:, 0]))*(70*np.cos(x[:, 0])))
        grad[:, 0, 1] = ((-(np.cos((10*x[:, 0])+(10*x[:, 1]))*10))+(np.cos(60*np.exp(x[:, 1]))*(60*np.exp(x[:, 1]))))+(np.cos(np.sin(80*x[:, 1]))*(np.cos(80*x[:, 1])*80))

        return grad

################################################################################
################################################################################
################################################################################

class Ursem01(Function):
    r"""Ursem01 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_U.html#go_benchmark.Ursem01]

        .. math::
            f(x)=- \sin(c_1x_1 - c_2 \pi) - c_3 \cos(x_2) - c_4x_1


        Default constant values are :math:`c = (2, 0.5, 3, 0.5)`.

        """
    def __init__(self, c1=2, c2=0.5, c3=3, c4=0.5, name="Ursem01"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-2.5, 3], [-2, 2]]))

    def __call__(self, x):
        return (((-np.sin((self.c1*x[:, 0])-(self.c2*np.pi)))-(self.c3*np.cos(x[:, 1])))-(self.c4*x[:, 0])).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (-(np.cos((self.c1*x[:, 0])-(self.c2*np.pi))*self.c1))-self.c4
        grad[:, 0, 1] = -(self.c3*(-np.sin(x[:, 1])))

        return grad

################################################################################
################################################################################
################################################################################

class Ursem03(Function):
    r"""Ursem03 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_U.html#go_benchmark.Ursem03]

        .. math::
            f(x)=- \sin(c_1 \pi x_1 + c_2 \pi) \frac{c_3 - |x_1|}{c_4} \frac{c_5 - |x_1|}{c_6} - \sin(c_7 \pi x_2 + c_8 \pi) \frac{c_9 - |x_2|}{c_{10}} \frac{c_{11} - |x_2|}{c_{12}}


        Default constant values are :math:`c = (2.2, 0.5, 2, 2, 3, 2, 2.2, 0.5, 2, 2, 3, 2)`.

        """
    def __init__(self, c1=2.2, c2=0.5, c3=2, c4=2, c5=3, c6=2, c7=2.2, c8=0.5, c9=2, c10=2, c11=3, c12=2, name="Ursem03"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10, self.c11, self.c12 = c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-2, 2], [-1.5, 1.5]]))

    def __call__(self, x):
        return ((((-np.sin(((self.c1*np.pi)*x[:, 0])+(self.c2*np.pi)))*((self.c3-np.abs(x[:, 0]))/self.c4))*((self.c5-np.abs(x[:, 0]))/self.c6))-((np.sin(((self.c7*np.pi)*x[:, 1])+(self.c8*np.pi))*((self.c9-np.abs(x[:, 1]))/self.c10))*((self.c11-np.abs(x[:, 1]))/self.c12))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((((-(np.cos(((self.c1*np.pi)*x[:, 0])+(self.c2*np.pi))*(self.c1*np.pi)))*((self.c3-np.abs(x[:, 0]))/self.c4))+((-np.sin(((self.c1*np.pi)*x[:, 0])+(self.c2*np.pi)))*(((-np.sign(x[:, 0]))*self.c4)/self.c4**2)))*((self.c5-np.abs(x[:, 0]))/self.c6))+(((-np.sin(((self.c1*np.pi)*x[:, 0])+(self.c2*np.pi)))*((self.c3-np.abs(x[:, 0]))/self.c4))*(((-np.sign(x[:, 0]))*self.c6)/self.c6**2))
        grad[:, 0, 1] = -(((((np.cos(((self.c7*np.pi)*x[:, 1])+(self.c8*np.pi))*(self.c7*np.pi))*((self.c9-np.abs(x[:, 1]))/self.c10))+(np.sin(((self.c7*np.pi)*x[:, 1])+(self.c8*np.pi))*(((-np.sign(x[:, 1]))*self.c10)/self.c10**2)))*((self.c11-np.abs(x[:, 1]))/self.c12))+((np.sin(((self.c7*np.pi)*x[:, 1])+(self.c8*np.pi))*((self.c9-np.abs(x[:, 1]))/self.c10))*(((-np.sign(x[:, 1]))*self.c12)/self.c12**2)))

        return grad


################################################################################
################################################################################
################################################################################

class Ursem04(Function):
    r"""
    Ursem04 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_U.html#go_benchmark.Ursem04]

    A multimodal minimization function

    .. math::
        f(x)=-c_1 \sin(c_2 \pi x_1 + c_3 \pi) \frac{c_4 - \sqrt{x_1^2 + x_2^2}}{c_5}

    Default constant values are :math:`c = (3.0, 0.5, 0.5, 2.0, 4.0)`.
    """
    def __init__(self, c1=3, c2=0.5, c3=0.5, c4=2, c5=4, name="Ursem04"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-2, 2], [-2, 2]]))

    def __call__(self, x):
        return (-self.c1*np.sin(self.c2*np.pi*x[:, 0]+self.c3*np.pi)*((self.c4-np.sqrt(x[:, 0]**2+x[:, 1]**2))/self.c5)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = -self.c1*np.cos(self.c2*np.pi*x[:, 0]+self.c3*np.pi)*self.c2*np.pi*(self.c4-np.sqrt(x[:, 0]**2+x[:, 1]**2))/self.c5 + self.c1*np.sin(self.c2*np.pi*x[:, 0]+self.c3*np.pi)*x[:,0] / (self.c5*np.sqrt(x[:, 0]**2+x[:, 1]**2))
        grad[:, 0, 1] = self.c1*np.sin(self.c2*np.pi*x[:, 0]+self.c3*np.pi)*x[:,1] / (self.c5*np.sqrt(x[:, 0]**2+x[:, 1]**2))

        return grad


################################################################################
################################################################################
################################################################################

class UrsemWaves(Function):
    r"""UrsemWaves function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_U.html#go_benchmark.UrsemWaves]

        .. math::
            f(x)=-c_1x_1^2 + (x_2^2 - c_2x_2^2)x_1x_2 + c_3 \cos \left[ c_4x_1 - x_2^2(c_5 + x_1) \right ] \sin(c_6 \pi x_1)


        Default constant values are :math:`c = (0.9, 4.5, 4.7, 2, 2, 2.5)`.

        """
    def __init__(self, c1=0.9, c2=4.5, c3=4.7, c4=2, c5=2, c6=2.5, name="UrsemWaves"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-0.9, 1.2], [-1.2, 1.2]]))

    def __call__(self, x):
        return ((((-self.c1)*x[:, 0]**2)+(((x[:, 1]**2-(self.c2*x[:, 1]**2))*x[:, 0])*x[:, 1]))+((self.c3*np.cos((self.c4*x[:, 0])-(x[:, 1]**2*(self.c5+x[:, 0]))))*np.sin((self.c6*np.pi)*x[:, 0]))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((-self.c1)*(x[:, 0]**2*(2*(1/x[:, 0]))))+((x[:, 1]**2-(self.c2*x[:, 1]**2))*x[:, 1]))+(((self.c3*((-np.sin((self.c4*x[:, 0])-(x[:, 1]**2*(self.c5+x[:, 0]))))*(self.c4-x[:, 1]**2)))*np.sin((self.c6*np.pi)*x[:, 0]))+((self.c3*np.cos((self.c4*x[:, 0])-(x[:, 1]**2*(self.c5+x[:, 0]))))*(np.cos((self.c6*np.pi)*x[:, 0])*(self.c6*np.pi))))
        grad[:, 0, 1] = (((((x[:, 1]**2*(2*(1/x[:, 1])))-(self.c2*(x[:, 1]**2*(2*(1/x[:, 1])))))*x[:, 0])*x[:, 1])+((x[:, 1]**2-(self.c2*x[:, 1]**2))*x[:, 0]))+((self.c3*((-np.sin((self.c4*x[:, 0])-(x[:, 1]**2*(self.c5+x[:, 0]))))*(-((x[:, 1]**2*(2*(1/x[:, 1])))*(self.c5+x[:, 0])))))*np.sin((self.c6*np.pi)*x[:, 0]))

        return grad

################################################################################
################################################################################
################################################################################

class VenterSobiezcczanskiSobieski(Function):
    r"""VenterSobiezcczanskiSobieski function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_V.html#go_benchmark.VenterSobiezcczanskiSobieski]

        .. math::
            f(x)=x_1^2 - c_1 \cos^2(x_1) - c_2 \cos(x_1^2/c_3) + x_2^2 - c_4 \cos^2(x_2) - c_5 \cos(x_2^2/c_6)


        Default constant values are :math:`c = (100, 100, 30, 100, 100, 30)`.

        """
    def __init__(self, c1=100, c2=100, c3=30, c4=100, c5=100, c6=30, name="VenterSobiezcczanskiSobieski"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-50, 50], [-50, 50]]))

    def __call__(self, x):
        return (((((x[:, 0]**2-(self.c1*np.cos(x[:, 0])**2))-(self.c2*np.cos(x[:, 0]**2/self.c3)))+x[:, 1]**2)-(self.c4*np.cos(x[:, 1])**2))-(self.c5*np.cos(x[:, 1]**2/self.c6))).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((x[:, 0]**2*(2*(1/x[:, 0])))-(self.c1*(np.cos(x[:, 0])**2*(2*((1/np.cos(x[:, 0]))*(-np.sin(x[:, 0])))))))-(self.c2*((-np.sin(x[:, 0]**2/self.c3))*(((x[:, 0]**2*(2*(1/x[:, 0])))*self.c3)/self.c3**2)))
        grad[:, 0, 1] = ((x[:, 1]**2*(2*(1/x[:, 1])))-(self.c4*(np.cos(x[:, 1])**2*(2*((1/np.cos(x[:, 1]))*(-np.sin(x[:, 1])))))))-(self.c5*((-np.sin(x[:, 1]**2/self.c6))*(((x[:, 1]**2*(2*(1/x[:, 1])))*self.c6)/self.c6**2)))

        return grad

################################################################################
################################################################################
################################################################################

class WayburnSeader01(Function):
    r"""WayburnSeader01 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.WayburnSeader01]

        .. math::
            f(x)=(x_1^6 + x_2^4 - c_1)^2 + (c_2x_1 + x_2 - c_3)^2


        Default constant values are :math:`c = (17, 2, 4)`.

        """
    def __init__(self, c1=17, c2=2, c3=4, name="WayburnSeader01"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

    def __call__(self, x):
        return (((x[:, 0]**6+x[:, 1]**4)-self.c1)**2+(((self.c2*x[:, 0])+x[:, 1])-self.c3)**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = (((x[:, 0]**6+x[:, 1]**4)-self.c1)**2*(2*((1/((x[:, 0]**6+x[:, 1]**4)-self.c1))*(x[:, 0]**6*(6*(1/x[:, 0]))))))+((((self.c2*x[:, 0])+x[:, 1])-self.c3)**2*(2*((1/(((self.c2*x[:, 0])+x[:, 1])-self.c3))*self.c2)))
        grad[:, 0, 1] = (((x[:, 0]**6+x[:, 1]**4)-self.c1)**2*(2*((1/((x[:, 0]**6+x[:, 1]**4)-self.c1))*(x[:, 1]**4*(4*(1/x[:, 1]))))))+((((self.c2*x[:, 0])+x[:, 1])-self.c3)**2*(2*(1/(((self.c2*x[:, 0])+x[:, 1])-self.c3))))

        return grad

################################################################################
################################################################################
################################################################################

class WayburnSeader02(Function):
    r"""WayburnSeader02 function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.WayburnSeader02]

        .. math::
            f(x)=\left[ c_1 - c_2(x_1 - c_3)^2 - c_4(x_2 - c_5)^2 \right]^2 + (x_2 - c_6)^2


        Default constant values are :math:`c = (1.613, 4, 0.3125, 4, 1.625, 1)`.

        """
    def __init__(self, c1=1.613, c2=4, c3=0.3125, c4=4, c5=1.625, c6=1, name="WayburnSeader02"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-500, 500], [-500, 500]]))

    def __call__(self, x):
        return (((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2))**2+(x[:, 1]-self.c6)**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2))**2*(2*((1/((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2)))*(-(self.c2*((x[:, 0]-self.c3)**2*(2*(1/(x[:, 0]-self.c3))))))))
        grad[:, 0, 1] = (((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2))**2*(2*((1/((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2)))*(-(self.c4*((x[:, 1]-self.c5)**2*(2*(1/(x[:, 1]-self.c5)))))))))+((x[:, 1]-self.c6)**2*(2*(1/(x[:, 1]-self.c6))))

        return grad



################################################################################
################################################################################
################################################################################

class Zettl(Function):
    r"""Zettl function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_Z.html#go_benchmark.Zettl]

        .. math::
            f(x)=c_1 x_{1} + \left(x_{1}^{2} - c_2 x_{1} + x_{2}^{2}\right)^{2}


        Default constant values are :math:`c = (0.25, 2)`.

        """
    def __init__(self, c1=0.25, c2=2, name="Zettl"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-1, 5], [-1, 5]]))

    def __call__(self, x):
        return ((self.c1*x[:, 0])+((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2)**2).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = self.c1+(((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2)**2*(2*((1/((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2))*((x[:, 0]**2*(2*(1/x[:, 0])))-self.c2))))
        grad[:, 0, 1] = ((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2)**2*(2*((1/((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2))*(x[:, 1]**2*(2*(1/x[:, 1])))))

        return grad

################################################################################
################################################################################
################################################################################

class Zirilli(Function):
    r"""Zirilli function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_Z.html#go_benchmark.Zirilli]

        .. math::
            f(x)=c_1x_1^4 - c_2x_1^2 + c_3x_1 + c_4x_2^2


        Default constant values are :math:`c = (0.25, 0.5, 0.1, 0.5)`.

        """
    def __init__(self, c1=0.25, c2=0.5, c3=0.1, c4=0.5, name="Zirilli"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

    def __call__(self, x):
        return ((((self.c1*x[:, 0]**4)-(self.c2*x[:, 0]**2))+(self.c3*x[:, 0]))+(self.c4*x[:, 1]**2)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((self.c1*(x[:, 0]**4*(4*(1/x[:, 0]))))-(self.c2*(x[:, 0]**2*(2*(1/x[:, 0])))))+self.c3
        grad[:, 0, 1] = self.c4*(x[:, 1]**2*(2*(1/x[:, 1])))

        return grad

# https://www.sfu.ca/~ssurjano/optimization.html, many local minima section,
# excluding discontinuous functions

################################################################################
################################################################################
################################################################################

class DropWave(Function):
    r"""DropWave function

        .. math::
            f(x)=-\frac{c_1+\cos(c_2\sqrt{x_1^2+x_2^2})}{c_3(x_1^2+x_2^2)+c_4}


        Default constant values are :math:`c = (1., 12., 0.5, 2.)`.

        """
    def __init__(self, c1=1., c2=12., c3=0.5, c4=2., name="DropWave"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-5.12, 5.12]))

    def __call__(self, x):
        self._numerator = (self.c1 + np.cos(self.c2 * np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)))[:, np.newaxis]
        self._denominator = (self.c3 * (x[:, 0] ** 2 + x[:, 1] ** 2) + self.c4)[:, np.newaxis]
        return -self._numerator / self._denominator

    def grad(self, x):
        _ = self.__call__(x)
        x1, x2 = x[:, 0], x[:, 1]
        dist_sq = (x1 ** 2 + x2 ** 2)[:, np.newaxis]
        num_grad = -np.sin(self.c2 * np.sqrt(dist_sq)) * self.c2 * x / np.sqrt(dist_sq)
        denom_grad = 2 * self.c3 * x

        return (-(num_grad * self._denominator - self._numerator * denom_grad) / (self._denominator ** 2))[:, np.newaxis, :]
