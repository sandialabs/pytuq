#!/usr/bin/env python
"""
Benchmark functions module that are not 1d, 2d, or Nd.

Most of the functions are taken from https://github.com/Vahanosi4ek/pytuq_funcs that autogenerates the codes given function's latex strings.
"""
import sys
import numpy as np
from scipy.stats import multivariate_normal

from .func import Function


################################################################################
################################################################################
################################################################################


class Ishigami(Function):
    r"""Ishigami function

    Reference: [https://www.sfu.ca/~ssurjano/ishigami.html]

    .. math::
        f(x) = \sin(x_1) + a \sin^2(x_2) + b x_3^4 \sin(x_1)

    Default constant values are :math:`a = 7` and :math:`b = 0.1`

    """
    def __init__(self, name='Ishigami'):
        super().__init__()
        self.name = name
        self.dim = 3
        self.outdim = 1


        self.setDimDom(np.ones((self.dim, 1)) * np.array([-np.pi, np.pi]))
        self.a = 7
        self.b = 0.1


    def __call__(self, x):
        sam = x.shape[0]
        self.checkDim(x)

        ydata=np.empty((sam, self.outdim))

        for j in range(sam):
            ydata[j, 0]=np.sin(x[j,0])+self.a*np.sin(x[j,1])**2+self.b*np.sin(x[j,0])*x[j,2]**4

        return ydata

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        grad[:,0,0] = np.cos(x[:,0]) + self.b * np.cos(x[:,0]) * x[:,2]**4
        grad[:,0,1] = 2. * self.a * np.sin(x[:,1]) * np.cos(x[:,1])
        grad[:,0,2] = 4. * self.b * np.sin(x[:,0]) * x[:,2]**3

        return grad


################################################################################
################################################################################
################################################################################


class Wolfe(Function):
    r"""
    Wolfe function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.Wolfe]

    .. math::
        f(x)=c_1(x_1^2 + x_2^2 - x_1x_2)^{c_2} + x_3


    Default constant values are :math:`c = (4/3, 0.75)`

    """
    def __init__(self, c1=4./3., c2=0.75, name="Wolfe"):
        super().__init__(name=name)

        self.c1, self.c2 = c1, c2
        self.dim = 3
        self.outdim = 1

        self.setDimDom(domain=np.array([[0, 2], [0, 2], [0, 2]]))

    def __call__(self, x):
        return ((self.c1*((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1]))**self.c2)+x[:, 2]).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = self.c1*(((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1]))**self.c2*(self.c2*((1/((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1])))*((x[:, 0]**2*(2*(1/x[:, 0])))-x[:, 1]))))
        grad[:, 0, 1] = self.c1*(((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1]))**self.c2*(self.c2*((1/((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1])))*((x[:, 1]**2*(2*(1/x[:, 1])))-x[:, 0]))))
        grad[:, 0, 2] = 1

        return grad

################################################################################
################################################################################
################################################################################

class Colville(Function):
    r"""
    Colville function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Colville]

    .. math::
        f(x)=(x_1 - c_1)^{2} + c_2 (x_1^{2} - x_2)^{2} + c_3 (x_2 - c_4)^{2} + (x_3 - c_5)^{2} + c_6 (x_3^{2} - x_4)^{2} + c_7 (x_4 - c_8)^{2} + c_9 \frac{x_4 - c_{10}}{x_2}


    Default constant values are :math:`c = (1.0, 100.0, 10.1, 1.0, 1.0, 90.0, 10.1, 1.0, 19.8, 1.0)`

    """
    def __init__(self, c1=1.0, c2=100.0, c3=10.1, c4=1.0, c5=1.0, c6=90.0, c7=10.1, c8=1.0, c9=19.8, c10=1.0, name="Colville"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10 = c1, c2, c3, c4, c5, c6, c7, c8, c9, c10
        self.dim = 4
        self.outdim = 1

        self.setDimDom(domain=np.array([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]))

    def __call__(self, x):
        return ((x[:, 0]-self.c1)**(2.0)+self.c2*(x[:, 0]**(2.0)-x[:, 1])**(2.0)+self.c3*(x[:, 1]-self.c4)**(2.0)+(x[:, 2]-self.c5)**(2.0)+self.c6*(x[:, 2]**(2.0)-x[:, 3])**(2.0)+self.c7*(x[:, 3]-self.c8)**(2.0)+self.c9*(x[:, 3]-self.c10)/(x[:, 1])).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = 2.0*(x[:, 0]-self.c1)+self.c2*2.0*(x[:, 0]**2.0-x[:, 1])*2.0*x[:, 0]
        grad[:, 0, 1] = -2.0*self.c2*(x[:, 0]**2.0-x[:, 1])+2.0*self.c3*(x[:, 1]-self.c4)-(self.c9*(x[:, 3]-self.c10))/(x[:, 1]**2.0)
        grad[:, 0, 2] = 2.0*(x[:, 2]-self.c5)+2.0*self.c6*(x[:, 2]**2.0-x[:, 3])*2.0*x[:, 2]
        grad[:, 0, 3] = -2.0*self.c6*(x[:, 2]**2.0-x[:, 3])+self.c7*2.0*(x[:, 3]-self.c8)+(self.c9)/(x[:, 1])

        return grad



################################################################################
################################################################################
################################################################################


class MieleCantrell(Function):
    r"""
    MieleCantrell function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.MieleCantrell]

    A multimodal minimization function

    .. math::
        f(x)=(e^{-x_1} - x_2)^4 + c_1(x_2 - x_3)^6 + \tan^4(x_3 - x_4) + x_1^8


    Default constant values are :math:`c_1 = 100.0`
    """
    def __init__(self, c1=100.0, name="MieleCantrell"):
        super().__init__(name=name)

        self.c1 = c1
        self.dim = 4
        self.outdim = 1

        self.setDimDom(domain=np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]]))

    def __call__(self, x):
        return ((np.exp(-x[:, 0])-x[:, 1])**4+self.c1*(x[:, 1]-x[:, 2])**6+np.tan(x[:, 2]-x[:, 3])**4+x[:, 0]**8).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = - 4.0*(np.exp(-x[:, 0])-x[:, 1])**3 * np.exp(-x[:, 0]) + 8.0*x[:, 0]**7

        grad[:, 0, 1] = - 4.0*(np.exp(-x[:, 0])-x[:, 1])**3  + self.c1*6*(x[:, 1]-x[:, 2])**5

        grad[:, 0, 2] = -self.c1*6.0*(x[:, 1]-x[:, 2])**5 + 4.0 * np.sin(x[:, 2]-x[:, 3])**3/np.cos(x[:, 2]-x[:, 3])**5

        grad[:, 0, 3] = - 4.0 * np.sin(x[:, 2]-x[:, 3])**3/np.cos(x[:, 2]-x[:, 3])**5

        return grad

################################################################################
################################################################################
################################################################################

class Powell(Function):
    r"""
    Powell function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Powell]

    .. math::
        f(x)=(x_3+c_1x_1)^2+c_2(x_2-x_4)^2+(x_1-c_3x_2)^4+c_4(x_3-x_4)^4


    Default constant values are :math:`c = (10.0, 5.0, 2.0, 10.0)`

    """
    def __init__(self, c1=10.0, c2=5.0, c3=2.0, c4=10.0, name="Powell"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.dim = 4
        self.outdim = 1

        self.setDimDom(domain=np.array([[-4, 5], [-4, 5], [-4, 5], [-4, 5]]))

    def __call__(self, x):
        return ((((x[:, 2]+(self.c1*x[:, 0]))**2+(self.c2*(x[:, 1]-x[:, 3])**2))+(x[:, 0]-(self.c3*x[:, 1]))**4)+(self.c4*(x[:, 2]-x[:, 3])**4)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = ((x[:, 2]+(self.c1*x[:, 0]))**2*(2*((1/(x[:, 2]+(self.c1*x[:, 0])))*self.c1)))+((x[:, 0]-(self.c3*x[:, 1]))**4*(4*(1/(x[:, 0]-(self.c3*x[:, 1])))))
        grad[:, 0, 1] = (self.c2*((x[:, 1]-x[:, 3])**2*(2*(1/(x[:, 1]-x[:, 3])))))+((x[:, 0]-(self.c3*x[:, 1]))**4*(4*((1/(x[:, 0]-(self.c3*x[:, 1])))*(-self.c3))))
        grad[:, 0, 2] = ((x[:, 2]+(self.c1*x[:, 0]))**2*(2*(1/(x[:, 2]+(self.c1*x[:, 0])))))+(self.c4*((x[:, 2]-x[:, 3])**4*(4*(1/(x[:, 2]-x[:, 3])))))
        grad[:, 0, 3] = (self.c2*((x[:, 1]-x[:, 3])**2*(2*(-(1/(x[:, 1]-x[:, 3]))))))+(self.c4*((x[:, 2]-x[:, 3])**4*(4*(-(1/(x[:, 2]-x[:, 3]))))))

        return grad


################################################################################
################################################################################
################################################################################

class Dolan(Function):
    r"""
    Dolan function

    Reference: [https://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Dolan]

    .. math::
        f(x)=|(x_1 + c_1x_2)\sin(x_1) - c_2x_3 - c_3x_4\cos(x_5 + x_5 - x_1) + c_4x_5^2 - x_2 - c_5|


    Default constant values are :math:`c = (1.7, 1.5, 0.1, 0.2, 1)`

    """
    def __init__(self, c1=1.7, c2=1.5, c3=0.1, c4=0.2, c5=1, name="Dolan"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.dim = 5
        self.outdim = 1

        self.setDimDom(domain=np.array([[-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100]]))

    def __call__(self, x):
        return (np.abs(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*((np.sin(x[:, 0])+((x[:, 0]+(self.c1*x[:, 1]))*np.cos(x[:, 0])))-((self.c3*x[:, 3])*(-(-np.sin((x[:, 4]+x[:, 4])-x[:, 0])))))
        grad[:, 0, 1] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*((self.c1*np.sin(x[:, 0]))-1)
        grad[:, 0, 2] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*(-self.c2)
        grad[:, 0, 3] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*(-(self.c3*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))
        grad[:, 0, 4] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*((-((self.c3*x[:, 3])*((-np.sin((x[:, 4]+x[:, 4])-x[:, 0]))*(1+1))))+(self.c4*(x[:, 4]**2*(2*(1/x[:, 4])))))

        return grad


################################################################################
################################################################################
################################################################################

class Friedman(Function):
    r"""
    Friedman function

    Reference: [https://www.sfu.ca/~ssurjano/fried.html]

    A 5d trigonometric function, linear in :math:`x_4` and :math:`x_5`


    Default constant values are :math:`c = (10., 20., 0.5, 2., 10., 5.)`

    .. math::
        f(x)=c_1\sin(\pi x_1x_2)+c_2(x_3-c_3)^{c_4}+c_5x_4+c_6x_5

    """
    def __init__(self, c1=10., c2=20., c3=-0.5, c4=2., c5=10., c6=5., name="Friedman"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
        self.dim = 5
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):
        x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        return (self.c1 * np.sin(np.pi * x1 * x2) + self.c2 * (x3 - self.c3) ** self.c4 + self.c5 * x4 + self.c6 * x5)[:, np.newaxis]

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        grad[:, 0, 0] = self.c1 * np.pi * x[:, 1] * np.cos(np.pi * x[:, 0] * x[:, 1])
        grad[:, 0, 1] = self.c1 * np.pi * x[:, 0] * np.cos(np.pi * x[:, 0] * x[:, 1])
        grad[:, 0, 2] = self.c2 * self.c4 * (x[:, 2] - self.c3) ** (self.c4 - 1)
        grad[:, 0, 3] = self.c5
        grad[:, 0, 4] = self.c6

        return grad



################################################################################
################################################################################
################################################################################

class GramacyLee(Function):
    r"""
    Gramacy and Lee function

    Reference: [https://www.sfu.ca/~ssurjano/grlee09.html]

    A 6d function, where :math:`x_5` and :math:`x_6` aren't active

    .. math::
        f(x)=e^{sin((c_1(x_1+c_2))^{c_3})}+x_2x_3+x_4

    Default constant values are :math:`c = (0.9, 0.48, 10.)`.

    """
    def __init__(self, c1=0.9, c2=0.48, c3=10., name="GramacyLee"):
        super().__init__(name=name)

        self.c1, self.c2, self.c3 = c1, c2, c3
        self.dim = 6
        self.outdim = 1

        self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

    def __call__(self, x):
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return (np.exp(np.sin((self.c1 * (x1 + self.c2)) ** self.c3)) + x2 * x3 + x4)[:, np.newaxis]

    def grad(self, x):
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        grad = np.zeros((x.shape[0], self.outdim, self.dim))

        grad[:, 0, 0] = (np.exp(np.sin((self.c1 * (x1 + self.c2)) ** self.c3))) \
                        * np.cos((self.c1 * (x1 + self.c2)) ** self.c3) \
                        * self.c3 * (self.c1 * (x1 + self.c2)) ** (self.c3 - 1) \
                        * self.c1
        grad[:, 0, 1] = x3
        grad[:, 0, 2] = x2
        grad[:, 0, 3] = 1.
        grad[:, 0, 4:] = 0.

        return grad
