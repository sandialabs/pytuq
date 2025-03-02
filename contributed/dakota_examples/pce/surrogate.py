#  _______________________________________________________________________
#
#  Dakota: Explore and predict with confidence.
#  Copyright 2014-2024
#  National Technology & Engineering Solutions of Sandia, LLC (NTESS).
#  This software is distributed under the GNU Lesser General Public License.
#  For more information, see the README file in the top Dakota directory.
#  _______________________________________________________________________

import numpy as np

import pytuq.utils.funcbank as fcb

from pytuq.surrogates.pce import PCE
from pytuq.utils.maps import scale01ToDom
from pytuq.lreg.anl import anl
from pytuq.lreg.lreg import lsq

########################################################
order = 4         # Polynomial order
dim = 3           # Dimensionality of input data
########################################################

#############################################################
#                  Pytuq Python Surrogate                   #
#############################################################

class Pytuq:

    def __init__(self, params=None):

        print("Pytuq surrogate class constructer...")
        self.pce = PCE(dim, order, 'LU')


    def construct(self, var, resp):

        # Try out the pytuq module
        self.pce.set_training_data(var,resp[:,0])
        print(self.pce.build())

        return


    def predict(self, pts):

        # Try out the pytuq module
        return self.pce.evaluate(pts)['Y_eval']


    def gradient(self, pts):

        # How to obtain?
        return None


def fcb_sin4(params):

    print("params:",params)

    num_fns = params["functions"]
    x       = params["cv"]
    ASV     = params["asv"]

    retval = {}

    if (ASV[0] & 1):  # fn vals
        retval["fns"] = fcb.sin4(np.array([x]))

    return retval


#############################################################
#                 Simple Python Surrogate                   #
#############################################################

class Polynomial:

    def __init__(self, params=None):

        print("python Polynomial Surrogate class constructer...")
        self.coeffs = None
        if params is not None:
            self.params = params


    def construct(self, var, resp):

        var2 = np.hstack((np.ones((var.shape[0], 1)), var))
        self.coeffs = np.zeros(var.shape[1])
        z = np.linalg.inv(np.dot(var2.T, var2))
        self.coeffs = np.dot(z, np.dot(var2.T, resp))
        return


    def predict(self, pts):
        return self.coeffs[0]+pts.dot(self.coeffs[1:])


    def gradient(self, pts):
        grad = self.coeffs[1:]
        return grad.T


# Dummy main
if __name__ == "__main__":
    print("Hello from surrogates test module.")
