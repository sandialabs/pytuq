
import sys
import torch
import numpy as np

from pytuq.surrogates.nn import NN
from quinn.solvers.nn_vi import NN_VI
from quinn.nns.rnet import RNet, Poly
from quinn.utils.plotting import myrc
from quinn.utils.maps import scale01ToDom
from quinn.func.funcs import Sine, Sine10, blundell


#############################################################
#                  Pytuq Python Surrogate                   #
#############################################################

class Pytuq:

    def __init__(self, params=None):

        torch.set_default_dtype(torch.double)
        myrc()

        device_id='cuda:0'
        device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        print("Using device",device)

        ndim = 1 # input dimensionality
        nout = 1  # Scalar valued output example

        # Initialize neural network with optional parameters for object instantiation
        net_options = {'wp_function': Poly(0),
                       'indim': ndim, 
                       'outdim': nout,
                       'layer_pre': True,
                       'layer_post': True,
                       'biasorno': True,
                       'nonlin': True,
                       'mlp': False, 
                       'final_layer': None,
                       'device': device,
                       }

        ## Pass in unpacked net_options dict to constructor through kwargs
        self.nnet = NN('RNet', 3, 3, **net_options)


    def construct(self, var, resp):

        self.nnet.set_training_data(var, resp)
        print(self.nnet.build(datanoise=0.02, lrate=0.01, batch_size=None, nsam=1, nepochs=500, verbose=False))
        return


    def predict(self, pts):

        return self.nnet.evaluate(pts)['Y_eval']


    def gradient(self, pts):

        # How to obtain?
        return None


def sine(params):

    #print("params:",params)

    num_fns = params["functions"]
    x       = params["cv"]
    ASV     = params["asv"]

    retval = {}

    xx = np.array([x])

    if (ASV[0] & 1):  # fn vals
        retval["fns"] = Sine(xx, datanoise=0.02)

    #print("retval:",retval)
    return retval


# Dummy main
if __name__ == "__main__":
    print("Hello from PyTUQ NNS TEST module.")
