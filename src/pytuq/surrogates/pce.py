#!/usr/bin/env python

"""This module provides a KLPC (Karhunen-Loeve and Polynomial Chaos) wrapper class to facilitate 
    the universal coupling of FASTMath UQ tools and libraries. This class focuses on the use case 
    of surrogate models built with PCE and linear regression, keeping in mind 
    flexibility to implement additional UQ functionalities in the future.

    The PCE class supports a minimal API, with methods to construct the model, build it with training data,
    evaluate it with input data, and offer predictions with covariance, variance, and standard deviation.
    It is capable of handling multidimensional inputs and is optimized for scalar-valued function outputs.

    Note:
        The current implementation focuses on providing a general foundation for polynomial chaos 
        expansions with simple least squares regression or advanced analytical regression. While not all 
        construct and build options are currently supported, the class was developed with future growth in mind.
"""

import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.lreg.lreg import lsq
from pytuq.lreg.anl import anl
from pytuq.lreg.bcs import bcs


class PCE:
    r"""A wrapper class to access KLPC functionalities for PCE surrogate models. 

    Attributes:
        pcrv (PCRV object): Polynomial Chaos random variable object, encapsulates details for polynomial chaos expansion.
        sdim (int): Stochastic dimensionality, i.e. # of stochastic inputs.
        order (int): Order of polynomial chaos expansion.
        pctype (list[str]): Type of PC polynomial used.
        outdim (int): Physical dimensionality, i.e. # of output variables.
        lreg (lreg object): Linear regression object used for fitting the model.
        mindex (int np.ndarray): Multiindex array carrying the powers to which the basis functions will be raised to within the PC terms.
        regression_method (str): Method used for linear regression. ex] anl, opt, lsq
        _x_train (np.ndarray): Input training data
        _y_train (np.ndarray): Output training data, corresponding to x_train
    """

    def __init__(self, pce_dim, pce_order, pce_type, verbose=0, **kwargs):
        r"""Initializes a Polynomial Chaos Random Variable (PCRV) with, at minimum:
        stochastic dimensionality, order, and polynomial chaos (PC) type. 

        Args:
            pce_dim (int): Stochastic dimensionality :math:`d` of the PC random variable/vector. 
                The number of sources of uncertainty (sdim).
            pce_order (int): Order of the PC expansion.
            pce_type (str or list): PC type. Either a list of :math:`s` strings (one per stochastic dimension), 
                or a single string for all dimensions. Supported types include 'LU' (Legendre) and 'HG' (Hermite-Gaussian).
            verbose (int): Output verbosity. Higher values print out more information. Default of 0
            pce_outdim (int, optional): Physical dimensionality :math:`s` of the PC random variable/vector.
                Default of 1 indicates a scalar-valued output.
            mi (list or np.ndarray, optional): List of :math:`d` multiindex arrays, each of size :math:`(K_i,s)` for :math:`i=1, \dots, d`. 
                Or a single multiindex array of size :math:`(K,s)`, meaning all dimensions get the same multiindex. 
                Defaults to None, which is a single 1d constant random variable i.e. a multiindex of all zeros.
            cfs (list or np.ndarray, optional): List of :math:`d` coefficient arrays, each of size :math:`K_i` for :math:`i=1, \dots, d`. 
                Or a single coefficient array of size :math:`K`, meaning all dimensions get the same coefficient array. 
                Or a 2d array of size :math:`(K,d)`. Defaults to None, which is populating coefficients with all zeros.
        
        Note:
            Future Implementations:
            - Support for automatically generated random coefficients (`setRandomCfs` method).
            - Initialization with different PCRV operations/related classes (`PCRV_mvn` class).
        """
        self.sdim = pce_dim
        self.order = pce_order
        self.pctype = pce_type   # Choose from options: 'LU', 'HG', or mix of ['HG', 'LU']
        self.outdim = kwargs.get('pce_outdim', 1) # Scalar valued output
        self.verbose = verbose

        self.lreg = None
        self._x_train = None
        self._y_train = None
        self.regression_method = None

        # Get the original multiindex, before possible modification in build
        self.mindex = kwargs.get('mi', get_mi(self.order, self.sdim))

        self.pcrv = PCRV(self.outdim, self.sdim, self.pctype, 
                    mi = self.mindex, 
                    cfs = kwargs.get('cfs'))

        if self.verbose > 0:
            print("Constructed PC Surrogate with the following attributes:")
            print(self.pcrv, end='\n\n')
            # self.pcrv.printInfo()

    def set_training_data(self, x_train, y_train):
        r"""Sets the training data with validation.

        Args:
            x_train (np.ndarray): 2d array of size `(N, d)` representing training input data,
                where `N` is the number of samples and `d` is the dimensionality (sdim) of each sample.
            y_train (np.ndarray): 1d array of size `N` representing the training output data,
                where each element corresponds to the output value for each input sample in `x_train`.

        Raises:
            ValueError: If x_train or y_train do not meet the required dimensions.
        """
        if not (isinstance(x_train, np.ndarray) and x_train.ndim == 2):
            raise ValueError("x_train must be a 2D numpy array.")
        if not (isinstance(y_train, np.ndarray) and y_train.ndim == 1):
            raise ValueError("y_train must be a 1D numpy array.")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in x and y must be the same.")

        self._x_train = x_train
        self._y_train = y_train

    def build(self, **kwargs): 
        """Builds and initializes the linear regression model for the pcrv object with training data.
        Returns coefficients for evaluated Polynomial Chaos Expansion.

        Args:
            **kwargs (dict): Optional keyword arguments for configuring the regression model, including:

                - regression (str): Type of regression to be used. 
                    Options are:
                        + 'anl': Advanced regression analysis.
                        + 'lsq': Least squares regression.

                    The default is 'lsq'. Future regression methods to implement include 'opt' and 'lreg_merr'.

                - method (str): Method to be used when 'regression' is set to 'anl'. 
                    Options are:
                        + 'vi': Variational inference.
                        + 'full': Full analytical solution, the default for anl().
                        
                    The default is 'vi' when 'regression' is 'anl'.
                    
                - datavar (float): Available for regression type 'anl'.
                - cov_nugget (float): Available for regression type 'anl'.
                - prior_var (float): Available for regression type 'anl', method 'full'.
        
        Returns:
            np.ndarray: PC coefficients  
        """
        if self._x_train is None or self._y_train is None:
            raise RuntimeError("Training data must be set using set_training_data() before calling build().")

        # Reset the multiindex and coefficients
        self.pcrv.setMiCfs(self.mindex, cfs=None)

        regression = kwargs.get('regression', 'lsq')

        if regression == 'lsq':
            self.lreg = lsq()
        elif regression == 'anl':
            method = kwargs.get('method', None)
            if method == 'vi':
                self.lreg = anl(method='vi', datavar=kwargs.get('datavar'), cov_nugget=kwargs.get('cov_nugget', 0.0))
            else:
                self.lreg = anl(datavar=kwargs.get('datavar'), prior_var=kwargs.get('prior_var'), cov_nugget=kwargs.get('cov_nugget', 0.0))
        elif regression == 'bcs':
            self.lreg = bcs(eta=kwargs.get('eta', 1.e-8), datavar_init=kwargs.get('datavar_init'))
        else:
            raise ValueError(f"Regression method '{regression}' is not recognized and/or supported yet.")

        self.regression_method = type(self.lreg).__name__

        if self.verbose > 0:
            print("Regression method:", self.regression_method)
                
        def basisevaluator(x, pars):
            self.pcrv, = pars
            return self.pcrv.evalBases(x, 0)

        self.lreg.setBasisEvaluator(basisevaluator, (self.pcrv,))
        self.lreg.fit(self._x_train, self._y_train)  

        # Update the multi-index and coefficients retained by BCS as attributes of pcrv object
        if regression == 'bcs':
            self.pcrv.setMiCfs([self.mindex[self.lreg.used,:]], [self.lreg.cf])
    
        return self.lreg.cf

    def evaluate(self, x_eval, **kwargs):
        r"""Generates predictions and related uncertainty calculations for given input data. 
        Returns predicted y-values, along with standard deviation, covariance, and variance of predictions if applicable. 

        Args:
            x_eval (np.ndarray): 2d array of size `(N,d)` as input data for evaluation. Can also be a single sample as input.
            data_variance (bool, optional): Whether to compute posterior-predictive (i.e. add data variance) or not.

        Returns:
            dict: Values for predicted y-values, standard deviation, covariance, and variance of predictions (if applicable) as np.ndarrays.
        """
        # If single output (scalar-valued function), standard deviation and variance are calculated,
        # but not covariance.
        if self.outdim == 1:
            y_eval, y_eval_var, _ = self.lreg.predict(x_eval, msc=1, pp=kwargs.get('data_variance', False))
            y_eval_std = np.sqrt(y_eval_var)
            y_eval_cov = None

        # If multiple outputs (vector-valued function), all three are calculated.
        else: 
            y_eval, y_eval_var, y_eval_cov = self.lreg.predict(x_eval, msc=2)
            y_eval_std = np.sqrt(np.diag(y_eval_cov))

        if self.regression_method == 'lsq':
            y_eval_std = None
            y_eval_var = None
            y_eval_cov = None

        # Output dictionary
        results = {
            'Y_eval': y_eval,
            'Y_eval_std': y_eval_std, 
            'Y_eval_cov': y_eval_cov, 
            'Y_eval_var': y_eval_var, 
        }

        return results
    