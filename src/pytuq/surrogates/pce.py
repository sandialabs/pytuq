#!/usr/bin/env python

"""This module provides a Polynomial Chaos Expansion (PCE) wrapper class to facilitate 
    the universal coupling of FASTMath UQ tools and libraries. This class focuses on the use case 
    of PC surrogate models built with linear regression, keeping in mind 
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
import math
import copy
from matplotlib import pyplot as plt

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.lreg.lreg import lsq
from pytuq.lreg.anl import anl
from pytuq.lreg.bcs import bcs


class PCE:
    r"""A wrapper class to access PyTUQ functionalities for PCE surrogate models. 

    Attributes:
        pcrv (PCRV object): Polynomial Chaos random variable object, encapsulates details for polynomial chaos expansion.
        sdim (int): Stochastic dimensionality, i.e. # of stochastic inputs.
        order (int): Order of polynomial chaos expansion.
        pctype (list[str]): Type of PC polynomial used.
        outdim (int): Physical dimensionality, i.e. # of output variables.
        lreg (lreg object): Linear regression object used for fitting the model.
        mindex (int np.ndarray): Multiindex array carrying the powers to which the basis functions will be raised to within the PC terms. Reset when build() is called again.
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


    def get_pc_terms(self):
        """Returns a list where each element represents the number of PC terms
        in the corresponding dimension of the PCE.

        Returns:
            list[int]
        """

        return [mi.shape[0] for mi in self.pcrv.mindices]


    def kfold_split(self, nsamples, nfolds, seed=13):
        """Return dictionary of training and testing pairs using k-fold cross-validation.

        Args:
            nsamples (int): Total number of training samples.
            nfolds (int): Number of folds to use for k-fold cross-validation.
            seed (int, optional): Random seed for reproducibility. Defaults to 13.

        Returns:
            dict: A dictionary where each key is the fold number (0 to nfolds-1) 
            and each value is a dictionary with:
                - "train index" (np.ndarray): Indices of training samples.
                - "val index" (np.ndarray): Indices of validation samples.
        """
        # Returns split data where each data is one fold left out
        KK = nfolds
        rn = np.random.RandomState(seed)

        # Creating a random permutation of the samples indices list
        indp=rn.permutation(nsamples)

        # Split the permuted indices into KK (or # folds) equal-sized chunks 
        split_index=np.array_split(indp,KK)

        # Dictionary to hold the indices of the training and validation samples
        cvindices = {}

        # create testing and training folds
        for j in range(KK):
            # Iterating through the number of folds
            fold = j
            # Iterate through # folds, if i != fold number, 
            newindex = [split_index[i] for i in range(len(split_index)) if i != (fold)]
            train_ind = np.array([],dtype='int64')
            for i in range(len(newindex)): train_ind = np.concatenate((train_ind,newindex[i]))
            test_ind = split_index[fold]
            cvindices[j] = {'train index': train_ind, 'val index': test_ind}

        return cvindices
    

    def kfold_cv(self, x, y, nfolds=3,seed=13):
        """Splits data into training/testing pairs for kfold cross-val
        x is a data matrix of size n x d1, d1 is dim of input
        y is a data matrix of size n x d2, d2 is dim of output

        Args:
            x (np.ndarray): Input matrix with shape (n, d1) or 1D array with shape (n,). Each row is a sample; columns are input features.
            y (np.ndarray): Target array with shape (n,) for single-output, or (n, d2) for multi-output. If 1D, it is internally reshaped to (n, 1) before slicing; outputs are `np.squeeze`d per fold.
            nfolds (int, optional): Number of folds for cross-validation. Defaults to 3.
            seed (int, optional): Random seed for reproducible shuffling in `kfold_split`. Defaults to 13.
        """

        if len(x.shape)>1:
            n,d1 = x.shape
        else:
            n=x.shape
        ynew = np.atleast_2d(y)
        if len(ynew) == 1: ynew = ynew.T # change to shape (n,1)
        _,d2 = ynew.shape
        cv_idx = self.kfold_split(n,nfolds,seed)

        kfold_data = {}
        for k in cv_idx.keys():
            kfold_data[k] = {
            'xtrain': x[cv_idx[k]['train index']],
            'xval': x[cv_idx[k]['val index']],
            'ytrain': np.squeeze(ynew[cv_idx[k]['train index']]),
            'yval': np.squeeze(ynew[cv_idx[k]['val index']])
            } # use squeeze to return 1d array

            # set train and test to the same if 1 fold
            if nfolds == 1:
                kfold_data[k]['xtrain'] = kfold_data[k]['xval']
                kfold_data[k]['ytrain'] = kfold_data[k]['yval']

        return kfold_data

    
    def optimize_eta(self, etas, verbose=0, nfolds=3, plot=False):
        """Choose the optimum eta for Bayesian compressive sensing. Calculates the RMSE for each eta for a specified number of folds. 
        Selects the eta with the lowest RMSE after averaging the RMSEs over the folds.

        Arg:
            y: 1D numpy array (vector) with function, evaluated at the sample points [#samples,]
            x: N-dimensional NumPy array with sample points [#samples, #dimensions]
            etas: NumPy array or list with the threshold for stopping the algorithm. Smaller values retain more nonzero coefficients.
            plot: Flag for whether to generate a plot for eta optimization
            verbose: Flag for print statements during cross-validation 

        Returns:
            eta_opt: Optimum eta value to be used in BCS build
        """
        # Split data in k folds -> Get dictionary of data split in training + testing folds
        kfold_data = self.kfold_cv(self._x_train, self._y_train, nfolds)

        # Each value has data for 1 fold. Each value is a list of the RMSEs for each possible eta in the fold. 
        RMSE_list_per_fold_tr = [] 

        # Same but for testing data
        RMSE_list_per_fold_test = []

        # Make a copy of the PCE object to run the cross-validation algorithm on
        pce_copy = PCE(self.sdim, self.order, self.pctype, verbose=0)

        pce_copy.pcrv = copy.deepcopy(self.pcrv) # copying over self attributes

        # Loop through each fold
        for i in range(nfolds):

            # Get the training and validation data
            x_tr = kfold_data[i]['xtrain']
            y_tr = kfold_data[i]['ytrain']
            x_test = kfold_data[i]['xval']
            y_test = kfold_data[i]['yval']
            
            # As we conduct BCS for this fold with each separate eta, the RMSEs will be added to these lists
            RMSE_per_eta_tr = [] 
            RMSE_per_eta_test = [] 

            # Set the x and y training data for the copied PCE object
            pce_copy.set_training_data(x_tr, y_tr)

            # Loop through each eta
            for eta in etas:

                # Conduct the BCS fitting. The object is automatically updated with new multiindex and coefficients received from the fitting.
                cfs = pce_copy.build(regression = 'bcs', eta=eta)

                # Evaluate the PCE object at the training and validation points 
                y_tr_eval = (pce_copy.evaluate(x_tr))['Y_eval']
                y_test_eval = (pce_copy.evaluate(x_test))['Y_eval']

                # Print statement for verbose flag
                if verbose > 0:
                    print("Fold " + str(i + 1) + ", eta " + str(eta) + ", " + str(len(cfs)) + " terms retained out of a full basis of size " + str(len(pce_copy.pcrv.mindices[0])))
                
                # Calculate the RMSEs for the training and validation points.
                # Append the values into the list of etas per fold.
                MSE = np.square(np.subtract(y_tr, y_tr_eval)).mean()
                RMSE = math.sqrt(MSE)
                RMSE_per_eta_tr.append(RMSE)

                MSE = np.square(np.subtract(y_test, y_test_eval)).mean()
                RMSE = math.sqrt(MSE)
                RMSE_per_eta_test.append(RMSE)

            # Now, append the fold's list of RMSEs for each eta into the list carrying the lists for all folds 
            RMSE_list_per_fold_tr.append(RMSE_per_eta_tr)
            RMSE_list_per_fold_test.append(RMSE_per_eta_test)

        # After compiling the RMSE data for each eta from all the folds, we find the eta with the lowest validation RMSE to be our optimal eta.
        # Compute the average and standard deviation of the training and testing RMSEs over the folds
        avg_RMSE_tr = np.array(RMSE_list_per_fold_tr).mean(axis=0)
        avg_RMSE_test = np.array(RMSE_list_per_fold_test).mean(axis=0)

        std_RMSE_tr = np.std(np.array(RMSE_list_per_fold_tr), axis=0)
        std_RMSE_test = np.std(np.array(RMSE_list_per_fold_test), axis=0)

        # Choose the eta with lowest RMSE across all folds' testing data
        eta_opt = etas[np.argmin(avg_RMSE_test)]

        # Plot RMSE vs. eta for training and testing RMSE
        if plot:

            fig, ax = plt.subplots(figsize=(10,10))

            plt.errorbar(etas, avg_RMSE_tr, xerr=None, yerr=std_RMSE_tr, linewidth=2, markersize=8, capsize=8, label=('Training'))
            plt.errorbar(etas, avg_RMSE_test, xerr=None, yerr=std_RMSE_test, linewidth=2, markersize=8, capsize=8, label=('Validation'))

            plt.plot(eta_opt, np.min(avg_RMSE_test), marker="o", markersize=15, color='black', label=("Optimum"))

            plt.xlabel("Eta",fontsize=20)
            plt.ylabel("RMSE",fontsize=20)

            # Change size of tick labels
            plt.tick_params(axis='both', labelsize=16)

            plt.xscale('log')
            plt.yscale('log')

            # Create legend
            plt.legend(loc='upper left')

            # Save
            plt.savefig('eta_opt.pdf', format='pdf', dpi=1200)

        return eta_opt


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
            eta = kwargs.get('eta', 1.e-8)
            if (isinstance(eta, list) and not any(isinstance(eta, list) for item in eta)) or (isinstance(eta, np.ndarray) and eta.ndim == 1):
                opt_eta = self.optimize_eta(eta, nfolds=kwargs.get('nfolds', 3), verbose=kwargs.get('eta_verbose', False), plot=kwargs.get('eta_plot', False))
                self.lreg = bcs(eta=opt_eta, datavar_init=kwargs.get('datavar_init'))
            elif isinstance(eta, float) and eta > 0:
                self.lreg = bcs(eta=eta, datavar_init=kwargs.get('datavar_init'))
            else:
                raise ValueError("You may provide either a positive float (defaulting to 1.e-8) or 1D list/numpy array for the value of eta. If a list/numpy array is provided, the most optimal eta from the array will be chosen.")
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
    