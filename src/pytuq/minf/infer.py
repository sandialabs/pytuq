#!/usr/bin/env python
"""Inference module including embedded model error."""

import sys
import numpy as np

from ..rv.pcrv import PCRV, PCRV_iid, PCRV_mvn
from ..utils.xutils import idt
from ..utils.mindex import get_mi

class Infer(object):
    """Inference class holding all necessary information of model parameter inference problem.

    Attributes:
        chainInit (np.ndarray): An array of size `p`, the initial condition of the MCMC.
        chdim (int): Dimensionality of chain `p`.
        chind (list): List of pairs (parameter index, coefficient index) for chain elements.
        datavar (np.ndarray): Data variance in the inference problem, an array of size `N`.
        dvparams (list or tuple): Parameter list relevant to data variance computation.
        dvtype (str): Data variance type. Options are 'var_fixed', 'std_infer', 'std_infer_log', 'std_prop_fixed', 'var_fromdata_fixed', 'log_var_fromdata_fixed', 'var_fromdata_infer', 'log_var_fromdata_infer', 'scale_var'.
        extrainferparams (int): Number of extra (hyper-)parameters to infer.
        fixindnom (np.ndarray): An array of size `(K,2)`, where first column indicates indices of parameters that are fixed (i.e. not part of the inference), and the second column is their nominal, fixed values.
        ind_calib (list): Model output indices that are used for calibration.
        inpcrv (rv.mrv.PCRV): Input PC object.
        inpdf_type (str): Embedded PDF type. Options are 'pci' and 'pct'.
        Likelihood (likelihoods.Likelihood): Likelihood object.
        md_transform (callable): Potentially, a transform to be applied to model and data before the likelihood computation starts.
        model (callable): Model with signature `f(p, q)`, where `p` are model parameters of interest, and `q` are other helpful model parameters.
        model_params (tuple or list): Model parameters `q`.
        ndata (int): Number of data points (design locations) `N`.
        neachs (list[int]): List of size `N` including the number of data samples per location.
        nqd (int): Number of quadrature points per dimension required for the output PC evaluation.
        outord (int): Order for the output PC in the likelihood computation.
        pc_type (str): Embedded PC type. Can be 'LU' or 'HG'.
        pcflat (list): List of flattened PC coefficients. The ones to be inferred are None, to be populated in prior and likelihood computation.
        Prior (priors.Prior): Prior object.
        prior_is_set (bool): Boolean flag indicating that the prior is set.
        verbose (int): Verbosity level.
        ydata (list[np.ndarray]): List of `N` 1d arrays corresponding to data for each design location.
        ydata_var (np.ndarray): An array of size `N` holding data variance for each data location.
    """

    def __init__(self, verbose=1):
        """Instantiation of an inference object.

        Args:
            verbose (int, optional): Verbosity level.
        """
        self.extrainferparams = 0
        self.verbose = verbose

        self.data_is_set = False
        self.datavar_is_set = False

        self.model_is_set = False
        self.modelinput_is_set = False
        self.modeloutput_is_set = False

        self.prior_is_set = False
        self.lik_is_set = False



    def setData(self, ydata, datamode=None):
        """Set the data.

        Args:
            ydata (list or np.ndarray): List of `N` 1d arrays corresponding to data for each design location, or a 2d array of size `(N,e)`, or an 1d array of size `N`.
            datamode (str, optional): If 'mean', work with data means per location.
        """
        if isinstance(ydata, list):
            self.ydata = ydata.copy()
        elif isinstance(ydata, np.ndarray):
            if len(ydata.shape) == 2:
                self.ydata = [ydata[j,:] for j in range(ydata.shape[0])]
            elif len(ydata.shape) == 1:
                self.ydata = [[ydata[j]] for j in range(ydata.shape[0])]
            else:
                print(f"ydata has a wrong shape {ydata.shape}. Exiting.")
                sys.exit()
        else:
            print(f"ydata has a wrong type {type(ydata)}. \
                  Can be a list or an array. Exiting.")
            sys.exit()

        self.ndata = len(self.ydata)
        self.ydata_var = np.zeros((self.ndata,))

        if datamode == 'mean':
            print("Taking data mean as data.")
            a, b = self.getDataStats()
            self.ydata = list([[j] for j in a])
            self.ydata_var = b

        self.neachs = [len(yy) for yy in self.ydata]

        if self.verbose>0:
            print(f"Set: Number of data  points : {self.ndata}")

        self.data_is_set = True

    def getDataStats(self):
        """Get the first two moments of the data set.

        Returns:
            tuple: Mean and variance arrays of length `N`.
        """
        ydata_mean = np.array([np.mean(yy) for yy in self.ydata])
        ydata_var = np.array([np.var(yy) for yy in self.ydata])

        return ydata_mean, ydata_var

    def setDataVar(self, dvtype, dvparams):
        """Set data variance for the inference.

        Args:
            dvtype (str): Data variance type. Options are 'var_fixed', 'std_infer', 'std_infer_log', 'std_prop_fixed', 'var_fromdata_fixed', 'log_var_fromdata_fixed', 'var_fromdata_infer', 'log_var_fromdata_infer', 'scale_var'.
            dvparams (list or tuple): Parameter list relevant to data variance computation.
        """
        assert(self.data_is_set)

        self.dvtype = dvtype
        self.dvparams = dvparams

        if self.dvtype == 'var_fixed':
            datavar = self.dvparams[0]
            if isinstance(datavar, np.ndarray):
                self.datavar = datavar
            elif isinstance(datavar, float):
                self.datavar=datavar*np.ones(self.ndata)
            elif isinstance(datavar, str):
                self.datavar=np.loadtxt(datavar).reshape(-1,)
            else:
                print('XError: datavar type not recognized. Exiting', type(datavar))
                sys.exit()

        elif self.dvtype == 'std_infer':
            self.extrainferparams += 1

        elif self.dvtype == 'std_infer_log':
            self.extrainferparams += 1

        elif self.dvtype == 'std_prop_fixed':
            stdfactor = self.dvparams[0]
            datastd = stdfactor * np.abs(np.array([np.mean(yy) for yy in self.ydata]))
            self.datavar = datastd**2

        elif self.dvtype == 'var_fromdata_fixed':
            varfactor = self.dvparams[0]
            self.datavar = varfactor * (np.array([np.var(yy) for yy in self.ydata]))
        elif self.dvtype == 'log_var_fromdata_fixed':
            varfactor = self.dvparams[0]
            self.datavar = varfactor * (np.array([np.var(np.log(yy)) for yy in self.ydata]))

        elif self.dvtype == 'var_fromdata_infer':
            self.extrainferparams += 1
        elif self.dvtype == 'log_var_fromdata_infer':
            self.extrainferparams += 1
        elif self.dvtype == 'scale_var':
            self.datavar = self.dvparams[0]
            self.extrainferparams += 1

        else:
            print(f"Error: data variance type {self.dvtype} is unknown. Exiting.")
            sys.exit()

        if self.verbose>0:
            print(f"Set: Data variance type : {self.dvtype}") # Need to make this more informative

        self.datavar_is_set = True


    def setModelRVoutput(self, outord, nqd=None):
        """Set the model output PC settings.

        Args:
            outord (int): Order for the output PC in the likelihood computation.
            nqd (int, optional): Number of quadrature points per dimension required for the output PC evaluation. If None, sets to 2*outord+1.
        """
        self.outord = outord
        if nqd is None:
            self.nqd = 2*outord+1
        else:
            self.nqd = nqd

        if self.verbose>0:
            print(f"Set: Order of embedded output PC : {self.outord}")
            print(f"Set: Number of quadrature points for embedded output PC : {self.nqd}")

        self.modeloutput_is_set = True

    def setModelRVinput(self, inpdf_type, pc_type, pdim, rndind):
        """Set the model input parameters PC.

        Args:
            inpdf_type (str): Embedded PDF type. Options are 'pci' and 'pct'.
            pc_type (str): Embedded PC type. Can be 'LU' or 'HG'.
            pdim (int): Parameter dimensionality, i.e. number of model parameters.
            rndind (list[ind]): List of indices of parameters to be embedded. If None, embeds in all parameters.
        """
        self.inpdf_type = inpdf_type
        self.pc_type = pc_type

        if rndind is None:
            rndind = range(pdim)

        if len(rndind)>0:
            assert(len(rndind)==len(set(rndind)))
            assert(max(rndind)<pdim)
            assert((np.array(rndind)>=0).all())
            assert((np.array(rndind)<pdim).all())
            assert(len(rndind)<=pdim)

        if self.inpdf_type=='pci':
            orders = np.zeros(pdim, dtype=int)
            orders[rndind]=1
            self.inpcrv = PCRV_iid(pdim, self.pc_type, orders=orders)
        elif self.inpdf_type=='pct':
            assert(self.pc_type=='HG')
            self.inpcrv = PCRV_mvn(pdim, rndind=rndind)
        else:
            print(f"Input PDF type {self.inpdf_type} is unknown")

        if self.verbose>0:
            print(f"Set: Embedded PC type : {self.pc_type}")
            print(f"Set: Embedding type : {self.inpdf_type}")
            print(f"Set: {self.inpcrv}")

        self.modelinput_is_set = True


    def setModel(self, model, model_params, md_transform=None, fixindnom=[], ind_calib=None):
        """Set the model evaluation necessities.

        Args:
            model (callable): Model with signature `f(p, q)`, where `p` are model parameters of interest, and `q` are other helpful model parameters.
            model_params (tuple or list): Model parameters `q`.
            md_transform (callable, optional): Potentially, a transform to be applied to model and data before the likelihood computation starts. Default is None, i.e. identity function.
            fixindnom (list, optional): An array of size `(K,2)`, where first column indicates indices of parameters that are fixed (i.e. not part of the inference), and the second column is their nominal, fixed values. Defaults to an empty list, i.e. all parameters are inferred, none are fixed.
            ind_calib (list, optional): Model output indices that are used for calibration. Default is None, i.e. all outputs being used for calibration.
        """
        assert(self.data_is_set)

        self.model = model
        self.model_params = model_params
        if md_transform is None:
            self.md_transform = idt
        else:
            self.md_transform = md_transform

        ff = np.array(fixindnom)
        if len(fixindnom)>0:
            self.fixindnom = ff[ff[:, 0].argsort()]
        else:
            self.fixindnom = np.empty((0, 2))

        assert len(np.unique(self.fixindnom[:,0]))==len(self.fixindnom[:,0]), "Fixindnom has duplicates"
        assert (self.fixindnom[:,0]>=0).all(), "Fixindnom indices have to be non-negative."

        if ind_calib is not None:
            self.ind_calib = ind_calib
        else:
            self.ind_calib = range(self.ndata)

        assert(len(self.ind_calib) == self.ndata)

        if self.verbose>0:
            print(f"Set: Calibrating outputs : {self.ind_calib}")
            for fixind in self.fixindnom:
                print(f"Set: Fixing Parameter {int(fixind[0])} at value {fixind[1]}")

        self.model_is_set = True

    def setChain(self, default_init=0.1):
        """Setting the necessities for the chain.

        Args:
            default_init (float, optional): Default value for initialization of all chain dimensions. It is used for non-constant PC coefficients, and for constant terms if domain is not given. Defaults to. 0.1.
        """

        assert(self.prior_is_set)
        assert(self.model_is_set)
        assert(self.datavar_is_set)

        self.chainInit = []
        self.pcflat = []
        self.chind = []
        suffix = ''
        for pind in self.inpcrv.pind:
            if pind[1] == 0:
                if pind[0] not in self.fixindnom[:,0]:
                    self.chainInit.append(self.Prior.mean[pind[0]])
                    self.pcflat.append(None)
                    self.chind.append(pind)
                    suffix = ''
                else:
                    self.pcflat.append(self.fixindnom[np.where(self.fixindnom[:,0]==pind[0])[0][0],1])
                    suffix = ": fixed"
            else:
                self.chainInit.append(default_init)
                self.pcflat.append(None)
                self.chind.append(pind)
                suffix = ''

            if self.verbose>0:
                print(f"     Parameter {pind[0]}, PC coef {pind[1]}"+suffix)

        for j in range(self.extrainferparams):
            self.chainInit.append(default_init)
            if self.verbose>0:
                print(f"     Data variance parameter {j}")

        self.chainInit = np.array(self.chainInit)
        self.chdim = len(self.chainInit)
        assert(self.chdim>0)


        if self.verbose>0:
            print(f"Set: Initial chain value : {self.chainInit}")
            print(f"Set: Chain dimensionality : {self.chdim}")

        self.chain_is_set = True

    def setLikelihood(self, likelihood):
        """Setting the likelihood object.

        Args:
            likelihood (likelihoods.Likelihood): Likelihood object.
        """
        self.Likelihood = likelihood
        if self.verbose>0:
            print(f"Set: {self.Likelihood}")

        self.lik_is_set = True

    def setPrior(self, prior):
        """Setting the prior object.

        Args:
            prior (priors.Prior): Prior object.
        """
        self.Prior = prior
        if self.verbose>0:
            print(f"Set: {self.Prior}")

        self.prior_is_set = True

    def evalLogPost(self, pp):
        """Evaluating log-posterior.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            float: Value of the log-posterior.
        """
        assert(self.prior_is_set)
        assert(self.lik_is_set)

        logprior = self.Prior.eval(pp)
        if logprior < -1.e+79:
            return logprior

        loglik = self.Likelihood.eval(pp)

        return logprior + loglik

    def getIOSamples(self, pp, nxi=1, fmode=None):
        """Get input and output samples.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.
            nxi (int or np.ndarray, optional): Number of samples requested. It can be integer number of samples, or a presampled array. Defaults to 1.
            fmode (tuple, optional): Mode of evaluation a tuple of [model, model_parameters, index of calibrated outputs, final transform]. Defaults to None, which means get this objects internal model features.

        Returns:
            tuple[np.ndarray]: Tuple of two arrays, one for input samples and one for the output samples.
        """
        psample = self.getInSamples(pp, nxi=nxi)
        fsample = self.getOutSamples(psample, fmode=fmode)

        return psample, fsample

    def getInSamples(self, pp, nxi=1):
        """Get input parameter samples.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.
            nxi (int or np.ndarray, optional): Number of samples requested. It can be integer number of samples, or a presampled array. Defaults to 1.

        Returns:
            np.ndarray: Array of parameter samples.
        """
        assert(self.chain_is_set)
        assert(self.modelinput_is_set)

        pcf_flat = np.array(self.pcflat)
        pcf_flat[pcf_flat==None]=pp[:len(pp)-self.extrainferparams]
        self.inpcrv.cfsUnflatten(pcf_flat)

        if isinstance(nxi, int):
            psample = self.inpcrv.sample(nxi)
        else:
            psample = self.inpcrv.evalPC(nxi)

        return psample

    def getOutSamples(self, psample, fmode=None):
        """Get model output parameter samples.

        Args:
            psample (np.ndarray): Input parameter array.
            fmode (tuple, optional): Mode of evaluation a tuple of [model, model_parameters, index of calibrated outputs, final transform]. Defaults to None, which means get this objects internal model features.

        Returns:
            np.ndarray: Array of model output samples.
        """
        assert(self.model_is_set)

        if fmode is None:
            model = self.model
            model_params = self.model_params
            idx = self.ind_calib
            md_transform = self.md_transform
        else:
            model, model_params, idx, md_transform = fmode
            if md_transform is None:
                md_transform = self.md_transform

        fsample = md_transform(model(psample, model_params))[:, idx]

        return fsample

    def getModelMoments_NISP(self, pp, fmode=None):
        """Get model output moments using non-intrusive spectral projection.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.
            fmode (tuple, optional): Mode of evaluation a tuple of [model, model_parameters, index of calibrated outputs, final transform]. Defaults to None, which means get this objects internal model features.

        Returns:
            tuple: Tuple of two elements, means and variances of model outputs.
        """
        assert(self.model_is_set)
        assert(self.modelinput_is_set)
        assert(self.modeloutput_is_set)
        assert(self.chain_is_set)

        if fmode is None:
            idx = self.ind_calib
        else:
            _, _, idx, _ = fmode


        mindex = get_mi(self.outord, self.inpcrv.sdim)
        outpc = PCRV(len(idx), self.inpcrv.sdim, self.pc_type, mi=mindex)

        xx, ww = outpc.quadGerm([self.nqd] * self.inpcrv.sdim)

        model_input, yy = self.getIOSamples(pp, nxi=xx, fmode=fmode)

        cfs_list = []
        for iout in range(yy.shape[1]):
            outnorms = outpc.evalBasesNormsSq(iout)
            Amat = outpc.evalBases(xx, iout)
            cfs = np.dot(Amat.T, ww * yy[:, iout]) / outnorms
            cfs_list.append(cfs)
        outpc.setCfs(cfs_list)
        means = outpc.computeMean()
        variances = outpc.computeVar()

        return means, variances


    def getDataVar(self, pp):
        """Get data variance depending on the model of data variance treatment.

        Args:
            pp (np.ndarray): Input array of the size `p`, the dimensionality of the chain.

        Returns:
            np.ndarray: An array of size `N` for data variance per data location.
        """
        assert(self.datavar_is_set)

        if self.dvtype == 'std_infer':
            datavar = ( pp[-1] ** 2 ) * np.ones(self.ndata,)
        elif self.dvtype == 'std_infer_log':
            datavar = ( np.exp(pp[-1]) ** 2 ) * np.ones(self.ndata,)
        elif self.dvtype == 'var_fromdata_infer':
            datavar =  pp[-1]  * np.array([np.var(yy) for yy in self.ydata])
        elif self.dvtype == 'log_var_fromdata_infer':
            datavar =  np.abs(pp[-1])  * np.array([np.var(np.log(yy)) for yy in self.ydata])
        elif self.dvtype == 'scale_var':
            datavar = np.exp(pp[-1]) * self.datavar
        else:
            datavar = self.datavar

        return datavar


