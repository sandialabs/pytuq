#!/usr/bin/env python
"""Various utilities and classes for global sensitivity analysis."""


import os
import sys
import numpy as np
import pickle as pk

from ..lreg.lreg import lsq
from ..rv.pcrv import PCRV
from ..utils.mindex import get_mi
from ..utils.plotting import plot_sens
from ..utils.maps import scale01ToDom

###################################################################
###################################################################
###################################################################

# Sensitivity of a multioutput model
def model_sens(model, model_params, domain, method='SamSobol', nsam=100, plot=True, **kwargs):
    r"""Sensitivities of a multioutput model.

    Args:
        model (callable): Model of a form :math:`M(x, p)` where we are interesting in sensitivities with respect to :math:`x`, and :math:`p` are auxiliary parameters. The model is multioutput, taking :math:`(N,d)`-sized array, and returning a :math:`(N,o)`-sized array.
        model_params (list): List of model's auxiliary parameters :math:`p`.
        domain (np.ndarray): Domain of input, a 2d array of size :math:`(d,2)`.
        method (str, optional): Sensitivity method. Default is SamSobol. Other options are PCSobol, LinReg or Moat.
        nsam (int, optional): Number of samples. Note that for some methods this is not the number of model evaluations needed.
        plot (bool, optional): Whether to plot or not.
        **kwargs: Other keyword arguments for initializing sensitivity objects.

    Returns:
        tuple(np.ndarray, np.ndarray): Main and total sensitivities arrays, each of size :math:`(o,d)`.
    """
    if method == "SamSobol":
        sMethod = SamSobol(domain, **kwargs)
    elif method == "PCSobol":
        sMethod = PCSobol(domain, **kwargs)
    elif method == "LinReg":
        sMethod = Linreg(domain, **kwargs)
    elif method == "Moat":
        sMethod = Moat(domain, **kwargs)
    else:
        print(f'Sensitivity method {method} is unknown. Exiting.')
        sys.exit()

    xsam = sMethod.sample(nsam)

    ysam = model(xsam, model_params)

    ndim = xsam.shape[1] #=domain.shape[0]
    nout = ysam.shape[1] # what if 1d??
    sens_main = np.empty((nout, ndim))
    sens_tot = np.empty((nout, ndim))
    for i in range(nout):
        print(f'Output # {i+1}')
        sens = sMethod.compute(ysam[:, i])
        sens_main[i, :] = sens['main']
        sens_tot[i, :] = sens['total']

    if plot:
        plot_sens(sens_main,range(ndim),range(nout),xticklabel_size=25, figname='sens_main.png')
        plot_sens(sens_tot,range(ndim),range(nout),xticklabel_size=25, figname='sens_tot.png')

    return sens_main, sens_tot

####################################################################
####################################################################
####################################################################

class SensMethod():
    """Base class for various sensitivity methods implementation.

    Attributes:
        dim (int): Dimensionality of the input.
        dom (np.ndarray): Domain of input, a 2d array of size :math:`(d,2)`.
        sens_names (list): List of names that are keys to dictionary sens.
        sens (dict): Dictionary of sensitivities under a given method.
    """

    def __init__(self,dom, sens_names):
        self.dom=dom
        self.dim=dom.shape[0]
        self.sens_names = sens_names
        self.sens = dict((k, [None] * self.dim) for k in self.sens_names)


    def sample(self,nsam):
        r"""Sampling routine.

        Args:
            nsam (int): Number of requested samples, :math:`N`. Note that for some methods this is not the number of model evaluations needed.

        Raises:
            NotImplementedError: Should be implemented in children classes.
        """
        raise NotImplementedError("Sampling for sensitivity is not implemented in the base class.")


    def compute(self,ysam):
        r"""Computing sensitivities, given model evaluations.

        Args:
            ysam (np.ndarray): A 2d array of model evaluations of size :math:`(M,d)`.

        Raises:
            NotImplementedError: Should be implemented in children classes.
        """
        raise NotImplementedError("Computing sensitivity is not implemented in the base class.")

###################################################################
###################################################################
###################################################################

class Linreg(SensMethod):
    r"""Sensitivities computed via linear regression.

    Attributes:
        nsam (int): Number of samples requested, :math:`N`.
        sens (dict): Dictionary of sensitivities.
        sens_names (list): Names of sensitivities: 'src' (scaled regression coefficient) or 'pear' (pearson).
        xsam (np.ndarray): Model evaluation input samples, a 2d array of size :math:`(N,d)`.
    """

    def __init__(self, dom):
        r"""Initialization.

        Args:
            dom (np.ndarray): Domain of input, a 2d array of size :math:`(d,2)`.
        """
        print("Initializing LINREG Sensitivity Method")
        self.sens_names=['src', 'pear']
        super().__init__(dom, self.sens_names)


    def sample(self, nsam):
        """Sampling routine.

        Args:
            nsam (int): Number of requested samples, :math:`N`.

        Returns:
            np.ndarray: Model evaluation input samples, a 2d array of size :math:`(N,d)`.
        """
        print("Sampling LINREG Sensitivity Method")
        self.nsam = nsam
        self.xsam = scale01ToDom(np.random.rand(self.nsam,self.dim),self.dom)


        return self.xsam

    def compute(self, ysam):
        r"""Computing sensitivities, given model evaluations.

        Args:
            ysam (np.ndarray): A 2d array of model evaluations of size :math:`(N,d)`.

        Returns:
            dict: Dictionary with keys src (scaled regression coefficient) and pear (Pearson)
        """
        assert(ysam.shape[0]==self.nsam)

        A=np.ones((self.nsam,self.dim+1))
        A[:,1:]=self.xsam
        coeff=np.linalg.lstsq(A, ysam)[0]

        sy=np.std(ysam, ddof=1)
        sxs=np.std(self.xsam, axis=0, ddof=1)

        self.sens['src']=np.multiply(coeff[1:],sxs)/sy

        pcor=np.zeros((self.dim,))
        yx=np.zeros((2,self.nsam))
        for id in range(self.dim):
            yx[0,:]=ysam
            yx[1,:]=self.xsam[:,id]
            cc=np.corrcoef(yx)
            pcor[id]=cc[0,1]

        self.sens['pear']=pcor

        return self.sens

###################################################################
###################################################################
###################################################################

class Moat(SensMethod):
    """Class for MOAT (Morris One At a Time) sensitivities.

    Attributes:
        delta (int): Delta-parameter of the method.
        indmap (np.ndarray): Working 2d array of size :math:`(d,2)`.
        nlev (int): Number of levels parameter of the method.
        nsam (int): Number of model evaluations :math:`M=R(d+1)`.
        repl (int): Number of replicas, :math:`R`.
        sens (dict): Dictionary of sensitivities.
        sens_names (list): Sensitivity names: mu, amu or sig.
    """

    def __init__(self,dom, delta=2, nlev=4):
        r"""Initialization

        Args:
            dom (np.ndarray): Domain of input, a 2d array of size :math:`(d,2)`.
            delta (int, optional): Delta-parameter of the method. Defaults to 2.
            nlev (int, optional): Number of levels parameter of the method. Defaults to 4.
        """
        print("Initializing MOAT Sensitivity Method")
        self.sens_names=['mu', 'amu', 'sig']
        super().__init__(dom, self.sens_names)

        self.delta = delta
        self.nlev = nlev

    def sample(self, repl):
        r"""Sampling routine.

        Args:
            repl (int): Number of replicas, :math:`R`.

        Returns:
            np.ndarray: Model evaluation input samples, a 2d array of size :math:`(M,d)`, where :math:`M=R(d+1)`.
        """
        print("Sampling MOAT Sensitivity Method")

        self.indmap=np.zeros((self.dim,2),dtype=int)

        self.nsam = repl * (self.dim + 1)
        self.repl = repl


        J     = np.ones((self.dim + 1, self.dim),dtype=int)
        B     = np.tril(J, -1)

        Dst   = np.diag(2*np.random.randint(2, size=self.dim)-1)
        Perm  = np.random.permutation(self.dim)
        Pst   = np.eye(self.dim,dtype=int)[:, Perm] #np.random.permutation(np.eye(self.dim,dtype=int))

        for id in range(self.dim):
            self.indmap[id,0] = Perm[id]
            self.indmap[id,1] = -Dst[Perm[id], Perm[id]]


        Bst=np.empty((repl,self.dim+1,self.dim),dtype=int)
        for ir in range(repl):
            xind=np.random.randint(self.nlev-self.delta, size=(self.dim,))
            #Bprime=np.dot(J,np.diag(xind))+delta*B
            Bst[ir,:,:] = np.dot(np.dot(J,np.diag(xind))+(self.delta/2)*(np.dot(2*B-J,Dst)+J),Pst)
            #print Bst[ir,:,:]

        xsam=np.empty((repl*(self.dim+1),self.dim))
        for ir in range(repl):
            for jd in range(self.dim+1):
                ind=Bst[ir,jd,:]
                xsam[ir*(self.dim+1)+jd,:]=self.dom[:,0]+(self.dom[:,1]-self.dom[:,0])*np.array(ind)/(self.nlev-1)

        return xsam

    def compute(self, ysam):
        r"""Computation of MOAT sensitivities.

        Args:
            ysam (np.ndarray): A 2d array of size :math:`(M, d)`.

        Returns:
            dict: Dictionary with keys mu, amu and sig.
        """
        assert(ysam.shape[0]==self.nsam)

        repl=self.nsam/(self.dim+1)
        ee=np.zeros((self.dim,repl))
        for id in range(self.dim):
            jd=self.indmap[id,0]
            flag=self.indmap[id,1]
            for ir in range(repl):
            #for jd in range(self.dim+1):
                ee[id,ir]=ysam[ir*(self.dim+1)+jd]-ysam[ir*(self.dim+1)+jd+1]
                ee[id,ir]/= float(flag*self.delta*(self.dom[id,1]-self.dom[id,0]))/float(self.nlev-1)

        self.sens['mu']  = np.average(ee,axis=1)
        self.sens['amu'] = np.average(abs(ee),axis=1)
        self.sens['sig'] = np.std(ee, axis=1, ddof=1)


        return self.sens


###################################################################
###################################################################
###################################################################

class SamSobol(SensMethod):
    """Computing of sampling based main and total sensitivities, see Saltelli 2010.

    Attributes:
        nsam (int): Number of model evaluations.
        sens (dict): Dictionary of sensitivities.
        sens_names (list): List of sensitivity names, main, total and jointt.

    Note:
        It computes joint in the total sense!
    """

    ##

    def __init__(self,dom):
        """Initialization.

        Args:
            dom (np.ndarray): Domain of input, a 2d array of size :math:`(d,2)`.
        """
        print("Initializing SamSOBOL Sensitivity Method")
        self.sens_names=['main', 'total', 'jointt']
        super().__init__(dom, self.sens_names)

    def sample(self,ninit):
        r"""Sampling routine.

        Args:
            ninit (int): Initial number of samples, :math:`N`.

        Returns:
            np.ndarray: Model evaluation input samples, a 2d array of size :math:`(M,d)`, where :math:`M=N(d+2)`.
        """
        print("Sampling SamSOBOL Sensitivity Method")

        sam1=scale01ToDom(np.random.rand(ninit, self.dim), self.dom)
        sam2=scale01ToDom(np.random.rand(ninit, self.dim), self.dom)

        xsam=np.vstack((sam1,sam2))

        for id in range(self.dim):
            samid=sam1.copy()
            samid[:,id]=sam2[:,id]
            xsam=np.vstack((xsam,samid))

        self.nsam=xsam.shape[0]


        return xsam

    def compute(self,ysam):
        """Computing sensitivities.

        Args:
            ysam (np.ndarray): A 2d array of size :math:`(M, d)`.

        Returns:
            dict: Dictionary with keys main, total and jointt.
        """
        ninit=self.nsam//(self.dim+2)
        y1=ysam[ninit:2*ninit]
        var=np.var(ysam[:2*ninit])
        si=np.zeros((self.dim,))
        ti=np.zeros((self.dim,))
        jtij=np.zeros((self.dim,self.dim))

        if var == 0:
            self.sens = {'main' : si, 'total' : ti, 'jointt' : jtij.T}
            return self.sens

        for id in range(self.dim):
            y2=ysam[2*ninit+id*ninit:2*ninit+(id+1)*ninit]-ysam[:ninit]
            si[id]=np.mean(y1*y2)/var
            ti[id]=0.5*np.mean(y2*y2)/var
            for jd in range(id):
                y3=ysam[2*ninit+id*ninit:2*ninit+(id+1)*ninit]-ysam[2*ninit+jd*ninit:2*ninit+(jd+1)*ninit]
                jtij[id,jd]=ti[id]+ti[jd]-0.5*np.mean(y3*y3)/var
                jtij[jd,id] = jtij[id,jd]

        self.sens['main']=si
        self.sens['total']=ti
        self.sens['jointt']=jtij.T

        return self.sens

###################################################################
###################################################################
###################################################################

class PCSobol(SensMethod):
    r"""PC-based Sobol senitivity computation.

    Attributes:
        nsam (int): Number of model evaluations.
        sens (dict): Dictionary of sensitivities.
        sens_names (list): List of sensitivity names, main, total and jointt.
        pcrv (PCRV): Working PCRV object
        pctype (str): PC type.
        order (int): PC order.
        xsam (np.ndarray): Model evaluation input samples, a 2d array of size :math:`(N,d)`.
        germ_sam (np.ndarray): Corresponding PC germ samples, a 2d array of size :math:`(N,d)`.
    """

    def __init__(self,dom, pctype='LU', order=3):
        r"""Initialization.

        Args:
            dom (np.ndarray): Domain of input, a 2d array of size :math:`(d,2)`.
            pctype (str, optional): PC type. Defaults to LU.
            order (int, optional): PC order. Defaults to :math:`3`.
        """
        print("Initializing PCSobol Sensitivity Method")
        self.sens_names=['main', 'total', 'jointt']
        super().__init__(dom,self.sens_names)

        self.pctype = pctype
        self.order = order

    def sample(self,nsam):
        r"""Sampling routine.

        Args:
            nsam (int): Number of requested samples, :math:`N`.

        Returns:
            np.ndarray: Model evaluation input samples, a 2d array of size :math:`(N,d)`.
        """
        print("Sampling PCSobol Sensitivity Method")
        self.pcrv = PCRV(1, self.dim, self.pctype, mi=get_mi(self.order, self.dim))
        self.nsam=nsam
        self.germ_sam=self.pcrv.sampleGerm(nsam=self.nsam)

        unif_sam = np.zeros((self.nsam, self.dim))
        for idim in range(self.dim):
            unif_sam[:, idim] = self.pcrv.PC1ds[idim].germCdf(self.germ_sam[:, idim])

        self.xsam = scale01ToDom(unif_sam,self.dom)


        return self.xsam

    def compute(self, ysam):
        r"""Computing sensitivities.

        Args:
            ysam (np.ndarray): A 2d array of size :math:`(N, d)`.

        Returns:
            dict: Dictionary with keys main, total and jointt.
        """
        assert(ysam.shape[0]==self.nsam)

        Amat = self.pcrv.evalBases(self.germ_sam, 0)
        lreg = lsq()
        lreg.fita(Amat, ysam)
        self.pcrv.setCfs([lreg.cf])
        self.sens['main'] = self.pcrv.computeSens()[0]
        self.sens['total'] = self.pcrv.computeTotSens()[0]
        self.sens['jointt'] = self.pcrv.computeJointSens()[0]



        return self.sens

