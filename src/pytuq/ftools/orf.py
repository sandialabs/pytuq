#!/usr/bin/env python
"""Module for orthonormalization of functions using Gram-Schmidt or QR decomposition.

Written by Habib N. Najm (2025).
"""

import sys
import time
import copy
import numpy as np


class GLMAP:
    r'''General linear map and data class

    Attributes:

        npt (int)       : number of data points
        x (np.ndarray)  : 2d float data array of shape :math:`(npt,n)`, i.e. :math:`npt` points each in :math:`R^n`.
        L (np.ndarray or callable or None)        : 2d float array of shape :math:`(M,npt)` or a callable user-provided function or None

    Notes:
        Builds linear map and data object, presumed to underly a regression problem of the form
        :math:`y = L(x,f(x;w))`, given data :math:`\{(x_1,y_1),...,(x_{npt},y_{npt})\}` where :math:`x_i \in R^n`, :math:`x \in R^{npt \times n}`, and :math:`y_i\in R`.
        and where :math:`w \in R^m` is a vector of parameters to be estimated.

        L needs to be linear in :math:`f()`, with separate L dependence on :math:`x` being an arbitrary function, so that, with :math:`u = f(x,w) = \sum_k w_k f_k(x)`, we have :math:`y = L(x, \sum_k w_k f_k(x) ) = \sum_k w_k L(x,f_k(x))` or :math:`y = A w` with :math:`A \in R^{npt \times m}` where :math:`A[i,j] = L(x_i,f_j(x_i))`.

        This class can take specs of different linear maps L on functions :math:`f(x): x \rightarrow u`, with :math:`u \in R^{npt}`. We list here multiple options for illustration. The specification of the linear map ``lmap=gso.GLMAP(x,L)`` can be done with L given as either no spec, a function, or an array
            * if L is a user-defined callable L (function), then `lmap.eval(f)` returns :math:`L(x,u)` as specified by the user. If want a map, e.g.: :math:`L(x,f(x)) = e^x + x^3 f(x)`, one can use e.g. ``lmap = gso.GLMAP(x, lambda x, u: np.exp(x) + x**3 * u)``.
            * if L is a user-defined array L, i.e. :math:`L(x,u):=Lu`, (L shape needs to be :math:`(M,npt)`) then ``lmap.eval(f)`` returns :math:`Lu`. If we want a linear matrix map, say :math:`L = B` (shape :math:`(m,npt)` ), then can use either ``lmap = gso.GLMAP(x, B)`` or ``lmap = gso.GLMAP(x, lambda x, u: B@u)``.
            * if L is None or not specified, then an identiy map is implied, and ``lmap.eval(f)`` returns :math:`L(x,u) = u`; Thus, the following are all equivalent specs of the None option for L
                * ``lmap = gso.GLMAP(x)``
                * ``lmap = gso.GLMAP(x, np.eye(npt))``
                * ``lmap = gso.GLMAP(x, lambda x, u: u)``.
    '''
    def __init__(self,x,L=None,verbose=False):
        '''Initialization.
        
        Args:
            x (np.ndarray)  : 2d float data array of shape (npt,n), npt points each in R^n
            L (np.ndarray or callable or None)        : 2d float array of shape :math:`(M,npt)` or a callable user-provided function or None.
            verbose  (bool) : controls verbosity
        '''
        self.x = np.array(x)
        if verbose:
            print(f'x: {self.x.shape}')

        self.npt = self.x.shape[0]

        if L is not None:
            if callable(L):
                self.L = copy.deepcopy(L)
                if verbose:
                    print(f'L: function:',self.L)
            else:
                self.L = np.array(L)
                assert (self.L.shape[1] == self.npt), 'with L as array, it must have shape (:,npt)'
                if verbose:
                    print(f'L: matrix: {self.L.shape}')
        else:
            self.L = None
            if verbose:
                print(f'L:',self.L)

        xsln = len(self.x.shape)
        if xsln == 0:
            if verbose: print('x is a single scalar data point')
        elif xsln == 1:
            if verbose: print(f'x is a 1d array with {self.x.shape[0]} data points, each being a scalar')
        elif xsln == 2:
            if verbose: print(f'x is a 2d array with {self.x.shape[0]} data points, each being a {self.x.shape[1]}-vector')
        else:
            print(f'With xsln: {xsln}, x is not allowed')
            sys.exit('GLMAP: x has to be either a scalar, a 1d array, or a 2d array')

        if self.L is None:
            self.Lmapfnc = lambda f, x, L : f(x)
        elif isinstance(self.L,np.ndarray):
            self.Lmapfnc = lambda f, x, L : L@f(x)
        else:
            self.Lmapfnc = lambda f, x, L : L(x,f(x))

    def eval(self, f):
        r'''Evaluate the map on function f.'''
        glm = self.Lmapfnc(f, self.x, self.L)
        return glm

class HMAT:
    r'''Utilities class for H matrix construction.

    Attributes:
        mgs (object)        : MGS class object handle.
        m   (int)           : number of basis functions in expansion.
        mat (np.ndarray)    : 3D array where `H` matrix is constructed.
        level (int)         : recursion level counter.
        verbose (bool)      : controls verbosity.
        Fthmap (np.ndarray) : 2d int array, :math:`(i,j)` entry `0/1` => corresponding :math:`(\theta_i,\theta_j)` inner product has-not/has been evaluated.
        Fmap (np.ndarray)   : 2d int array, :math:`(i,j)` entry `0/1` => corresponding :math:`(\phi_i,\theta_j)` inner product has-not/has been evaluated.
        Fthval (np.ndarray) : 2d array of :math:`(\theta,\theta)` inner products.
        Fval (np.ndarray)   : 2d array of :math:`(\phi,\theta)` inner products.
    '''

    def __init__(self,mgs_,verbose=False):
        '''Initialization.

        Args:
            mgs_ (object)       : MGS class object handle.
            verbose (bool)      : control on verbosity.
        '''
        self.mgs = mgs_
        self.m   = self.mgs.m
        self.mat = np.zeros((self.m,self.m,self.m))
        self.level = 0
        self.verbose = verbose
        self.Fthmap = np.zeros((self.m,self.m),dtype=int)
        self.Fmap   = np.zeros((self.m,self.m),dtype=int)
        self.Fthval = np.zeros((self.m,self.m))
        self.Fval = np.zeros((self.m,self.m))

    def Fth(self,i,j):
        r'''Evaluate :math:`(\theta_i,\theta_j)` inner product.

        Args:
            i (int) : :math:`0 <= i < m` index for :math:`\theta_i`
            j (int) : :math:`0 <= j < m` index for :math:`\theta_j`
        '''
        self.Fthmap[i,j]+=1
        if self.Fthmap[i,j] == 1:
            self.Fthval[i,j] = self.mgs.iprod(self.mgs.tht[i],self.mgs.tht[j])[0][0] #Fprod(Tht_vec_fnc.f,Tht_vec_fnc.f,i,j,ntrn,g,t,dv,None,None)[0][0]
        return self.Fthval[i,j]

    def F(self,i,j):
        r'''Evaluate :math:`(\phi_i,\theta_j)` inner product.

        Args:
            i (int) : :math:`0 <= i < m` index for :math:`\phi_i`
            j (int) : :math:`0 <= j < m` index for :math:`\theta_j`
        '''
        self.Fmap[i,j]+=1
        if self.Fmap[i,j] == 1:
            self.Fval[i,j] = self.mgs.iprod(self.mgs.phi[i],self.mgs.tht[j])[0][0] #Fprod(Phi_vec_fnc.f,Tht_vec_fnc.f,i,j,ntrn,g,t,dv,None,None)[0][0]
        return self.Fval[i,j]

    def Heval(self,n,r,ilom,ihim):
        r'''Evaluate :math:`H` matrix recursively.

        Args:
            n (int)     : summation index limit
            r (int)     : summation index limit
            ilom (int)  : initial summation index in previous sum
            ihim (int)  : final summation index in previous sum
        '''
        self.level += 1

        ilo = ilom + 1
        ihi = ihim + 1
        sum = 0.0

        if ihi == r-1:
            if self.verbose: print('Heval(',ilo,',',ihi,') : level (last):',self.level)
            for idx in range(ilo,ihi+1):
                if self.verbose: 
                    print('idx =',idx,'\n','Fth(',ilom,',',idx,') * Fth(',idx,',',r,')')
                sum += self.Fth(ilom,idx)*self.Fth(idx,r)
            return sum

        if self.verbose: print('Heval(',ilo,',',ihi,') : level:',self.level)    
        for idx in range(ilo,ihi+1):
            if self.verbose: 
                print('idx =',idx,'\n','Fth(',ilom,',',idx,') * Heval(',ilo+1,',',ihi+1,')')
            sum += self.Fth(ilom,idx)*self.Heval(n,r,ilo,ihi)
            self.level -= 1
            if self.verbose: print('Heval(',ilo,',',ihi,') : level (return):',self.level)

        return sum

    def fill(self,n,r,i1,i):
        r'''Fill in :math:`H` matrix recursively.

        Args:
            n (int)  : summation index limit
            r (int)  : summation index limit
            i1 (int) : initial summation index
            i (int)  : relevant i index
        '''
        # i relevant here only in terms of checking bounds on n,r
        if n < 1 or n > i-1:
            print('fill: n:',n, 'out of bounds')
            sys.exit('Abort')
        if r < n or r > i-1:
            print('fill: r:',r, 'out of bounds')
            sys.exit('Abort')
        if i1 < 0 or i1 > r-n:
            print('fill: i1:',i1, 'out of bounds')
            sys.exit('Abort')

        self.level = 1

        if n == 1:
            if self.verbose: print('fill: n=1','\n','level: (last)',self.level,'\n','Fth(',i1,',',r,')')
            Hnri1 = self.Fth(i1,r)
        else:
            ilo = i1
            ihi = r-n
            if self.verbose: print('fill: n>1','\n','level:',self.level)
            Hnri1 = self.Heval(n,r,ilo,ihi)
            self.level -= 1
            if self.verbose: print('level (return):',self.level)

        if self.verbose: print('done')
        self.mat[n,r,i1] = Hnri1
        return

class MGSV:
    r'''Utilities class for :math:`V` matrix construction.

    Attributes:
        m   (int)           : number of basis functions in expansion.
        mat (np.ndarray)    : 2D :math:`(m, m)` array where the :math:`V` matrix is constructed.
        verbose (bool)      : controls verbosity.
        Smat (np.ndarray)   : 3D :math:`(m, m, m)` array where the :math:`S` matrix is constructed.
        H (object)          : pointer to HMAT object.
    '''

    def __init__(self,mgs_,verbose=False):
        '''Initialization.

        Args:
            mgs_ (object)       : MGS class object handle.
            verbose (bool)      : control on verbosity.
        '''
        self.m    = mgs_.m
        self.mat  = np.zeros((self.m,self.m))
        self.Smat = np.zeros((self.m,self.m,self.m))
        self.H    = HMAT(mgs_,verbose=False)

    def fill_row(self,i):
        r'''Fill in :math:`V` matrix row.

        Args:
            i (int)  : row index
        '''
        for n in range(1,i):
            for r in range(n,i):
                sum = 0
                for i1 in range(0,r-n+1):
                    if self.H.verbose: 
                        print('===========================')
                        print('i:',i,'n =',n,'in [1,',i-1,']  ||  r =',r,'in [',n,',',i-1,']  ||  i1 = ',i1,' in [0,',r-n,']')
                    if not self.H.mat[n,r,i1]:
                        self.H.fill(n,r,i1,i)
                    if self.H.verbose: 
                        print('H matrix:',self.H.mat[n,r,i1])
                        print('===========================')
                    sum += self.H.F(i,i1) * self.H.mat[n,r,i1]
                self.Smat[n,r,i-1] = (-1)**(n+1) * sum
        if self.H.verbose: 
            print('Smat[1:',i-1,',1:',i-1,',',i-1,']:\n',self.Smat[1:i,1:i,i-1],sep='')

        for l in range(1,i):
            self.mat[i,l] = np.sum(self.Smat[:,l,i-1])
        if self.H.verbose:
            print('Vmat[0:',self.m-1,',0:',self.m-1,']:\n',self.mat[0:self.m,0:self.m],sep='')

        return

class MGS:
    r'''Modified Gram-Schmidt (MGS) class. Builds MGS object for functions.

    Attributes:
        m   (int)           : number of basis functions in expansion.
        phi (np.ndarray)    : 1d numpy array (size :math:`m`) of starting functions.
        psi (np.ndarray)    : 1d numpy array (size :math:`m`) of to-be-constructed orthogonal functions.
        tht (np.ndarray)    : 1d numpy array (size :math:`m`) of to-be-constructed orthonormal functions.
        Lmap (object)       : pointer to local copy of Linear map and data object.
        modified (bool)     : True/False for modified/original GS orthogonalization.
        V (np.ndarray)      : 2d :math:`(m, m)` numpy array ... the V matrix.
        Z (np.ndarray)      : 2d :math:`(m, m)` numpy array ... the Z matrix.
        Pmat (np.ndarray)   : 2d :math:`(m, m)` numpy array ... the P projection matrix
        lam (np.ndarray)    : 1d numpy array of size :math:`m`.
    '''

    def __init__(self,phi_,Lmap):
        '''Initialization.

        Args:
            phi_ (np.ndarray): input numpy array of pointers to starting functions.
            Lmap (object)    : input pointer to Linear map and data object.
        '''
        self.phi    = copy.deepcopy(phi_)
        self.psi    = np.full_like(self.phi,None,dtype=object)
        self.tht    = np.full_like(self.phi,None,dtype=object)
        self.m      = self.phi.shape[0]
        self.Lmap   = copy.deepcopy(Lmap)
        return

    def iprod(self, pk, pl, **kwargs): 
        r'''Pairwise inner product between (lists of) functions.

        Args:
            pk  : a list|tuple|numpy array of function pointers, or otherwise a function pointer
            pl  : a list|tuple|numpy array of function pointers, or otherwise a function pointer
            kwargs : optional keyword arguments:

                - k    (int)  : starting index in `pk`. Required iff `pk` is a list|tuple|array
                - l    (int)  : starting index in `pl`. Required iff `pl` is a list|tuple|array
                - kmxp (int)  : (default: k+1) max-k plus 1, so that range(k,kmxp) goes over `pk[k], ..., pk[kmxp-1]`
                - lmxp (int)  : (default: l+1) max-l plus 1, so that range(l,lmxp) goes over `pl[l], ..., pl[lmxp-1]`
                - verbose_warning (bool) : controls verbosity of warnings.

        Returns:
            fklT (np.ndarray)   : 2d float numpy array with `kmxp-k` rows and `lmxp-l` columns

        Example:
            Say, we have `pk` list and `pl` single function

                - ``pk = [lambda x : 2*x, lambda x : x**2, lambda x : x**3]``
                - ``pl = lambda x : 10*x``
            then

                - ``iprod(pk[3],pl)`` returns: ``[[<Lmap.eval(pk[3]),Lmap.eval(pl)>]]``, a 2d `(1, 1)` numpy array and
                - ``iprod(pk,pl,k=1,kmxp=3)`` returns ``[[<Lmap.eval(pk[1]),Lmap.eval(pl)>, <Lmap.eval(pk[2]),Lmap.eval(pl)>]]``, a 2d `(1, 2)` numpy array.
                - ``iprod(pk,pl,k=0,kmxp=3,l=0,lmxp=2)`` returns  ``[ [<Lmap.eval(pk[0]),Lmap.eval(pl[0])>, <Lmap.eval(pk[0]),Lmap.eval(pl[1])>], [<Lmap.eval(pk[1]),Lmap.eval(pl[0])>, <Lmap.eval(pk[1]),Lmap.eval(pl[1])>], [<Lmap.eval(pk[2]),Lmap.eval(pl[0])>, <Lmap.eval(pk[2]),Lmap.eval(pl[1])>]]``  a 2d `(3, 2)` numpy array.

            NB. if x is an `npt`-long vector of data points, then for any of the above functions, say ``pk[2]``, ``Lmap.eval(pk[2])`` will return a 2d `(1, npt)` numpy array.

        '''

        k    = kwargs.get('k')
        kmxp = kwargs.get('kmxp')
        l    = kwargs.get('l')
        lmxp = kwargs.get('lmxp')
        verbose_warning = kwargs.get('verbose_warning',False)

        if isinstance(pk,(list,tuple,np.ndarray)):
            if k is None: sys.exit('Need k spec for this pk')            
            if kmxp is None: kmxp = k + 1
            fk = np.array([self.Lmap.eval(pk[ki]) for ki in range(k,kmxp)])
        elif callable(pk):
            if any(v is None for v in [k,kmxp]) and verbose_warning:
                print('Warning: no use for k|kmxp for this pk')            
            fk = self.Lmap.eval(pk).reshape(1,-1)
        else:
            sys.exit('Unexpected input pk')

        if isinstance(pl,(list,tuple,np.ndarray)):
            if l is None: sys.exit('Need l spec for this pl')            
            if lmxp is None: lmxp = l + 1
            fl = np.array([self.Lmap.eval(pl[li]) for li in range(l,lmxp)])
        elif callable(pl):
            if any(v is None for v in [l,lmxp]) and verbose_warning:
                print('Warning: no use for l|lmxp for this pl')            
            fl = self.Lmap.eval(pl).reshape(1,-1)
        else:
            sys.exit('Unexpected input pl')

        # for x containing npt data points (whether each is a scalar or a vector is immaterial), 
        # then fk is a 2d numpy array with kmxp-k rows and npt columns
        # and fl is a 2d numpy array with lmxp-l rows and npt columns
        # fklT is a 2d numpy array with kmxp-k rows and lmxp-l columns

        fklT = np.matmul(fk,fl.T)

        return fklT


    def bld_psi(self,i,Rinv):
        r'''Build and return \psi function for index `i`
            
        Args:
            i (int)             : row index
            Rinv (np.ndarray)   : 2d float matrix
        Returns:
            lfnc (function)     : :math:`\psi` function for index `i`
        '''
        def lfnc(x):
            arr = np.array([Rinv[i,j]*self.phi[j](x) for j in range(i+1)])
            sf  = np.sum(arr,axis=0)
            return sf
        return lfnc

    def bld_tht(self,i,laml):
        r'''Build and return tht (:math:`\theta`) function for index `i`
            
        Args:
            i (int)             : row index
            laml (float)        : float specified scale factor to normalize :math:`\psi[i]`
        Returns:
            lfnc (function)     : tht function for index `i`
        '''
        def lfnc(x):
            return laml * self.psi[i](x) 
        return lfnc

    def ortho(self,modified=False,verbose=False,stage=0):
        '''Orthonormalize phi functions to provide the tht functions            

        Args:
            modified (bool)     : controls whether using modified Gram-Schmidt, or unmodified
            verbose (bool)      : controls verbosity
            stage (int)         : stage index within multistage Gram-Schmidt

        Returns:
            Pmat (np.ndarray)   : 2d float projection matrix 
            tht (np.ndarray)    : 1d array of tht function pointers
        '''

        if verbose:
            print('ortho: modified =',modified)

        if verbose:
            fname = 'gso_gs'+str(stage)+'.txt'
            if modified:
                fname = 'gso_gsmod'+str(stage)+'.txt'
            with open(fname, 'w') as f:
                f.write('ortho\n')

        t0 = time.time()

        self.modified = modified

        if self.modified:
            self.V = MGSV(self,)

        self.Z = np.zeros((self.m,self.m))
        self.lam = np.zeros(self.m)

        for i in range(self.m):

            self.Z[i,i] = 1.0

            if i > 0:
                rgam = self.iprod(self.phi,self.tht,k=i,l=0,kmxp=i+1,lmxp=i)[0] 
                if np.all(self.lam[0:i]):
                    self.Z[i,0:i] = np.multiply(rgam,self.lam[0:i])
                else:
                    print('ortho: i:',i,', lam[0:i]:',self.lam[0:i],' has at least one zero')
                    sys.exit('Aborting')

            Zi = self.Z[0:i+1,0:i+1]

            if verbose:
                with open(fname, 'a') as f:
                    f.write('i:'+str(i)+', Zi:\n')
                    np.savetxt(f, Zi, fmt='%.18e', delimiter=' ')

            if self.modified:
                # fill in row i in V matrix
                self.V.fill_row(i)
                Vi = self.V.mat[0:i+1,0:i+1]
                Li = np.diag(self.lam[0:i+1])
                VLi = np.matmul(Vi,Li)
                R = Zi - VLi
                if verbose:
                    with open(fname, 'a') as f:
                        f.write('i:'+str(i)+', R:\n')
                        np.savetxt(f, R, fmt='%.18e', delimiter=' ')
            else:
                R = Zi

            # invert Z matrix, of dimension i+1 x i+1
            Rinv = np.linalg.inv(R)
            if verbose:
                with open(fname, 'a') as f:
                    f.write('i:'+str(i)+', Rinv:\n')
                    np.savetxt(f, Rinv, fmt='%.18e', delimiter=' ')

            # Evaluate Psi = Rinv * Phi, for row i
            self.psi[i] = self.bld_psi(i,Rinv)
            if verbose:
                print('i,psi[i]:',i,self.psi[i],type(self.psi[i]))

            # fill i-th entry of Lambda matrix diagonal
            self.lam[i] = 1.0/np.sqrt(self.iprod(self.psi[i],self.psi[i])[0][0])

            if verbose:
                with open(fname, 'a') as f:
                    f.write('i:'+str(i)+', lam:\n')
                    np.savetxt(f, self.lam[0:i+1], fmt='%.18e', delimiter=' ')

            # Evaluate Theta = Lambda * Psi, for row i
            self.tht[i] = self.bld_tht(i,self.lam[i])

            if verbose:
                t1 = time.time()
                print('i:',i,'dt:','%10.3e'%(t1-t0),'sec')
                with np.printoptions(precision=3,suppress=True):
                    print('Rinv:\n',Rinv)

            if verbose:
                with open(fname, 'a') as f:
                    f.write('i:'+str(i)+', Pmat:\n')
                    np.savetxt(f, np.matmul(np.diag(self.lam[0:i+1]),Rinv), fmt='%.18e', delimiter=' ')

        self.Pmat = np.matmul(np.diag(self.lam),Rinv)

        if verbose:
            with np.printoptions(precision=6,suppress=False):
                print('P:\n',self.Pmat)    

        return self.Pmat, self.tht

    def ortho_check_phi(self,):
        '''Check orthonormality of phi functions

        Returns:
            ipmat (np.ndarray)  : float 2d array containing orthonormality check matrix output
        '''
        ipmat = np.zeros((self.m,self.m))
        for i in range(self.m):
            ipmat[i,i:self.m] = self.iprod(self.phi,self.phi,k=i,l=i,kmxp=i+1,lmxp=self.m)[0] 
        return ipmat

    def ortho_check(self,):
        '''Check orthogonality of tht functions

        Returns:
            ipmat (np.ndarray)  : float 2d array containing orthonormality check matrix output
        '''
        ipmat = np.zeros((self.m,self.m))
        for i in range(self.m):
            ipmat[i,i:self.m] = self.iprod(self.tht,self.tht,k=i,l=i,kmxp=i+1,lmxp=self.m)[0] 
        return ipmat

class MMGS:
    r'''
    Multistage Modified Gram-Schmidt (MMGS) class.
    Builds MMGS object for functions.

    Attributes:
        m (int)             : number of functions
        phi (np.ndarray)    : 1d numpy array of starting functions
        phi_ (np.ndarray)   : 1d numpy array of starting functions
        Lmap (object)       : Linear map and data object
        Pmat (np.ndarray)   : 2d `(m, m)` numpy array, the projection matrix `P`
        mgs (np.ndarray)    : 1d `(nstage,)` array of MGS objects
        phia (np.ndarray)   : 2d `(nstage, m)` numpy array of starting functions
        thta (np.ndarray)   : 2d `(nstage, m)` numpy array of orthonormalized functions
        Parr (np.ndarray)   : 3d `(nstage, m, m)` matrix, holds nstage P matrices
    ''' 
    def __init__(self,phi_,Lmap):
        '''Initialize.
        Args:
            phi_  : numpy array of starting functions
            Lmap  : Linear map and data object
        '''
        self.phi_   = copy.deepcopy(phi_)
        self.m      = self.phi_.shape[0]
        self.Lmap   = copy.deepcopy(Lmap)
        return

    def ortho(self,modified=False,nstage=1,verbose=False):
        r'''Multiscale Modified Gram-Schmidt class orthonormalization
        Runs multistage and/or Modified GS for functions

        Returns:
            Pmat (np.ndarray)   : float `(m,m)` aggregated projection matrix
            thta[-1]            : 1d array of final tht functions 
        '''

        nmode        = self.m
        self.mod     = modified
        self.nstage  = nstage
        self.Parr    = np.empty((nstage,nmode,nmode))
        self.phia    = np.empty((nstage,nmode),dtype=object) 
        self.thta    = np.empty((nstage,nmode),dtype=object)

        self.phia[0] = copy.deepcopy(self.phi_)
        self.Pmat    = np.eye(nmode)
        self.mgs     = np.empty(nstage,dtype=object)

        for stage in range(nstage):
            # build mgs object 
            self.mgs[stage]   = MGS(self.phia[stage],self.Lmap)
            if stage == 0:
                ipmat_phi = self.mgs[stage].ortho_check_phi()
                if verbose:
                    print('ortho check phi_ maxabs:',np.max(np.abs(ipmat_phi-np.eye(self.m))))

            # orthonormalize with MGS
            self.mgs[stage].ortho(modified=modified,verbose=verbose,stage=stage)

            if False:
                with open('gso_P'+str(stage)+'.txt', 'w') as f:
                    f.write('gso P'+str(stage)+':\n')
                    np.savetxt(f, self.mgs[stage].Pmat, fmt='%.18e', delimiter=' ')
                xt = np.linspace(0,1,100)
                phixt = np.array([f(xt) for f in self.mgs[stage].phi]).T
                thtxt = np.array([f(xt) for f in self.mgs[stage].tht]).T
                with open('gso_phi'+str(stage)+'.txt', 'w') as f:
                    f.write('gso phixt'+str(stage)+':\n')
                    np.savetxt(f, phixt, fmt='%.18e', delimiter=' ')
                with open('gso_tht'+str(stage)+'.txt', 'w') as f:
                    f.write('gso thtxt'+str(stage)+':\n')
                    np.savetxt(f, thtxt, fmt='%.18e', delimiter=' ')

            self.thta[stage] = copy.deepcopy(self.mgs[stage].tht)

            if False:
                def bld_Theta(P,k,phiv):
                    def lfnc(x):
                        Phix = np.array([ph(x) for ph in phiv])
                        arr  = np.sum(np.broadcast_to(P[k], Phix.T.shape).T * Phix,axis=0)
                        return arr
                    return lfnc
                self.thta[stage] = np.array([bld_Theta(self.mgs[stage].Pmat,k,self.phia[stage]) for k in range(self.m)])

            # Save Pmat in Parr
            self.Parr[stage] = copy.deepcopy(self.mgs[stage].Pmat)

            # aggregate projector
            self.Pmat = np.matmul(self.Parr[stage],self.Pmat)

            # orthonormality check
            ipmat = self.mgs[stage].ortho_check()

            print('stage:',stage,', ortho check maxabs:',np.max(np.abs(ipmat-np.eye(self.m))))
            if verbose:
                print('P-stage:\n',self.Parr[stage])
                print('ortho_check:\n',ipmat)
                print('Pmat:\n',self.Pmat)

            if stage < nstage-1:
                # update phi
                self.phia[stage+1] = copy.deepcopy(self.mgs[stage].tht)    

        return self.Pmat, self.thta[stage]

    def ortho_check(self,stage=None):
        '''Check orthonormality of tht functions at given stage

        Args:
            stage (int)         : stage within multistage MGS

        Returns:
            ipmat (np.ndarray)  : float 2d array containing orthonormality check matrix output for this stage
        '''
        if stage is None:
            stage = self.nstage-1
        return self.mgs[stage].ortho_check()

    def ortho_check_phi(self,stage=None):
        '''Check orthonormality of phi functions at given stage

        Args:
            stage (int)         : stage within multistage MGS

        Returns:
            ipmat (np.ndarray)  : float 2d array containing orthonormality check matrix output for this stage
        '''
        if stage is None:
            stage = self.nstage-1
        return self.mgs[stage].ortho_check_phi()


class QR:
    r'''QR decomposition class. Builds QR object for functions.

    Attributes:
        m   (int)           : number of basis functions in expansion.
        phi (np.ndarray)    : 1d numpy array (size :math:`m`) of starting functions.
        tht (np.ndarray)    : 1d numpy array (size :math:`m`) of to-be-constructed orthonormal functions.
        Lmap (object)       : pointer to local copy of Linear map and data object.
        Pmat (np.ndarray)   : 2d :math:`(m, m)` numpy array ... the P projection matrix
    '''

    def __init__(self,phi_,Lmap):
        '''Initialization.

        Args:
            phi_ (np.ndarray): input numpy array of pointers to starting functions.
            Lmap (object)    : input pointer to Linear map and data object.
        '''
        self.phi    = copy.deepcopy(phi_)
        self.tht    = np.full_like(self.phi,None,dtype=object)
        self.m      = self.phi.shape[0]
        self.Lmap   = copy.deepcopy(Lmap)
        return

    def iprod(self, pk, pl, **kwargs):
        r'''Pairwise inner product between (lists of) functions.

        Args:
            pk  : a list|tuple|numpy array of function pointers, or otherwise a function pointer
            pl  : a list|tuple|numpy array of function pointers, or otherwise a function pointer
            kwargs : optional keyword arguments:

                - k    (int)  : starting index in `pk`. Required iff `pk` is a list|tuple|array
                - l    (int)  : starting index in `pl`. Required iff `pl` is a list|tuple|array
                - kmxp (int)  : (default: k+1) max-k plus 1, so that range(k,kmxp) goes over `pk[k], ..., pk[kmxp-1]`
                - lmxp (int)  : (default: l+1) max-l plus 1, so that range(l,lmxp) goes over `pl[l], ..., pl[lmxp-1]`
                - verbose_warning (bool) : controls verbosity of warnings.

        Returns:
            fklT (np.ndarray)   : 2d float numpy array with `kmxp-k` rows and `lmxp-l` columns

        Example:
            Say, we have `pk` list and `pl` single function

                - ``pk = [lambda x : 2*x, lambda x : x**2, lambda x : x**3]``
                - ``pl = lambda x : 10*x``
            then

                - ``iprod(pk[3],pl)`` returns: ``[[<Lmap.eval(pk[3]),Lmap.eval(pl)>]]``, a 2d `(1, 1)` numpy array and
                - ``iprod(pk,pl,k=1,kmxp=3)`` returns ``[[<Lmap.eval(pk[1]),Lmap.eval(pl)>, <Lmap.eval(pk[2]),Lmap.eval(pl)>]]``, a 2d `(1, 2)` numpy array.
                - ``iprod(pk,pl,k=0,kmxp=3,l=0,lmxp=2)`` returns  ``[ [<Lmap.eval(pk[0]),Lmap.eval(pl[0])>, <Lmap.eval(pk[0]),Lmap.eval(pl[1])>], [<Lmap.eval(pk[1]),Lmap.eval(pl[0])>, <Lmap.eval(pk[1]),Lmap.eval(pl[1])>], [<Lmap.eval(pk[2]),Lmap.eval(pl[0])>, <Lmap.eval(pk[2]),Lmap.eval(pl[1])>]]``  a 2d `(3, 2)` numpy array.

            NB. if x is an `npt`-long vector of data points, then for any of the above functions, say ``pk[2]``, ``Lmap.eval(pk[2])`` will return a 2d `(1, npt)` numpy array.

        '''

        k    = kwargs.get('k')
        kmxp = kwargs.get('kmxp')
        l    = kwargs.get('l')
        lmxp = kwargs.get('lmxp')
        verbose_warning = kwargs.get('verbose_warning',False)

        if isinstance(pk,(list,tuple,np.ndarray)):
            if k is None: sys.exit('Need k spec for this pk')
            if kmxp is None: kmxp = k + 1
            fk = np.array([self.Lmap.eval(pk[ki]) for ki in range(k,kmxp)])
        elif callable(pk):
            if any(v is None for v in [k,kmxp]) and verbose_warning:
                print('Warning: no use for k|kmxp for this pk')
            fk = self.Lmap.eval(pk).reshape(1,-1)
        else:
            sys.exit('Unexpected input pk')

        if isinstance(pl,(list,tuple,np.ndarray)):
            if l is None: sys.exit('Need l spec for this pl')
            if lmxp is None: lmxp = l + 1
            fl = np.array([self.Lmap.eval(pl[li]) for li in range(l,lmxp)])
        elif callable(pl):
            if any(v is None for v in [l,lmxp]) and verbose_warning:
                print('Warning: no use for l|lmxp for this pl')
            fl = self.Lmap.eval(pl).reshape(1,-1)
        else:
            sys.exit('Unexpected input pl')

        # for x containing npt data points (whether each is a scalar or a vector is immaterial),
        # then fk is a 2d numpy array with kmxp-k rows and npt columns
        # and fl is a 2d numpy array with lmxp-l rows and npt columns
        # fklT is a 2d numpy array with kmxp-k rows and lmxp-l columns

        fklT = np.matmul(fk,fl.T)

        return fklT

    def bld_tht(self,P,i,phiv):
        r'''Build and return tht (:math:`\theta`) function for index `i`

        Args:
            i (int)             : row index
            P (p.ndarray)       : 2d float projection matrix
        Returns:
            lfnc (function)     : tht function for index `i`
        '''
        def lfnc(x):
            Phix = np.array([ph(x) for ph in phiv])
            arr  = np.sum(np.broadcast_to(P[i], Phix.T.shape).T * Phix,axis=0)
            return arr
        return lfnc

    def ortho(self,verbose=False):
        '''Orthonormalize phi functions to provide the tht functions

        Args:
            verbose (bool)      : controls verbosity

        Returns:
            Pmat (np.ndarray)   : 2d float projection matrix
            tht (np.ndarray)    : 1d array of tht function pointers

        Uses QR factorization to find Pmat
        '''

        if verbose:
            fname = 'qro_qr.txt'
            with open(fname, 'w') as f:
                f.write('ortho QR\n')

        Aphi      = np.array([self.Lmap.eval(pf) for pf in self.phi]).T
        Q, R      = np.linalg.qr(Aphi)

        if R.shape[0] != R.shape[1] :
            print('Failure in orf: QR.ortho : orthonormalization via QR decomposition')
            print('Aphi:',Aphi.shape,'Q:',Q.shape,'R:',R.shape)
            print('R is not square, and cannot be inverted')
            print('Make sure you have more data points than basis functions to avoid this failure!')
            sys.exit('1')

        Pqr       = np.linalg.inv(R)

        self.Pmat = Pqr.T
        self.tht  = np.array([self.bld_tht(self.Pmat,i,self.phi) for i in range(self.m)])

        if verbose:
            with np.printoptions(precision=6,suppress=False):
                print('P:\n',self.Pmat)

        return self.Pmat, self.tht

    def ortho_check_phi(self,):
        '''Check orthonormality of phi functions

        Returns:
            ipmat (np.ndarray)  : float 2d array containing orthonormality check matrix output
        '''
        ipmat = np.zeros((self.m,self.m))
        for i in range(self.m):
            ipmat[i,i:self.m] = self.iprod(self.phi,self.phi,k=i,l=i,kmxp=i+1,lmxp=self.m)[0]
        return ipmat

    def ortho_check(self,):
        '''Check orthogonality of tht functions

        Returns:
            ipmat (np.ndarray)  : float 2d array containing orthonormality check matrix output
        '''
        ipmat = np.zeros((self.m,self.m))
        for i in range(self.m):
            ipmat[i,i:self.m] = self.iprod(self.tht,self.tht,k=i,l=i,kmxp=i+1,lmxp=self.m)[0]
        return ipmat


