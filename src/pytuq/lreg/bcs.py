#!/usr/bin/env python
"""A module for Bayesian compressive sensing."""

import sys
import numpy as np

from .lreg import lreg



class bcs(lreg):
    """Bayesian compressive sensing (BCS) class.

    Attributes:
        eta (float): The tolerance parameter of the BCS algorithm.
        datavar_init (float): Initial value of the data variance.
        fitted (bool): Indicates whether the fit is done or note.
        used (np.ndarray): Indices of retained bases.
    """

    def __init__(self, eta=1.e-8, datavar_init=None):
        """Initialization.

        Args:
            eta (float, optional): The tolerance parameter of the BCS algorithm.
            datavar_init (float, optional): Initial value of the data variance to start the algorithm with. If None, picks a rule-of-thumb default value.

        Note:
            This is the conventional (not weighted) BCS. The weighted BCS is implemented in UQTk.
        """
        super().__init__()
        self.eta = eta
        self.used = None
        self.datavar_init = datavar_init


    def fita(self, Amat, y):
        r"""Fit given A-matrix of basis evaluations and data array.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            y (np.ndarray): An 1d array of size :math:`N` holding the data.
        """

        npt, nbas = Amat.shape

        self.cf, errbars, self.used, self.datavar, basis, self.cf_cov = bcs_fit(Amat, y, sigma2=self.datavar_init, eta=self.eta)


        self.fitted = True

        return

######################################################################
######################################################################
######################################################################

def bcs_fit(A, y, sigma2=None, eta=1.e-8, adaptive=0, optimal=1, scale=0.1, nugget=1.e-6):
    r"""BCS fitting algorithm. See "Bayesian Compressive Sesning" (Preprint, 2007). The algorithm
    adopts from the fast RVM algorithm, https://www.miketipping.com/papers/met-fastsbl.pdf [Tipping & Faul, 2003]. Original code in Matlab by Shihao Ji, ECE, Duke University.


    Args:
        A (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
        y (np.ndarray): An 1d array of size :math:`N` holding the data.
        sigma2 (None, optional): Initial data variance. Defaults to None, which picks a rule-of-thumb default value.
        eta (float, optional): The tolerance parameter of the BCS algorithm. Defaults to 1.e-6.
        adaptive (int, optional): Integer-Boolean flag whether to turn on the adaptive algorithm or not. Defaults to 0.
        optimal (int, optional): Integer-Boolean flag whether to use the rigorous implementation of adaptive algorithm. Defaults to 1.
        scale (float, optional): Diagonal loading parameter of adaptive algorithm. Defaults to 0.1.
        nugget (float, optional): Small diagonal nugget to improve the conditioning of matrix :math:`\Sigma`.

    Returns:
        tuple(weights, errbars, used, sigma2, basis, Sig): Tuple of sparse coefficients (weights), one st.dev. errorbars on coefficients, used indices, re-estimated noise variance, the next estimated basis vector (if adaptive method), and covariance matrix (Sig).
    """


    if sigma2 is None:
        sigma2 = max(np.std(y)**2/1.e+2, nugget)

    # find initial alpha
    N, M = A.shape
    Aty = np.dot(A.T, y)  # (M,)
    A2 = np.sum(A**2, axis=0) # (M,)
    ratio = (Aty**2 + nugget * np.ones_like(Aty)) / (A2 + nugget * np.ones_like(A2)) # (M,)


    index = [np.argmax(ratio)] # vector of dynamic size K with values = 0..M-1
    maxr = ratio[index] # (K,)

    alpha = A2[index] / (maxr - sigma2) # (K,)

    # compute initial mu, Sig, S, Q
    Asl = A[:, index] # (N,K)
    Hessian = alpha + np.dot(Asl.T, Asl) / sigma2 # (K,K)
    Sig = 1. / (Hessian + nugget * np.ones_like(Hessian)) # (K,K)
    mu = np.zeros((1,))
    mu[0] = Sig[0,0] * Aty[index[0]] / sigma2 # (K,)

    left = np.dot(A.T, Asl) / sigma2 # (M,K)
    S = A2 / sigma2 - Sig[0, 0] * left[:, 0]**2 # (M,)
    Q = Aty / sigma2 - Sig[0, 0] * Aty[index[0]] / sigma2 * left[:, 0] # (M,)


    itermax = 100
    mlhist = np.empty(itermax)

    for count in range(itermax):  # careful with index below
        ss = S.copy()
        qq = Q.copy()
        #print(ss.shape, alpha.shape, S.shape, index, A2.shape, maxr.shape, sigma2, ratio.shape, Aty.shape)
        ss[index] = alpha * S[index] / (alpha - S[index])
        qq[index] = alpha * Q[index] / (alpha - S[index])
        theta = qq**2 - ss # (M,)

        # choose the next alpha that maximizes marginal likelihood
        ml = -np.inf * np.ones(M)
        ig0 = np.where(theta > 0)[0] # vector of values 0..M-1 of size L<=M

        # index for re-estimate ire=ig0[foo]=index[which]
        ire, foo, which = np.intersect1d(ig0, index, return_indices=True)
        if len(ire) > 0:
            alpha_ = ss[ire]**2 / theta[ire] + nugget

            delta = (alpha[which] - alpha_) / (alpha_ * alpha[which])
            if (1 + S[ire] * delta<=0.0).any():
                ml[ire] = -1.e+80
            else:
                ml[ire] = Q[ire]**2 * delta / (S[ire] * delta + 1) - np.log(1 + S[ire] * delta)

        # index for adding
        iad = np.setdiff1d(ig0, ire)
        if len(iad) > 0:
            if (S[iad]<=0.0).any():
                ml[iad] = -1.e+80
            else:
                ml[iad] = (Q[iad]**2 - S[iad]) / S[iad] + np.log(S[iad] / (Q[iad]**2))

        is0 = np.setdiff1d(np.arange(M), ig0)

        # index for deleting
        ide, foo, which = np.intersect1d(is0, index, return_indices=True)
        if len(ide) > 0:
            ml[ide] = Q[ide]**2 / (S[ide] - alpha[which]) - np.log(1. - S[ide] / alpha[which])


        idx = np.argmax(ml)  #TODO check single value?
        mlhist[count] = ml[idx]

        # check if terminates?
        if count > 1 and \
           abs(mlhist[count] - mlhist[count - 1]) < abs(mlhist[count] - mlhist[0]) * eta:
            break

        # update alphas
        which = np.where(index == idx)[0] # TODO assert length 1?
        if theta[idx] > 0:
            if len(which) > 0:            # re-estimate
                alpha_ = ss[idx]**2 / theta[idx] + nugget
                Sigii = Sig[which[0], which[0]]
                mui = mu[which[0]]
                Sigi = Sig[:, which[0]]  # (K,)

                delta = alpha_ - alpha[which[0]]
                ki = delta / (1. + Sigii * delta)
                comm = np.dot(A.T, np.dot(Asl, Sigi) / sigma2)  # (M,)
                mu = mu - ki * mui * Sigi  # (K,)
                Sig = Sig - ki * np.dot(Sigi.reshape(-1, 1), Sigi.reshape(1, -1))
                comm = np.dot(A.T, np.dot(Asl, Sigi) / sigma2)  # (M,)

                S = S + ki * comm**2 # (M,)
                Q = Q + ki * mui * comm # (M,)

                alpha[which] = alpha_
            else:            # adding
                alpha_ = ss[idx]**2 / theta[idx] + nugget
                Ai = A[:, idx]  # (N,)
                Sigii = 1. / (alpha_ + S[idx])
                mui = Sigii * Q[idx]

                comm1 = np.dot(Sig, np.dot(Asl.T, Ai)) / sigma2 # (K,)

                ei = Ai - np.dot(Asl, comm1) # (N,)
                off = -Sigii * comm1 #( K,)
                Sig = np.block([[
                               Sig + Sigii * np.dot(comm1.reshape(-1, 1),
                                                    comm1.reshape(1, -1)),
                               off.reshape(-1, 1)],
                               [off.reshape(1, -1),
                               Sigii]])

                mu = np.append(mu - mui * comm1, mui)
                comm2 = np.dot(A.T, ei) / sigma2 #(M,)
                S = S - Sigii * comm2**2
                Q = Q - mui * comm2
                #
                index = np.append(index, idx)
                alpha = np.append(alpha, alpha_)
                Asl = np.hstack((Asl, Ai.reshape(-1, 1))) # (N, K++)

        else:
            if len(which) > 0 and len(index) > 1:            # deleting
                Sigii = Sig[which[0], which[0]]
                mui = mu[which[0]]
                Sigi = Sig[:, which[0]] # (K,)

                Sig -= np.dot(Sigi.reshape(-1, 1), Sigi.reshape(1, -1)) / Sigii
                Sig = np.delete(Sig, which[0], 0)
                Sig = np.delete(Sig, which[0], 1)

                mu = mu - (mui / Sigii) * Sigi # (K,)
                mu = np.delete(mu, which[0])
                comm = np.dot(A.T, np.dot(Asl, Sigi)) / sigma2 # (M,)
                S = S + (comm**2 / Sigii) # (M,)
                Q = Q + (mui / Sigii) * comm # (M,)

                #
                index = np.delete(index, which[0])
                alpha = np.delete(alpha, which[0])
                Asl = np.delete(Asl, which[0], 1)
            if len(which) > 0 and len(index) == 1:
                break

    #print("MLHIST ", mlhist)

    weights = mu
    used = index
    # re-estimated sigma2
    sigma2 = np.sum((y - np.dot(Asl, mu))**2) / (N - len(index) +
                                                 np.dot(alpha.reshape(1, -1),
                                                        np.diag(Sig)))

    detsig = np.linalg.det(Sig)
    if detsig<-nugget:
        print(f"Warning: Sigma matrix has a determinant {detsig} that is not positive. Use with care.")

    # assert((np.diag(Sig)>0.0).all())
    # if (np.diag(Sig)<0.0).any():
    #     print(f"Error: Sigma matrix has a negative diagonal element. Something is wrong. Exiting.")
    #     sys.exit()

    if (np.diag(Sig)<0.0).any():
        print(f"Warning: Sigma matrix has a negative diagonal element. Setting them to zero, but this may lead to inaccuracies.")
        for i in range(Sig.shape[0]):
            if Sig[i,i]<0.0:
                Sig[i,i]=0.0


    errbars = np.sqrt(np.diag(Sig))

    # generate a basis for adaptive CS?
    basis = None
    if adaptive:
        if optimal:
            D, V = np.linalg.eig(Sig)
            idx = np.argmax(D) # TODO is it a single number?
            basis = V[:, idx]
        else:
            temp = np.dot(Asl.T, Asl) / sigma2
            Sig_inv = temp + scale * np.mean(np.diag(temp)) * np.eye(len(used))
            D, V = np.linalg.eig(Sig_inv)
            idx = np.argmin(D)
            basis = V[:, idx]

    return weights, errbars, used, sigma2, basis, Sig # KS: the last Sig was recently added



