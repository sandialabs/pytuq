#!/usr/bin/env python
"""Example demonstrating Gaussian Mixture Model (GMM) sampling and visualization.

This script creates a GMM with multiple components, samples from it within
a specified domain, and visualizes the samples and probability density.
"""

import numpy as np
import matplotlib.pyplot as plt

from pytuq.rv.mrv import GMM
from pytuq.utils.plotting import plot_pdfs, plot_xrv

means = np.array([[2.0, 0.0], [12.0, 4.0]])
ncl, dim = means.shape
weights = [4., 2.]
mygmm = GMM(means, weights=weights)

domain = np.array([[-20., 30.], [-10., 20.]])
sam = mygmm.sample_indomain(11111, domain)
#sam = mygmm.sample(333)
np.savetxt('sam.txt', sam)
plot_xrv(sam)

plot_pdfs(ind_show=[0, 1], samples_=sam, plot_type='tri',
          names_=None, burnin=0, every=1, lsize=11)
plt.clf()

# a, b = -2., 3.
# x = np.linspace(-10, 10, 1000)
# # x = np.linspace(truncnorm.ppf(0.000001, a, b),
# #                 truncnorm.ppf(0.999999, a, b), 1000)
# plt.plot(x, truncnorm.cdf(x, a, b, loc=0, scale=1), 'k-', lw=2)
# plt.savefig('cdf.png')
# plt.clf()


# dist = multivariate_normal(mean=means[0], cov=np.eye(2)) # use covariance above, too, otherwise bugs will creep into this
# print("CDF:", dist.cdf(np.array([[2, 0.0], [3, 2]])))

pdf_integral = mygmm.volume_indomain(domain)
assert(np.isclose(pdf_integral, 1.0))

