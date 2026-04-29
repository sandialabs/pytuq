#!/usr/bin/env python
"""Plot probability density functions from MCMC or other samples.

This script reads a samples file and produces either a triangular
pair-plot or individual marginal PDF plots (histograms or KDEs).
It supports burn-in trimming, thinning, optional prior-range overlays,
and nominal-value markers.

Example::

    python plot_pdfs.py -p pchain.dat -t tri -b 1000 -e 5

Command-line arguments:
    Positional            Indices of parameters to show (default: all).
    -p, --samples_file    Samples file (default: ``pchain.dat``).
    -n, --names_file      Parameter names file.
    -l, --nominal_file    Nominal parameter values file.
    -g, --prange_file     Prior range file.
    -t, --plot_type       ``tri`` (triangular), ``ind`` (individual), or ``inds``.
    -f, --pdf_type        ``hist`` or ``kde`` (default: ``hist``).
    -b, --burnin          Number of burn-in samples to discard (default: 0).
    -e, --every           Thinning interval (default: 1).
    -x, --lsize           Axes label font size (default: 10).
    -z, --zsize           Tick label font size (default: 10).
"""

import argparse

from pytuq.utils.plotting import plot_pdfs, myrc


usage_str = 'Script to plot PDFs given samples, either triangular or individual.'
parser = argparse.ArgumentParser(description=usage_str)
parser.add_argument('ind_show', type=int, nargs='*',
                    help="indices of requested parameters (count from 0)")
parser.add_argument("-p", "--samples_file", dest="samples_file", type=str, default='pchain.dat',
                    help="Samples file")
parser.add_argument("-n", "--names_file", dest="names_file", type=str, default=None,
                    help="Names file")
parser.add_argument("-l", "--nominal_file", dest="nominal_file", type=str, default=None,
                    help="Nominals file")
parser.add_argument("-g", "--prange_file", dest="prange_file", type=str, default=None,
                    help="Prior range file")
parser.add_argument("-t", "--plot_type", dest="plot_type", type=str, default='tri',
                    help="Plot type", choices=['tri', 'ind', 'inds'])
parser.add_argument("-f", "--pdf_type", dest="pdf_type", type=str, default='hist',
                    help="Histogram or KDE", choices=['hist', 'kde', 'sam'])
parser.add_argument("-b", "--burnin", dest="burnin", type=int, default=0,
                    help="Samples burnin")
parser.add_argument("-e", "--every", dest="every", type=int, default=1,
                    help="Samples thinning")
parser.add_argument("-x", "--lsize", type=int, dest="lsize", default=10,
                    help="Axes label size")
parser.add_argument("-z", "--zsize", type=int, dest="zsize", default=10,
                    help="Tick label size")
args = parser.parse_args()

if len(args.ind_show)==0:
    ind_show=None
else:
    ind_show = args.ind_show

# TODO: plenty of beautifications and tests in plot_pdfs.py
# handle axes labels' size
# handle legends for ind plot_type
# formatting ticklabels
# subplot or add_axes afterall?


# plt.rc('axes', labelsize=lsize)
# plt.rc('xtick', labelsize=lsize)
# plt.rc('ytick', labelsize=lsize)

myrc()

_ = plot_pdfs(ind_show=ind_show, samples_=args.samples_file,
              plot_type=args.plot_type, pdf_type=args.pdf_type,
              burnin=args.burnin, every=args.every,
              names_=args.names_file, prange_=args.prange_file,
              nominal_=args.nominal_file,
              lsize=args.lsize, zsize=args.zsize)
