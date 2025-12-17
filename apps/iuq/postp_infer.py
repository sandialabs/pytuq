#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


import pytuq.gsa.gsa as gsa
import pytuq.minf.minf as minf
import pytuq.utils.plotting as pup
from pytuq.utils.xutils import loadpk, read_textlist


pup.myrc()



# Parse input arguments
usage_str = 'Workflow to postprocess the inference workflow.'
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=usage_str)
parser.add_argument("-c", "--cfile", dest="par_true_file", type=str, default=None, help="True parameters' file name, if the true parameters are known.")

args = parser.parse_args()


par_true_file=args.par_true_file


###############################################################
###############################################################
###############################################################

# Load surrogate results
results = loadpk('results') # really need to do LU

pcrv = results['pcrv']
ptrain = results['training'][1]
ytrain = results['training'][2]
surr_error_var = np.loadtxt('surr_error_var.txt')


# Load data
ydata = np.loadtxt('ydata.txt')
if len(ydata.shape)==1:
    ydata = ydata.reshape(-1, 1)
ydatavar = np.loadtxt('ydatavar.txt')
nx = ydata.shape[0]
xdata = np.arange(nx)

# Optionally, load true parameter values
p_true = np.loadtxt(par_true_file) if par_true_file is not None else np.random.rand(pcrv.function.dim)

pdim = pcrv.function.dim
nx_ = pcrv.function.outdim
nx__ = ydata.shape[0]
assert(nx_==nx)
assert(nx__==nx)

pnames = read_textlist('pnames.txt', pdim, names_prefix='Par')
outnames = read_textlist('outnames.txt', nx, names_prefix='Out')

print(f"Number of inputs  : {pcrv.function.dim}")
print(f"Number of outputs : {pcrv.function.outdim}")

###############################################################
###############################################################
###############################################################

## Load calibration results
calib_results = loadpk('calib_results')

## Postprocess model at grid points for plotting
checkmode = [calib_results['post'].model, calib_results['post'].model_params, range(nx), None]
ycheck_pred, samples =  minf.model_infer_postp(calib_results,
                                               checkmode=checkmode,
                                               nburn=len(calib_results['chain'])//10,
                                               nevery=10, nxi=10)

psamples, fsamples = samples['pmcmc'], samples['fmcmc']
psample_map, fsample_map = samples['pmap'], samples['fmap']


#############################################################
#############################################################
#############################################################

## Plot the chains
print("Plotting chains")

nchain, chdim = calib_results['chain'].shape

# Chain diagnostic
fig, axes = plt.subplots(chdim, 1, figsize=(10,4*chdim),
                             gridspec_kw={'hspace': 0.2, 'wspace': 0.0})

for idim in range(chdim):
    axes[idim].plot(calib_results['chain'][:, idim], '-', lw=1)
    axes[idim].set_ylabel(rf'c$_{idim}$')
axes[-1].set_xlabel('MCMC Step')
plt.savefig(f'chain.png')

#############################################################
#############################################################

# ## More plots to visualize fits
# minf.plot_1dfit(xdata, ycheck_pred, ygrid_true=None, xydata=(xdata,ydata))
# minf.plot_1dfit_vars(xdata, ycheck_pred, xydata=(xdata,ydata))
# minf.plot_1dfit_shade(xdata, samples['fmcmc'][:,0,:].T, xydata=(xdata,ydata)) #TODO update this with model error
# # minf.plot_ndfit_vars(ycheck, ycheck_pred)

#############################################################
#############################################################
## Plot triangular PDFs of inputs and output posteriors
print("Plot triangular PDFs of inputs and output posteriors")

pup.plot_pdfs(ind_show=None, samples_=psamples.reshape(-1, pdim),
              plot_type='tri', pdf_type='kde',
              burnin=0, every=1,
              names_=pnames, prange_=None,
              nominal_=p_true,
              lsize=15, zsize=15)
os.system('mv pdf_tri.png pdf_tri_post_inputs.png')

pup.plot_pdfs(ind_show=None, samples_=fsamples.reshape(-1, nx),
              plot_type='tri', pdf_type='kde',
              burnin=0, every=1,
              names_=outnames, prange_=None,
              nominal_=ydata,
              lsize=15, zsize=15)
os.system('mv pdf_tri.png pdf_tri_post_outputs.png')

#############################################################
#############################################################

## Plot parameter prior and posterior PDFs
print("Plot parameter prior and posterior PDFs")

psamples_ = psamples.reshape(-1, pdim)

# Plot parameter posterior PDF and the associated samples
fig, axarr = plt.subplots(nrows=pdim, ncols=pdim, figsize=(15, 15))
for i in range(pdim):
    thisax = axarr[i][i]
    #thisax.hist(ptrain[:, i], color='b', alpha=0.7)
    #thisax.hist(psamples_phys[:, 0, i], color='m', alpha=0.7)
    pup.plot_pdf1d(ptrain[:, i], pltype='kde', color='b', lw=1.0,
                   histalpha=0.7, label='Prior', ax=thisax)
    pup.plot_pdf1d(psamples_[:, i], pltype='kde', color='m', lw=1.0,
                   histalpha=0.7, label='Posterior', ax=thisax)

    if i == 0:
        thisax.set_ylabel(pnames[i])
    if i == pdim - 1:
        thisax.set_xlabel(pnames[i])
    if i > 0:
        thisax.yaxis.set_ticks_position("right")
    # thisax.yaxis.set_label_coords(-0.12, 0.5)

    for j in range(i):
        thisax = axarr[i][j]
        axarr[j][i].axis('off')

        thisax.plot(ptrain[:, j], ptrain[:, i], 'bo')
        thisax.plot(psamples_[:, j], psamples_[:, i], 'md')
        pup.plot_pdf2d(ptrain[:, j], ptrain[:, i], pltype='kde',
                       ncont=10, color='b', lwidth=1.0, mstyle='o', ax=thisax)
        pup.plot_pdf2d(psamples_[:, j], psamples_[:, i], pltype='kde',
                       ncont=10, color='m', lwidth=1.0, mstyle='o', ax=thisax)

        # x0, x1 = thisax.get_xlim()
        # y0, y1 = thisax.get_ylim()
        # #thisax.set_aspect((x1 - x0) / (y1 - y0))

        if j == 0:
            thisax.set_ylabel(pnames[i])
        if i == pdim - 1:
            thisax.set_xlabel(pnames[j])
        if j > 0:
            thisax.yaxis.set_ticklabels([])

plt.tight_layout()
plt.savefig('pdf_tri_prior_post.png')

#############################################################
#############################################################

## Plot output prior and posterior PDFs
print("Plot output prior and posterior PDFs")

fsamples_ = fsamples.reshape(-1, nx)

for ix in range(nx):
    plt.figure(figsize=(10, 12))
    plt.hist(ytrain[:, ix], label='Prior')
    plt.hist(fsamples_[:, ix], color='m', alpha=0.4, label='Posterior')
    plt.xlabel(f'{outnames[ix]}')
    ymin, ymax = plt.gca().get_ylim()
    plt.plot([ydata[ix]] * 2, [ymin, ymax], 'r--', label='Ref. data')
    plt.legend()
    plt.savefig(f'hist_out{ix}.png')
    plt.clf()


pup.plot_pdfs(ind_show=None, plot_type='ind', pdf_type='kde',
              samples_=ytrain, burnin=0, every=1,
              names_=outnames, nominal_=ydata, prange_=None)

pup.plot_samples_pdfs([ytrain, fsamples.reshape(-1, nx)],
                      legends=['Prior', 'Posterior'], colors=['blue', 'green'],
                      file_prefix='pdfs_prior_post', title='')


#############################################################
#############################################################

## Optional mask to show only select subset of outputs
mask = np.arange(0, nx, dtype=int) #[1, 3]

#############################################################
#############################################################

## Plot prior/posterior PDFs in a joyplot/ridge manner
print("Plot prior/posterior PDFs in a joyplot/ridge manner")

fig = plt.figure(figsize=(8, 12))
axjoy = plt.gca()
sams_prior = ytrain[:, mask]
sams_post = fsamples[:, :, mask].reshape(-1, len(mask))
domain = np.array([sams_prior.min(), sams_prior.max()])
outnames_this = [outnames[j] for j in mask]
obs_this = ydata[mask]
pup.plot_joy([sams_prior, sams_post], xdata[mask], outnames_this, [
    'pink', 'lightgreen'], nominal=ydata[mask], ax=axjoy)
plt.tight_layout()
plt.savefig('joyplot_prior_post.png')


#############################################################
#############################################################

## Plot prior/posterior ensemble
print("Plot prior/posterior ensemble")

plt.figure(figsize=(16, 8))
axens = plt.gca()
pup.plot_ens(np.arange(nx)[mask], ytrain[:, mask].T, color='r', lw=0.5, ax=axens, label='Prior')
pup.plot_ens(np.arange(nx)[mask],
             fsamples[:, :, mask].reshape(-1, len(mask)).T[:, ::1],
             color='g', lw=0.5, ax=axens, label='Posterior')
#plt.errorbar(np.arange(nx)[mask], ydata[mask], yerr=ydata_std[mask], fmt='ko')
plt.plot(np.arange(nx)[mask], ydata[mask], 'ko')
axens.set_xticks(np.arange(nx)[mask])
axens.set_xticklabels(np.array([str(j) for j in outnames])[mask], rotation=0)
h, l = axens.get_legend_handles_labels()
axens.legend([h[0],h[-1]], [l[0], l[-1]], fontsize=22, ncol=1)
# plt.xlabel('Output Index', fontsize=22)
# plt.ylabel('Output QoI Value', fontsize=22)
plt.tight_layout()
plt.savefig('ensemble_prior_post.png')

#############################################################
#############################################################

## Plot fit with variance decomposition
print("Plot fit with variance decomposition")

fsample_mean = np.mean(fsamples.reshape(-1, nx), axis=0)

fsample_vars = np.empty((fsample_mean.shape[0], 3))
fsample_vars[:, 0] = np.var(np.mean(fsamples, axis=1), axis=0)
fsample_vars[:, 1] = np.mean(np.var(fsamples, axis=1), axis=0)
fsample_vars[:, 2] = surr_error_var
varlabels = ['Posterior uncertainty', 'Model error', 'Surrogate error']

fsample_std = np.sqrt(np.sum(fsample_vars, axis=1))

plt.figure(figsize=(16, 8))
axvar = plt.gca()
pup.plot_vars(np.arange(nx)[mask], fsample_mean[mask], variances=fsample_vars[mask, :],
              ysam=None, stdfactor=1.,
              varlabels=varlabels, varcolors=None, grid_show=True,
              connected=True, interp=None, offset=(None, None), ax=axvar)

# axvar.errorbar(np.arange(nx)[mask], ydata[mask], yerr=ydata_std[mask],
#                fmt='ko', zorder=11111, label='Ref. data')
axvar.plot(np.arange(nx)[mask], ydata[mask], 'ko', zorder=11111, label='Ref. data')

axvar.set_xticks(np.arange(nx)[mask])
axvar.set_xticklabels(np.array([str(j) for j in outnames])[mask], rotation=90)
h, l = axvar.get_legend_handles_labels()
axvar.legend(h[0:5], l[0:5], fontsize=18, ncol=3)
#axvar.legend(bbox_to_anchor=(0.07, 1.02), ncol=5, fontsize=15)
plt.tight_layout()
plt.savefig('fit_variances.png')

#############################################################
#############################################################

## Plot prior/posterior with shaded quantile fit
print("Plot prior/posterior with shaded quantile fit")

plt.figure(figsize=(16, 8))
axshade = plt.gca()
pup.plot_shade(np.arange(nx)[mask], ytrain.T[mask, :],
               cmap=mpl.cm.Reds, bounds_show=False, grid_show=True, ax=axshade)
pup.plot_shade(np.arange(nx)[mask], fsamples.reshape(-1, nx).T[mask, :],
               cmap=mpl.cm.BuGn, bounds_show=False, grid_show=True, ax=axshade)

# axshade.errorbar(np.arange(nx)[mask], ydata[mask],
#                  yerr=ydata_std[mask], fmt='ko', zorder=11111, label='Ref. data')
axshade.plot(np.arange(nx)[mask], ydata[mask], 'ko', zorder=11111, label='Ref. data')
#axshade.legend(bbox_to_anchor=(0.07, 1.02), ncol=5, fontsize=15)
axshade.set_xticks(np.arange(nx)[mask])
axshade.set_xticklabels(np.array([str(j) for j in outnames])[mask], rotation=90)

handles, labels = axshade.get_legend_handles_labels()
handles.append(plt.Rectangle((0, 0), 1, 1, fc='r'))
labels.append('Model Prior')
handles.append(plt.Rectangle((0, 0), 1, 1, fc='g'))
labels.append('Model Posterior')
axshade.legend(handles, labels, fontsize=24, ncol=3, loc='upper center')
axshade.legend(handles[-3:], labels[-3:], fontsize=22, ncol=1)
plt.tight_layout()
plt.savefig('fit_shade_post_prior.png')

#############################################################
#############################################################


fig, axs = plt.subplots(2)
pup.plot_shade(np.arange(nx)[mask], fsamples.reshape(-1, nx).T[mask, :],
               cmap=mpl.cm.BuGn, bounds_show=False, grid_show=True, ax=axs[1])
# axs[1].errorbar(np.arange(nx)[mask], ydata[mask], yerr=ydata_std[mask],
#                 fmt='ko', zorder=11111, label='Ref. data')
axs[1].plot(np.arange(nx)[mask], ydata[mask], 'ko', zorder=11111, label='Ref. data')
colors = ['pink', 'lightblue', 'blue']
wd = 2./len(mask)
curr = np.zeros((len(mask)))
for i in range(fsample_vars.shape[1]):
    axs[0].bar(np.arange(nx)[mask], fsample_vars[mask, i], width=wd,
               color=colors[i], bottom=curr, label=varlabels[i], align='center')
    curr = fsample_vars[mask, i] + curr

axs[0].legend(bbox_to_anchor=(0.9, 1.2), ncol=3, fontsize=16)
axs[0].set_xticks(np.arange(nx)[mask])
axs[0].set_xticklabels([''] * len(mask))
axs[0].set_xlim(-wd - 0.1, len(mask) - 1 + wd + 0.1)
# axs[0].set_ylim(ymax=270)
axs[0].set_ylabel('QoI Variance')
# axs[0].set_yscale('log')
axs[1].set_xticks(np.arange(nx)[mask])
axs[1].set_xticklabels(np.array([str(j) for j in outnames])[mask], rotation=90)
axs[1].set_ylabel('QoI Value')
axs[1].set_xlim(-wd - 0.1, len(mask) - 1 + wd + 0.1)
plt.tight_layout()
plt.savefig('fit_shade_post_sens.png')


#############################################################
#############################################################

plt.figure(figsize=(16, 8))
axres = plt.gca()
normalize = np.abs(np.mean(ydata, axis=1))[mask]
axres.plot(np.arange(nx)[mask], np.abs(ydata-fsample_mean[:, np.newaxis])
           [mask] / normalize[:, np.newaxis], 'o-', color='b', lw=0.5, label='Bias (|Pred.Mean - Ref.|)')
axres.fill_between(np.arange(nx)[mask], np.zeros_like(fsample_std[mask]),
                   fsample_std[mask] / normalize,
                   fc='lightgrey', ec='black', lw=0.5, label='Pred. StDev', alpha=0.4)

axres.legend()
axres.set_xticks(np.arange(nx)[mask])
axres.set_xticklabels(np.array([str(j) for j in outnames])[mask], rotation=90)
axres.set_yscale('log')
axres.set_ylabel('Relative error')
h, l = axres.get_legend_handles_labels()
axres.legend(h[-2:], l[-2:], fontsize=18, ncol=2)
plt.tight_layout()
plt.savefig('fit_residual.png')
