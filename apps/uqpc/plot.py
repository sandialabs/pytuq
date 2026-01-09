#!/usr/bin/env python

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytuq.utils.plotting import myrc, plot_dm, plot_sens, plot_jsens, plot_pdf1d, plot_sensmat, plot_vars, plot_1d, plot_2d, plot_joy
from pytuq.utils.xutils import loadpk, read_textlist


myrc()




#############################################################
#############################################################

# Parsing the inputs
usage_str = '           Reads results.pk and makes visualizations.\n\
           First argument defines the plot type. \n\
           Try "plot.py <plot_type> -h" for help on additional arguments under each plot_type. \n\
           Parameter names and output names can be read in pnames.txt and outnames.txt (if files not present, will use generic names).\n\
           Many options are quite experimental yet, and not optimal visually. \n\
           One can use this script as an example of how to unroll results.pk for subsequent plotting on one\'s own.'

parser = argparse.ArgumentParser(
    description=usage_str, formatter_class=argparse.RawTextHelpFormatter)
arg1_choices = ['sens', 'jsens', 'sensmat', 'dm', 'fit', '1d', '2d', 'pdf', 'joy']
#parser.add_argument('plot_type', type=str,nargs=1,help="Plot type", choices=arg1_choices)
subparsers = parser.add_subparsers()
for arg1 in arg1_choices:
    if arg1 == 'sens':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots sensitivity barplot for all outputs',
                                         epilog='Examples: \n"plot.py sens main", \n"plot.py sens total"')
        sbparser.add_argument('sensmode', type=str, nargs='?',
                              help="Sensitivity type", choices=['main', 'total'])
    elif arg1 == 'jsens':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots sensitivity circular plots for all outputs, and averaged as well.',
                                         epilog='Example: \n"plot.py jsens"')
    elif arg1 == 'sensmat':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots sensitivity matrix for all outputs and for the most important inputs',
                                         epilog='Examples: \n"plot.py sensmat main", \n"plot.py sensmat total"')
        sbparser.add_argument('sensmode', type=str, nargs='?',
                              help="Sensitivity type", choices=['main', 'total'])
    elif arg1 == 'dm':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots model-vs-surrogate for all outputs',
                                         epilog='Examples: \n"plot.py dm training", \n"plot.py dm testing", \n"plot.py dm training testing"')
        sbparser.add_argument('trvals', type=str, nargs='*',
                              help="Data types", choices=['training', 'testing'])

    elif arg1 == 'fit':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots model-vs-surrogate for all sample points',
                                         epilog='Examples: \n"plot.py fit training", \n"plot.py fit testing", \n"plot.py fit training testing"')
        sbparser.add_argument('trvals', type=str, nargs='*',
                              help="Data types", choices=['training', 'testing'])


    elif arg1 == '1d':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots 1d PC surrogate slices (the rest of parameters, if any, at nominal or integrated out), for all outputs. Optionally overlay training/testing points',
                                         epilog='Examples: \n"plot.py 1d 0 training"')
        sbparser.add_argument('trvals', type=str, nargs='*', help="Data type",
                              choices=['training', 'testing'])
    elif arg1 == '2d':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots 2d PC surrogate contours (the rest of parameters, if any, at nominal), for all outputs and for all input pairs.',
                                         epilog='Examples: \n"plot.py 2d"')
    elif arg1 == 'pdf':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots the PDF of the output. Sampling parameter is hardwired.',
                                         epilog='Example: \n"plot.py pdf"')

    elif arg1 == 'joy':
        sbparser = subparsers.add_parser(arg1, formatter_class=argparse.RawTextHelpFormatter,
                                         description='Plots the PDF of the output with a joyplot. Sampling parameter is hardwired.',
                                         epilog='Example: \n"plot.py joy"')

    else:
        print("The code should not get here. The first argument is not recognized. Exiting.")
        sys.exit()

args = parser.parse_args()
plot_type = sys.argv[1]

# Read surrogate results
results = loadpk('results')
print("results.pk dictionary contents : ", results.keys())

# Get basic dimensions
pcrv = results['pcrv']
ndim = pcrv.sdim
nout = pcrv.pdim

pctypes = pcrv.pctypes
print(f"Dimensionality : {ndim}")
print(f"Num of outputs : {nout}")

# Parameter names and output names files, if any. If files do not exist, uses the defaults.
pnames = read_textlist('pnames.txt', ndim, names_prefix='Par')
outnames = read_textlist('outnames.txt', nout, names_prefix='Out')



if(plot_type == 'sens'):
    allsens_main, allsens_total, allsens_joint = results['sens']

    sensmode = args.sensmode
    if sensmode == 'main':
        sensdata = allsens_main
    elif sensmode == 'total':
        sensdata = allsens_total

    pars = range(ndim)
    cases = range(nout)

    np.savetxt(f'allsens_{sensmode}.txt', sensdata)
    print(f'Sensitivities are reported in allsens_{sensmode}.txt')

    print('===================================================')
    print(f"Plotting {sensmode} sensitivities across all output QoIs (bar plot)")
    plot_sens(sensdata, pars, cases, par_labels=pnames, case_labels=outnames,
              ncol=5, grid_show=False, xlbl='', figname=f'sens_{sensmode}.png')
    print(f'Created file sens_{sensmode}.png')

elif(plot_type == 'jsens'):
    allsens_main, allsens_total, allsens_joint = results['sens']

    assert(allsens_main[0].shape[0]==ndim)
    allsens_main_ave = np.zeros((ndim,))
    allsens_joint_ave = np.zeros((ndim, ndim))
    print('Plotting main sand joint sensitivities (circular plots)')
    for iout in range(nout):
        print('======== Output # ', str(iout + 1), ' (QoI ', outnames[iout], ') =========')
        plot_jsens(allsens_main[iout], allsens_joint[iout],
                   inpar_names=pnames, figname=f'jsens_{iout}.png')
        print(f'Created file jsens_{iout}.png')

        allsens_main_ave += allsens_main[iout] / ndim
        allsens_joint_ave += allsens_joint[iout] / ndim

    print('Plotting averaged main sand joint sensitivities (circular plots)')
    plot_jsens(allsens_main_ave, allsens_joint_ave, inpar_names=pnames, figname='jsens_ave.png')
    print('Created file jsens_ave.png')


elif(plot_type=='sensmat'):
    allsens_main,allsens_total,allsens_joint=results['sens']

    sensmode=sys.argv[2]
    if sensmode=='main':
        sensdata=allsens_main
    elif sensmode=='total':
        sensdata=allsens_total


    pars=range(ndim)
    cases=range(nout)

    print('===================================================')
    print("Plotting ",sensmode," sensitivities across all output QoIs (matrix plot)")
    plot_sensmat(sensdata,pars,cases, par_labels=pnames,case_labels=outnames, cutoff=0.0, figname=f'sensmat_{sensmode}.png')
    print(f'Created file sensmat_{sensmode}.png')


elif(plot_type == 'dm'):

    # Parse the arguments
    trvals = args.trvals  # sys.argv[2:]

    for iout in range(nout):
        axes_labels = ['Model (' + outnames[iout] + ')', 'PC Apprx.']
        print('======== Output # ', str(iout + 1), ' (QoI ', outnames[iout], ') =========')
        datas = []
        models = []
        errorbars = []
        labels = []
        for trval in trvals:
            print("Plotting model-vs-surrogate at ", trval, " points")
            if trval not in results.keys():
                print(trval, " points are not present in results.")
                continue

            data, model, erb = results[trval][2][:, iout], results[trval][3][:, iout], results[trval][4][:, iout]
            datas.append(data)
            models.append(model)
            errorbars.append([erb]*2)
            labels.append(trval.capitalize() + ' points')

            plt.figure(figsize=(12,4))
            plt.errorbar(np.arange(len(data)), model, label='PC Apprx.', \
                         yerr=[erb, erb], fmt='o', color='r')
            plt.plot(np.arange(len(data)), data, 'bd', label='Model')
            plt.legend()
            plt.xlabel('Point Id')
            plt.title(f'QoI {outnames[iout]} ({trval.capitalize()})')
            plt.tight_layout()
            plt.savefig(f'idm_{iout}_{trval}.png')
            print(f'Created file idm_{iout}_{trval}.png')
            plt.clf()

        plot_dm(datas, models, errorbars=errorbars,
                labels=labels, axes_labels=['Model', 'PC Apprx.'],
                msize=11, figname=f'dm_{iout}.png')
        print(f'Created file dm_{iout}.png')
        plt.clf()

elif(plot_type == 'fit'):

    # Parse the arguments
    trvals = args.trvals  # sys.argv[2:]
    nevery = 1

    for trval in trvals:
        print("Plotting model-vs-surrogate at ", trval, " points")
        if trval not in results.keys():
            print(trval, " points are not present in results.")
            continue

        data, model, erb = results[trval][2], results[trval][3], results[trval][4]

        nsam = data.shape[0]
        for isam in range(0, nsam, nevery):
            f = plt.figure(figsize=(12,4))
            plt.plot(np.arange(nout), data[isam,:], 'bd', label='Model')
            #plt.plot(xc, model[isam,:], 'go-', ms=8, label='PC Apprx.')
            plt.errorbar(np.arange(nout), model[isam, :], label='PC Apprx.', \
                         yerr=[erb[isam, :], erb[isam, :]], fmt='o', color='r')
            plt.title(f'Sample #{isam+1}')
            #plt.xlabel(f'Output id')
            #plt.ylabel(f'Output values')
            plt.legend()
            plt.title(f'{trval.capitalize()} sample #{isam+1}/{nsam}')
            plt.xticks(range(nout), outnames)
            plt.tight_layout()
            plt.savefig(f'fit_s{str(isam).zfill(3)}_{trval}.png')
            plt.close(f)

            #print(f'Created file fit_s{str(isam).zfill(3)}_{trval}.png')
            plt.clf()

elif(plot_type=='1d'):
    # Parse the arguments
    trvals=args.trvals

    for iout in range(nout):
        fig, axes = plt.subplots(ndim, 1, figsize=(10,10*ndim),
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

        for idim in range(ndim):
            nom = np.zeros((ndim,))
            plot_1d(pcrv.evalPC, pcrv.function.domain, idim=idim, odim=iout,
                        nom=nom, ngr=111, color='g', lstyle='--', label='PC slice (rest at zero)', ax=axes[idim])
            pcrv1d = pcrv.slice_1d(idim)
            plot_1d(pcrv1d.evalPC, pcrv.function.domain[idim][np.newaxis,:], idim=0, odim=iout,
                    nom=nom[idim][np.newaxis], ngr=111, color='g', lstyle='-', label='PC slice (rest integ. out)', ax=axes[idim])
            axes[idim].set_xlabel(f'{pnames[idim]}')
            axes[idim].set_ylabel(f'{outnames[iout]}')

            for trval in trvals:
                if trval not in results.keys():
                    print(trval, " points are not present in results.")
                    continue
                output=results[trval][2][:,iout]
                germ_input=results[trval][0][:,idim]
                axes[idim].plot(germ_input, output, 'o', label=trval.capitalize()+' points')

            axes[idim].legend()
        plt.savefig(f'pcslices_o{iout}.png')
        plt.clf()

        print(f'Created file pcslices_o{iout}.png')

elif(plot_type=='2d'):

    for d1 in range(ndim):
        for d2 in range(d1+1,ndim):

            for iout in range(nout):
                nom = np.zeros((ndim,))
                plot_2d(pcrv.evalPC, pcrv.function.domain, idim=d1, jdim=d2, odim=iout,
                        nom=nom, ngr=111)
                plt.grid(False)
                plt.xlabel(f'{pnames[d1]}')
                plt.ylabel(f'{pnames[d2]}')
                plt.savefig(f'pccont_o{iout}_d{d1}_d{d2}.png')
                plt.clf()

                print(f'Created file pccont_o{iout}_d{d1}_d{d2}.png')


elif(plot_type == 'pdf'):
    nsam = 100000

    print('Computing and plotting output PDFs')

    pcsam = pcrv.sample(nsam)
    ysam = results['training'][2]

    for iout in range(nout):
        print('======== Output # ' + str(iout + 1) + ' (QoI ' + outnames[iout] + ') =========')
        plot_pdf1d(ysam[:, iout], color='lightblue', pltype='hist', histalpha=0.5)
        plot_pdf1d(pcsam[:, iout], color='pink', pltype='hist', histalpha=0.5)

        plt.gca().set_xlabel(f'Output QoI ({outnames[iout]})')
        plt.savefig(f'pdf_output_{iout}.png')
        print(f'Created file pdf_output_{iout}.png')
        plt.clf()


elif(plot_type == 'joy'):
    nsam = 100000

    print('Computing and plotting output PDFs with a joyplot layout')
    pcsam = pcrv.sample(nsam)
    ysam = results['training'][2]

    plot_joy([ysam, pcsam], np.arange(nout), outnames, ['lightblue', 'pink'], offset_factor=1., nominal=None, figname='pdf_joyplot.png')

    print(f'Created file pdf_joyplot.png')
    plt.clf()


else:
    print("plot_type not recognized. Exiting.")
    sys.exit()
