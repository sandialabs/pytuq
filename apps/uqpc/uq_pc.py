#!/usr/bin/env python


#=====================================================================================
#=====================================================================================

import argparse
import os
import sys
import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.func.benchmark import Ishigami
from pytuq.utils.xutils import savepk
from pytuq.utils.mindex import get_mi, get_npc
from pytuq.workflows.fits import pc_fit

#######################################################################################
#######################################################################################
#######################################################################################
# Model for online_example only
model = Ishigami()


# Parse input arguments
usage_str = 'Workflow to build PC surrogates of multioutput models.'
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=usage_str)
parser.add_argument("-r", "--regime", dest="run_regime", type=str, default='online_example',
                    help="Run regime", choices=['online_example', 'online_bb', 'offline'])
parser.add_argument("-p", "--pdom", dest="pdomain_file", type=str,
                    default=None, help="Parameter domain file")
parser.add_argument("-c", "--pcfile", dest="inpc_file", type=str,
                    default=None, help="Input PC coef. file")
parser.add_argument("-d", "--pcdim", dest="in_pcdim", type=int,
                    default=None, help="Input PC stoch. dimension")
parser.add_argument("-x", "--pctype", dest="pc_type", type=str, default='LU',
                    help="PC type", choices=['HG', 'LU', 'LU_N', 'LG', 'JB', 'SW'])
parser.add_argument("-o", "--pcord", dest="in_pcord", type=int, default=1, help="Input PC order")
parser.add_argument("-m", "--method", dest="fit_method", type=str, default='anl',
                    help="Surrogate construction method", choices=['anl', 'lsq', 'bcs'])
parser.add_argument("-s", "--sampl", dest="sam_method", type=str, default='quad',
                    help="Sampling method", choices=['quad', 'rand'])
parser.add_argument("-n", "--nqd", dest="num_pt", type=int, default=7, \
                    help="Number of quadrature points per dim (if sampl=quad), \
                    Number of training points (if sampl=rand)")
parser.add_argument("-v", "--ntst", dest="num_test", type=int, default=0,
                    help="Number of testing (testing) points; can be 0")  # (default: %(default)s)
parser.add_argument("-t", "--outord", dest="out_pcord", type=int, default=3, help="Output PC order")
parser.add_argument("-e", "--tol", dest="tolerance", type=float, default=1.e-3,
                    help="Tolerance (currently for method=bcs only)")
parser.add_argument("-z", "--seed", dest="seed", type=int, default=None,
                    help="Seed for exact reproduction. If None, random seed is used.")
args = parser.parse_args()

# Flags for input checks
pflag = False
cflag = False
dflag = False




# Hardwired file names
input_train_file = 'ptrain.txt'
input_test_file  = 'ptest.txt'
germ_train_file = 'qtrain.txt'
germ_test_file = 'qtest.txt'
output_train_file = 'ytrain.txt'
output_test_file = 'ytest.txt'


# Argument compatibility checks
if args.pdomain_file != None:
    pflag = True
if args.inpc_file != None:
    cflag = True
if args.in_pcdim != None:
    dflag = True



# Arguments that may be overwritten
in_pcord = args.in_pcord
pc_type = args.pc_type

# Organize input PC structure
if (pflag and cflag):
    print("Need to provide input domain or input PC coef file, not both. Exiting.")
    sys.exit()
elif (pflag and dflag):
    print("Need to provide input domain or input dimensionality, not both. Exiting.")
    sys.exit()
elif (int(pflag) + int(cflag) + int(dflag) == 0):
    in_pcdim = 2  # default if no argument is given
    pcf_all = np.vstack((np.zeros((in_pcdim,)), np.eye(in_pcdim)))
elif pflag:
    # Load parameter domain file
    if (os.path.isfile(args.pdomain_file)):
        pdom = np.loadtxt(args.pdomain_file).reshape(-1, 2)
        pcf_all = np.vstack((0.5 * (pdom[:, 1] + pdom[:, 0]),
                             np.diag(0.5 * (pdom[:, 1] - pdom[:, 0]))))
        in_pcdim = pdom.shape[0]
        print(
            f"Input order (-o) and pctype (-x) are overwritten since domain file {args.pdomain_file} is given.")
        in_pcord = 1
        pc_type = "LU"

        for i in range(pdom.shape[0]):
            if(pdom[i, 0] > pdom[i, 1]):
                print("Error: The domain file %s contains wrong bounds. Check the row number %d. Exiting." % (
                    args.pdomain_file, i + 1))
                sys.exit()
    else:
        print("Error: The requested domain file %s does not exist. Exiting." % args.pdomain_file)
        sys.exit()
elif (cflag):
    if (os.path.isfile(args.inpc_file)):
        in_pcdim = args.in_pcdim
        pcf_all = np.loadtxt(args.inpc_file).reshape(-1, in_pcdim)
        assert(pcf_all.shape[0]==get_npc(in_pcord, in_pcdim))

    else:
        print("Error: The requested input PC coefficient file %s does not exist. Exiting." % args.inpc_file)
        sys.exit()
elif (dflag):
    in_pcdim = args.in_pcdim
    pcf_all = np.vstack((np.zeros((in_pcdim,)), np.eye(in_pcdim)))
else:
    print("If this message appears, there must be a bug in the code. Exiting.")
    sys.exit()


# Get the dimensionality of the problem
_, npar = pcf_all.shape

# Print the inputs for reference
print("Run regime                        %s" % args.run_regime)
if (cflag):
    print("Input PC coefficient file         %s" % args.inpc_file)
if (pflag):
    print("Input parameter domain file       %s" % args.pdomain_file)
print("Input PC dim                      %d" % in_pcdim)
print("Input PC order                    %d" % in_pcord)
print("PC type                           %s" % pc_type)
print("The number of input parameters    %d" % npar)
print("Method                            %s" % args.fit_method)
print("Sampling method                   %s" % args.sam_method)
print(" with parameter                   %d" % args.num_pt)
print("Number of testing points          %d" % args.num_test)
print("Output PC order                   %d" % args.out_pcord)

# Set up the input PC object
mi = get_mi(in_pcord, in_pcdim)
pc = PCRV(in_pcdim, in_pcdim, pc_type, mi=mi, cfs=pcf_all.T)

# (1) Generate sample points for online regimes
if args.run_regime != "offline":
    print("######################## Generate input samples #####################")
    print("#### Generating training samples")

    if args.sam_method == "quad":
        germ_train, inqdw = pc.quadGerm(pts=[args.num_pt] * in_pcdim)
        #np.savetxt(wg_train, inqdw)
        np.savetxt(germ_train_file, germ_train)

    elif args.sam_method == "rand":
        germ_train = pc.sampleGerm(args.num_pt, seed=args.seed)
        np.savetxt(germ_train_file, germ_train)

    else:
        print("Error: Sampling method is not recognized. Should be 'quad' or 'rand'. Exiting.")
        sys.exit()

    ntrn = germ_train.shape[0]
    print(f"Germ samples for training are in {germ_train_file} in a format {ntrn} x {in_pcdim}")

    # Evaluate input PCs at training points
    ptrain = pc.evalPC(germ_train)
    np.savetxt(input_train_file, ptrain)
    print(f"Parameter samples for training are in {input_train_file} in a format {ntrn} x {npar}")

    # Generate points, if requested, for the testing of the surrogate
    if args.num_test > 0:
        print("#### Generating testing samples")
        germ_test = pc.sampleGerm(args.num_test, seed=args.seed)
        np.savetxt(germ_test_file, germ_test)

        print(f"Germ samples for testing are in {germ_test_file} in a format {args.num_test} x {in_pcdim}")
        ptest = pc.evalPC(germ_test)
        np.savetxt(input_test_file , ptest)
        print(f"Parameter samples for testing are in {input_test_file } in a format {args.num_test} x {npar}")

print("######################## Load input samples #########################")
# (2) Load sample points for online or offline regimes
ptrain = np.loadtxt(input_train_file).reshape(-1, npar)
germ_train = np.loadtxt(germ_train_file).reshape(-1, in_pcdim)
ntrn = ptrain.shape[0]

if (args.num_test > 0):
    ptest = np.loadtxt(input_test_file ).reshape(-1, npar)
    germ_test = np.loadtxt(germ_test_file).reshape(-1, in_pcdim)



print(f"Number of training points for surrogate construction   : {ntrn}")
print(f"Number of testing points for surrogate construction    : {args.num_test}")

print("######################## Evaluate/Load forward model  ###############")
# (3) Get model outputs

# Run the model online or....
if args.run_regime == "online_example":
    ytrain = model(ptrain)
    np.savetxt(output_train_file , ytrain)

    if (args.num_test > 0):
        ytest = model(ptest)
        np.savetxt(output_test_file , ytest)

elif args.run_regime == "online_bb":
    assert(os.path.exists('model.x'))

    os.system('./model.x ' + input_train_file + ' ' + output_train_file)
    ytrain = np.loadtxt(output_train_file).reshape(ntrn, -1)
    if (args.num_test > 0):
        os.system('./model.x ' + input_test_file  + ' ' + output_test_file)
        ytest = np.loadtxt(output_test_file).reshape(args.num_test, -1)

# ...or read the results from offline simulations
elif args.run_regime == "offline":
    ytrain = np.loadtxt(output_train_file).reshape(ntrn, -1)
    if (args.num_test > 0):
        ytest = np.loadtxt(output_test_file).reshape(args.num_test, -1)


# Read the number of output observables or the number of values of design parameters (e.g. location, time etc..)
_, nout = ytrain.shape
print(f"Number of output observables of the model : {nout}")


print("######################## Construct PC surrogates  ###################")
# (4) Obtain the PC surrogate using model simulations

# Empty arrays and lists to store results

# All sensitivities
allsens_main = np.empty((nout, in_pcdim))
allsens_total = np.empty((nout, in_pcdim))
allsens_joint = np.empty((nout, in_pcdim, in_pcdim))

# Train and test predictions and their variances
ytrain_pc = np.empty((ntrn, nout))
ytest_pc = np.empty((args.num_test, nout))
ytrain_pc_var = np.empty((ntrn, nout))
ytest_pc_var = np.empty((args.num_test, nout))
# Train and test relative errors
relerr_train = np.empty((nout,))
relerr_test = np.empty((nout,))



output_pcrv, linregs = pc_fit(germ_train, ytrain,
                              order=args.out_pcord, pctype=pc_type,
                              method=args.fit_method, eta=args.tolerance)




ytrain_pc = output_pcrv.function(germ_train)
if (args.num_test > 0):
    ytest_pc = output_pcrv.function(germ_test)


for iout, lreg in enumerate(linregs):
    ytrain_pc_var[:,iout] = lreg.predicta(output_pcrv.evalBases(germ_train, iout), msc=1)[1]
    if (args.num_test > 0):
        ytest_pc_var[:,iout] = lreg.predicta(output_pcrv.evalBases(germ_test, iout), msc=1)[1]


print("######################## Compute relative errors  ###################")
# (5) Obtain the global sensitivity indices from the PC surrogate

relerr_train=np.linalg.norm(ytrain-ytrain_pc, axis=0)/np.linalg.norm(ytrain, axis=0)
print("Surrogate relative error for all outputs at training points : \n", relerr_train)


if (args.num_test>0):
    relerr_test=np.linalg.norm(ytest-ytest_pc, axis=0)/np.linalg.norm(ytest, axis=0)
    print("Surrogate relative error for all outputs at testing points : \n", relerr_test)

print("######################## Compute PC sensitivities  ###################")
# (6) Obtain the global sensitivity indices from the PC surrogate

allsens_main = output_pcrv.computeSens()
allsens_total = output_pcrv.computeTotSens()
allsens_joint = output_pcrv.computeJointSens()

print(f"Sum of main sensitivities  : {np.sum(allsens_main, axis=1)}")
print(f"Sum of total sensitivities : {np.sum(allsens_total, axis=1)}")


# Results container
results = {'training': (germ_train, ptrain, ytrain, ytrain_pc, np.sqrt(ytrain_pc_var), relerr_train), 'sens': (
        allsens_main, allsens_total, allsens_joint), 'pcrv': output_pcrv}
if(args.num_test > 0):
    results['testing']= (germ_test, ptest, ytest, ytest_pc, np.sqrt(ytest_pc_var), relerr_test)


# Save results
savepk(results, nameprefix='results')


