#!/usr/bin/env python3
"""Standalone MCMC runner.

Reads all run parameters from a JSON input file (default: mcmc_input.json)
and runs an Adaptive MCMC chain to calibrate a linear model to synthetic data.

Usage:
    python mcmc_run.py                          # uses mcmc_input.json
    python mcmc_run.py --input my_params.json   # uses a custom input file

Required input file keys:
    nmcmc         (int)   Number of MCMC samples to draw
    gamma         (float) AMCMC step-size/scaling parameter
    output_file   (str)   Path for the chain samples output file
    t0            (int)   Iteration at which adaptive proposal updates begin
    tadapt        (int)   How often the adaptive proposal covariance is updated

Optional input file keys:
    cov_file              (str)  Path for the covariance matrix output file;
                                 defaults to <output_file_stem>_cov<ext>
    restart               (bool) If true, restart from an existing samples file
                                 (default: false)
    restart_samples_file  (str)  Required when restart=true. Path to the samples
                                 file to restart from. The last sample is used as
                                 the initial chain state and indices continue from
                                 where that chain left off.
    restart_cov_file      (str)  Path to the covariance file to initialise the
                                 proposal from. Defaults to
                                 <restart_samples_file_stem>_cov<ext> when omitted.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize

from pytuq.minf.mcmc import AMCMC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cov_path(samples_path: str) -> str:
    """Derive a default covariance file path from the samples file path."""
    root, ext = os.path.splitext(samples_path)
    return f"{root}_cov{ext}"


def _parse_samples_file(filepath: str) -> dict:
    """Parse a chain samples .dat file.

    Returns a dict with keys:
        'hyperparams': dict of parsed key=value comment lines
        'col_names':   str (the column-name comment line, without leading '# ')
        'samples':     np.ndarray of data rows
        'last_index':  int (last value in the index column)
    Raises FileNotFoundError or ValueError on problems.
    """
    hyperparams = {}
    col_names = None
    data_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                content = line[1:].strip()
                m = re.fullmatch(r"(\w+)=(.+)", content)
                if m:
                    hyperparams[m.group(1)] = m.group(2).strip()
                else:
                    col_names = content
            else:
                if line.strip():
                    data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No data rows found in '{filepath}'")
    if col_names is None:
        raise ValueError(f"No column-name comment found in '{filepath}'")

    samples = np.loadtxt(data_lines, delimiter=",")
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    return {
        "hyperparams": hyperparams,
        "col_names": col_names,
        "samples": samples,
        "last_index": int(samples[-1, 0]),
    }


def _setup_lpinfo():
    """Build the synthetic dataset and log-posterior info dict."""
    def linear_model(x, par):
        return par[0] + par[1] * x

    def logpost(modelpars, lpinfo):
        ypred = lpinfo["model"](lpinfo["xd"], modelpars)
        ydata = lpinfo["yd"]
        nd = len(ydata)
        lpostm = 0.0
        for i in range(nd):
            for yy in ydata[i]:
                lpostm -= 0.5 * (ypred[i] - yy) ** 2 / lpinfo["lparams"]["sigma"] ** 2
                lpostm -= 0.5 * np.log(2 * np.pi)
                lpostm -= np.log(lpinfo["lparams"]["sigma"])
        return lpostm

    def negfcn(x, *pars):
        return -pars[0](x, pars[1])

    a_true, b_true, sigma = 1.0, 2.0, 0.2
    npt = 13
    xmin, xmax = 1.0, 2.0
    np.random.seed(42)
    xd = xmin + (xmax - xmin) * np.random.rand(npt)
    yd = linear_model(xd, [a_true, b_true]) + sigma * np.random.randn(npt)
    yd = yd.reshape(-1, 1)

    lpinfo = {
        "model": linear_model,
        "xd": xd,
        "yd": [y for y in yd],
        "ltype": "classical",
        "lparams": {"sigma": sigma},
    }
    return lpinfo, negfcn, logpost


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run an AMCMC chain from a JSON input file.")
    parser.add_argument(
        "--input", "-i",
        default="mcmc_input.json",
        metavar="INPUT_FILE",
        help="Path to the JSON input file (default: mcmc_input.json)",
    )
    args = parser.parse_args()

    # --- Load input file ---
    if not os.path.isfile(args.input):
        sys.exit(f"Error: input file '{args.input}' not found.")

    with open(args.input, "r") as f:
        params = json.load(f)

    required = {"nmcmc", "gamma", "output_file", "t0", "tadapt"}
    missing = required - params.keys()
    if missing:
        sys.exit(f"Error: input file is missing required keys: {sorted(missing)}")

    nmcmc       = int(params["nmcmc"])
    gamma       = float(params["gamma"])
    output_file = str(params["output_file"])
    t0          = int(params["t0"])
    tadapt      = int(params["tadapt"])
    cov_file    = str(params.get("cov_file", _cov_path(output_file)))
    restart     = bool(params.get("restart", False))

    print(f"Input file   : {args.input}")
    print(f"nmcmc        : {nmcmc}")
    print(f"gamma        : {gamma}")
    print(f"t0           : {t0}")
    print(f"tadapt       : {tadapt}")
    print(f"output_file  : {output_file}")
    print(f"cov_file     : {cov_file}")
    print(f"restart      : {restart}")

    # --- Set up model, data, and log-posterior ---
    lpinfo, negfcn, logpost = _setup_lpinfo()

    # --- Initialise chain state ---
    if restart:
        restart_samples_file = params.get("restart_samples_file")
        if not restart_samples_file:
            sys.exit("Error: 'restart_samples_file' must be set when restart=true.")

        print(f"restart_samples_file : {restart_samples_file}")

        try:
            parsed = _parse_samples_file(restart_samples_file)
        except FileNotFoundError:
            sys.exit(f"Error: restart samples file '{restart_samples_file}' not found.")
        except ValueError as e:
            sys.exit(f"Error: could not parse restart samples file: {e}")

        param_ini  = parsed["samples"][-1, 1:]   # last sample, skip index column
        last_index = parsed["last_index"]
        index_offset = last_index + 1

        restart_cov_file = str(params.get("restart_cov_file",
                                          _cov_path(restart_samples_file)))
        print(f"restart_cov_file     : {restart_cov_file}")

        cov_ini = None
        if os.path.exists(restart_cov_file):
            try:
                cov_ini = np.loadtxt(restart_cov_file, delimiter=",")
            except Exception:
                pass  # fall back to AMCMC default if file is unreadable

        amcmc_kwargs = {"cov_ini": cov_ini, "t0": t0, "tadapt": tadapt, "gamma": gamma}

    else:
        # Fresh run: initialise with BFGS optimum
        pdim = 2
        param_ini = np.random.rand(pdim)
        try:
            res = minimize(negfcn, param_ini, args=(logpost, lpinfo),
                           method="BFGS", options={"gtol": 1e-13})
            param_ini = res.x
        except Exception:
            pass  # fall back to random init if optimisation fails

        index_offset = 0
        amcmc_kwargs = {"t0": t0, "tadapt": tadapt, "gamma": gamma}

    # --- Run AMCMC ---
    print("\nRunning AMCMC chain...")
    my_amcmc = AMCMC(**amcmc_kwargs)
    my_amcmc.setLogPost(logpost, None, lpinfo=lpinfo)
    mcmc_results = my_amcmc.run(param_ini=param_ini, nmcmc=nmcmc)

    samples      = mcmc_results["chain"]
    map_params   = mcmc_results["mapparams"].tolist()
    max_log_post = float(mcmc_results["maxpost"])
    acc_rate     = float(mcmc_results["accrate"])

    print(f"Acceptance rate : {acc_rate:.4f}")
    print(f"MAP params      : {map_params}")
    print(f"Max log-post    : {max_log_post:.6f}")

    # --- Ensure output directory exists ---
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- Save chain samples ---
    indices   = (index_offset + np.arange(samples.shape[0])).reshape(-1, 1)
    data      = np.hstack([indices, samples])
    col_names = "index," + ",".join(f"param_{i}" for i in range(samples.shape[1]))
    header = (
        f"# nmcmc={nmcmc}\n"
        f"# t0={t0}\n"
        f"# tadapt={tadapt}\n"
        f"# gamma={gamma}\n"
        f"# accrate={acc_rate}\n"
        f"# {col_names}"
    )
    np.savetxt(output_file, data, delimiter=",", header=header, comments="")
    print(f"\nChain samples written to : {output_file}")

    # --- Save final proposal covariance ---
    cov_dir = os.path.dirname(cov_file)
    if cov_dir:
        os.makedirs(cov_dir, exist_ok=True)
    np.savetxt(cov_file, my_amcmc.propcov, delimiter=",")
    print(f"Covariance matrix written to : {cov_file}")


if __name__ == "__main__":
    main()
