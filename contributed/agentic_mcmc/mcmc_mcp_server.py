# MCP server script. See https://modelcontextprotocol.info/docs/quickstart/server/

import re
import subprocess
import sys
import time
import json
import os
import numpy as np
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("MCMC Sampling MCP Server")


def _finalize_meta(meta: Dict[str, Any], start: float) -> Dict[str, Any]:
    """Attach timing info and return meta_data dict."""
    meta = dict(meta)  # shallow copy
    meta["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return meta


def _parse_samples_file(filepath: str) -> Dict[str, Any]:
    """Parse a chain samples .dat file.

    Returns a dict with keys:
        'hyperparams': dict of parsed key=value comment lines
        'col_names': str (the column-name comment line, without leading '# ')
        'samples': np.ndarray of data rows
        'last_index': int (last value in the index column)
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
                # key=value hyperparameter lines
                m = re.fullmatch(r"(\w+)=(.+)", content)
                if m:
                    hyperparams[m.group(1)] = m.group(2).strip()
                else:
                    # column-name line (contains commas)
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

    last_index = int(samples[-1, 0])
    return {
        "hyperparams": hyperparams,
        "col_names": col_names,
        "samples": samples,
        "last_index": last_index,
    }


def _cov_path(samples_path: str) -> str:
    """Return the covariance file path for a given samples file path.

    my_file.dat -> my_file_cov.dat
    """
    root, ext = os.path.splitext(samples_path)
    return f"{root}_cov{ext}"


@mcp.tool()
def create_mcmc_input_file(
    output_path: str = "mcmc_input.json",
    nmcmc: int = 10000,
    gamma: float = 0.1,
    output_file: str = "mcmc_chain_samples.dat",
    t0: int = 100,
    tadapt: int = 100,
    cov_file: Optional[str] = None,
    restart: bool = False,
    restart_samples_file: Optional[str] = None,
    restart_cov_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a JSON input file for the mcmc_run.py standalone application.

    Writes all parameters required by mcmc_run.py into a JSON file. Supports
    both fresh runs and restarts from an existing chain.

    Path handling for all file arguments:
        Absolute paths are written as-is. Relative paths are stored as-is
        (relative to the directory that contains the input file), since
        mcmc_run.py is launched with that directory as its working directory.

    Args:
        output_path: Path (including filename) where the JSON input file will
            be written (default: "mcmc_input.json")
        nmcmc: Number of MCMC samples to draw (default: 10000)
        gamma: AMCMC step-size / scaling parameter (default: 0.1)
        output_file: Path for the chain samples output file. Relative paths
            are interpreted relative to the input file's directory
            (default: "mcmc_chain_samples.dat")
        t0: Iteration at which adaptive proposal covariance updates begin
            (default: 100)
        tadapt: How often the adaptive proposal covariance is updated
            (default: 100)
        cov_file: Path for the covariance matrix output file. Defaults to
            <output_file_stem>_cov<ext> when omitted.
        restart: If true, restart the chain from restart_samples_file
            (default: false)
        restart_samples_file: Path to the samples file to restart from.
            Required when restart=true. The last sample becomes the initial
            chain state and sample indices continue from that file's last index.
        restart_cov_file: Path to the covariance file to initialise the
            proposal from. Defaults to <restart_samples_file_stem>_cov<ext>
            when omitted.

    Returns:
        MCP contract shape with results and timing metadata:
        {
          "results": {
              "operation": "create_mcmc_input_file",
              "output_path": str,
              "run_directory": str,
              "parameters": { ... all written keys ... },
              "file_written": bool
          },
          "meta_data": {
              "is_error": bool,
              "elapsed_ms": float,
              "reason": str (if error)
          }
        }
    """
    start = time.perf_counter()
    meta: Dict[str, Any] = {}

    if restart and not restart_samples_file:
        meta.update({"is_error": True,
                     "reason": "'restart_samples_file' is required when restart=true"})
        return {
            "results": {
                "error": "'restart_samples_file' must be provided when restart=true",
                "file_written": False,
            },
            "meta_data": _finalize_meta(meta, start),
        }

    # The directory that will serve as the working directory for mcmc_run.py.
    run_dir = os.path.dirname(os.path.abspath(output_path))

    def _rel(path: str) -> str:
        """Normalise a file path to be relative to run_dir.

        Absolute paths are kept as-is.  Relative paths are first resolved
        against the current working directory, then expressed relative to
        run_dir.  This prevents double-nesting when the caller includes the
        run directory name in the path (e.g. 'my_mcmc_test/samples.dat'
        passed to a tool whose run_dir is already '.../my_mcmc_test').
        """
        if os.path.isabs(path):
            return path
        return os.path.relpath(os.path.abspath(path), run_dir)

    output_file_eff    = _rel(output_file)
    cov_file_eff       = _rel(cov_file) if cov_file is not None else _cov_path(output_file_eff)

    parameters: Dict[str, Any] = {
        "nmcmc": nmcmc,
        "gamma": gamma,
        "output_file": output_file_eff,
        "t0": t0,
        "tadapt": tadapt,
        "cov_file": cov_file_eff,
        "restart": restart,
    }

    if restart:
        rsf = _rel(restart_samples_file)
        parameters["restart_samples_file"] = rsf
        parameters["restart_cov_file"] = (
            _rel(restart_cov_file) if restart_cov_file is not None
            else _cov_path(rsf)
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            meta.update({"is_error": True, "reason": f"Directory creation error: {e}"})
            return {
                "results": {
                    "error": f"Could not create directory '{out_dir}': {e}",
                    "file_written": False,
                },
                "meta_data": _finalize_meta(meta, start),
            }

    try:
        with open(output_path, "w") as f:
            json.dump(parameters, f, indent=4)
            f.write("\n")
    except IOError as e:
        meta.update({"is_error": True, "reason": f"File write error: {e}"})
        return {
            "results": {
                "error": f"Could not write to '{output_path}': {e}",
                "file_written": False,
            },
            "meta_data": _finalize_meta(meta, start),
        }

    meta.update({"is_error": False})
    return {
        "results": {
            "operation": "create_mcmc_input_file",
            "output_path": output_path,
            "run_directory": run_dir,
            "parameters": parameters,
            "file_written": True,
        },
        "meta_data": _finalize_meta(meta, start),
    }


@mcp.tool()
def run_mcmc_run(
    input_file: str = "mcmc_input.json",
    mcmc_app: str = "mcmc_run.py",
) -> Dict[str, Any]:
    """Run an MCMC application with a given input file.

    Launches the specified MCMC application as a subprocess using the same
    Python interpreter that is running the MCP server. The subprocess is
    started with its working directory set to the folder that contains the
    input file, so that all relative paths inside the input file resolve
    correctly. The input file is passed as a filename-only argument since the
    working directory is already set to its containing folder.

    Args:
        input_file: Path to the JSON input file for the MCMC application
            (default: "mcmc_input.json")
        mcmc_app: Path to the MCMC Python application to run. A relative path
            is resolved relative to the directory that contains the input file.
            An absolute path is used as-is. (default: "mcmc_run.py")

    Returns:
        MCP contract shape with results and timing metadata:
        {
          "results": {
              "operation": "run_mcmc_run",
              "input_file": str,
              "mcmc_app": str,
              "run_directory": str,
              "returncode": int,
              "stdout": str,
              "stderr": str,
              "success": bool
          },
          "meta_data": {
              "is_error": bool,
              "elapsed_ms": float,
              "reason": str (if error)
          }
        }
    """
    start = time.perf_counter()
    meta: Dict[str, Any] = {}

    # Run from the directory that contains the input file so that all relative
    # paths inside it resolve correctly.
    abs_input = os.path.abspath(input_file)
    run_dir = os.path.dirname(abs_input)
    input_arg = os.path.basename(abs_input)

    # Resolve the application path: relative paths are anchored to run_dir.
    if os.path.isabs(mcmc_app):
        script_path = mcmc_app
    else:
        script_path = os.path.join(run_dir, mcmc_app)

    if not os.path.isfile(script_path):
        meta.update({"is_error": True, "reason": f"MCMC application not found at '{script_path}'"})
        return {
            "results": {
                "error": f"MCMC application not found at '{script_path}'",
                "success": False,
            },
            "meta_data": _finalize_meta(meta, start),
        }

    try:
        proc = subprocess.run(
            [sys.executable, script_path, "--input", input_arg],
            capture_output=True,
            text=True,
            cwd=run_dir,
        )
    except Exception as e:
        meta.update({"is_error": True, "reason": f"Subprocess error: {e}"})
        return {
            "results": {
                "error": f"Failed to launch '{script_path}': {e}",
                "success": False,
            },
            "meta_data": _finalize_meta(meta, start),
        }

    success = proc.returncode == 0
    meta.update({"is_error": not success})
    if not success:
        meta["reason"] = f"'{script_path}' exited with code {proc.returncode}"

    return {
        "results": {
            "operation": "run_mcmc_run",
            "input_file": input_file,
            "mcmc_app": script_path,
            "run_directory": run_dir,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "success": success,
        },
        "meta_data": _finalize_meta(meta, start),
    }


@mcp.tool()
def get_acceptance_rate(samples_file: str) -> Dict[str, Any]:
    """Read a saved MCMC samples file and return the acceptance rate.

    Parses the header of the samples file produced by mcmc_run.py (or
    restart_mcmc_chain) and extracts the acceptance rate that was recorded
    as '# accrate=<value>' when the chain was saved.

    Args:
        samples_file: Path to the chain samples file (.dat)

    Returns:
        MCP contract shape with results and timing metadata:
        {
          "results": {
              "operation": "get_acceptance_rate",
              "samples_file": str,
              "accrate": float,
              "hyperparams": dict   (all other header key=value pairs)
          },
          "meta_data": {
              "is_error": bool,
              "elapsed_ms": float,
              "reason": str (if error)
          }
        }
    """
    start = time.perf_counter()
    meta: Dict[str, Any] = {}

    try:
        parsed = _parse_samples_file(samples_file)
    except FileNotFoundError:
        meta.update({"is_error": True, "reason": f"File not found: {samples_file}"})
        return {
            "results": {"error": f"Samples file not found: {samples_file}"},
            "meta_data": _finalize_meta(meta, start),
        }
    except ValueError as e:
        meta.update({"is_error": True, "reason": f"Parse error: {e}"})
        return {
            "results": {"error": f"Could not parse samples file: {e}"},
            "meta_data": _finalize_meta(meta, start),
        }

    hp = parsed["hyperparams"]
    if "accrate" not in hp:
        meta.update({"is_error": True, "reason": "'accrate' not found in file header"})
        return {
            "results": {
                "error": "'accrate' key not present in header; was this file written by a recent version of mcmc_run.py?",
                "hyperparams": hp,
            },
            "meta_data": _finalize_meta(meta, start),
        }

    accrate = float(hp["accrate"])
    other_hp = {k: v for k, v in hp.items() if k != "accrate"}

    meta.update({"is_error": False})
    return {
        "results": {
            "operation": "get_acceptance_rate",
            "samples_file": samples_file,
            "accrate": accrate,
            "hyperparams": other_hp,
        },
        "meta_data": _finalize_meta(meta, start),
    }


@mcp.tool()
def thinking(thought: str) -> str:
    """
    Record a thought or plan for the next step. The LLM should call this tool
    before executing other tools to explain its reasoning.

    Returns a JSON string indicating the thought was recorded.
    """
    print(f"--- THOUGHT: {thought} ---")
    return json.dumps({"status": "Thought recorded"})


@mcp.tool()
def task_finished(summary: str) -> str:
    """
    Signal that the overall task is complete.

    The model is expected to call this when it believes all steps have
    been accomplished. The argument should be a human-readable summary of
    the work performed. This tool simply echoes the summary.
    """
    print(f"--- TOOL: Task Finished! ---")
    return json.dumps({"final_summary": summary})


if __name__ == "__main__":
    mcp.run(transport='stdio')
