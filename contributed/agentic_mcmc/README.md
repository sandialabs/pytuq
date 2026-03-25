# PyTUQ MCMC — Agentic MCMC Workflow (Proof of Concept)

**Note:** To successfully build and run this example, copy the files over to 
another folder outside of the pytuq distribution"

> **This is a proof of concept.** The current implementation uses a fixed
> synthetic dataset (a noisy linear model) and a hard-wired AMCMC sampler as
> a demonstration vehicle. It is deliberately designed to be extended: swap in
> your own model, likelihood, data, and sampler by modifying `mcmc_run.py` and
> the MCP server tools.

---

## Overview

This folder implements an **agentic MCMC workflow** in which a large language
model (LLM) autonomously manages an Adaptive Markov Chain Monte Carlo (AMCMC)
calibration run via the
[Model Context Protocol (MCP)](https://modelcontextprotocol.info). The agent
can create run configurations, launch chains, evaluate convergence diagnostics
(acceptance rate), tune hyperparameters, restart chains, and decide when to
stop — all without manual intervention.

The workflow has three layers:

```
Jupyter notebook  ←→  MCP client  ←→  MCP server  ←→  mcmc_run.py
  (user / prompt)      (LLM loop)      (tools)         (sampler)
```

---

## Files

| File | Description |
|------|-------------|
| `mcmc_run.py` | **Standalone MCMC application.** Reads all run parameters from a JSON input file, runs an AMCMC chain using [PyTUQ](https://github.com/sandialabs/pytuq), writes the chain samples and final proposal covariance to output files. Supports fresh starts (initialised via BFGS) and restarts from a previous chain. Run directly with `python mcmc_run.py --input <file.json>`. |
| `mcmc_mcp_server.py` | **MCP server.** Exposes the workflow steps as MCP tools that the LLM can call: create input files, launch `mcmc_run.py`, read acceptance rates from saved chains, and signal task completion. Run as a stdio MCP server; the client starts it automatically. |
| `mcp_client.py` | **MCP client with multi-provider LLM support.** Connects to the MCP server, manages the agentic tool-call loop, and exposes `run_agent_loop()` / `process_query()` / `chat_loop()`. Provider is auto-detected from the model name (`claude-*` → Anthropic, `gpt-*`/`o1*`/`o3*` → OpenAI, `gemini-*` → Google). All tool results from a run are stored in `client.tool_results` for post-hoc inspection in the notebook. |
| `mcmc_mcp_test.ipynb` | **Demo Jupyter notebook.** Configures the model and prompt, runs the agentic loop, extracts output file paths into Python variables, and plots chain traces for all parameters. |

### MCP server tools

| Tool | Purpose |
|------|---------|
| `create_mcmc_input_file` | Write a JSON input file for `mcmc_run.py`. Accepts all AMCMC hyperparameters plus optional restart settings. All relative file paths in the input file are normalised to be relative to the input file's directory, so that `mcmc_run.py` always finds them regardless of the caller's working directory. |
| `run_mcmc_run` | Launch `mcmc_run.py` as a subprocess (using the same Python interpreter as the server) with a given input file. The subprocess runs from the input file's directory. Accepts an optional `mcmc_app` argument so a different application script can be used. |
| `get_acceptance_rate` | Parse the header of a saved samples file and return the acceptance rate recorded there. |
| `thinking` | Utility tool that lets the agent log its reasoning before taking an action. |
| `task_finished` | Signals the end of the agentic loop and returns a human-readable summary. |

### Input file format (`mcmc_input.json`)

```json
{
    "nmcmc":        10000,
    "gamma":        0.1,
    "output_file":  "mcmc_chain_samples.dat",
    "t0":           100,
    "tadapt":       100,
    "cov_file":     "mcmc_chain_samples_cov.dat",
    "restart":      false
}
```

For a restart, add:

```json
{
    "restart":              true,
    "restart_samples_file": "mcmc_chain_samples.dat",
    "restart_cov_file":     "mcmc_chain_samples_cov.dat"
}
```

All paths are relative to the directory containing the input file.

### Samples file format

Chain samples are saved as comma-separated `.dat` files with a commented
header:

```
# nmcmc=10000
# t0=100
# tadapt=100
# gamma=0.1
# accrate=0.3124
# index,param_0,param_1
0,1.0312,...
1,1.0289,...
```

The `index` column provides a global sample number that carries over across
restarts, making it straightforward to concatenate multiple chain files and
plot traces.

---

## Installation

A Python virtual environment is recommended. All dependencies are declared in
`pyproject.toml`, so a single install command is sufficient.

### With `pip`

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install .
```

### With `uv` (faster)

```bash
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install .
```

For an editable install (useful during development):

```bash
pip install -e .   # or: uv pip install -e .
```

### Required packages

The following packages are declared in `pyproject.toml` and installed
automatically by the commands above:

| Package | Purpose |
|---------|---------|
| `pytuq` | PyTUQ UQ library — provides the `AMCMC` sampler |
| `numpy` | Array operations |
| `scipy` | BFGS optimisation for chain initialisation |
| `mcp` | Model Context Protocol SDK (server + client) |
| `python-dotenv` | Load API keys from a `.env` file |
| `anthropic` | Anthropic API SDK *(Claude models)* |
| `openai` | OpenAI API SDK *(GPT / o-series models)* |
| `google-genai` | Google Generative AI SDK *(Gemini models)* |
| `matplotlib` | Plotting chain traces in the notebook |
| `ipykernel` | Running the demo notebook in Jupyter |

All three LLM SDKs are included so you can switch providers without
reinstalling. Only the SDK matching your chosen model actually needs a valid
API key at runtime.

---

## Environment variables

API keys must be available as environment variables before running the
notebook or the client. The easiest way is to create a `.env` file in this
directory (it is loaded automatically by `python-dotenv`):

```
# .env  —  do NOT commit this file to version control

# Required for Anthropic / Claude models
ANTHROPIC_API_KEY=your-key-here

# Required for OpenAI / GPT models
OPENAI_API_KEY=your-key-here

# Required for Google / Gemini models
GOOGLE_API_KEY=your-key-here
```

Only the key corresponding to the model you select needs to be set. The
client will exit immediately with a clear error message if the required key
is missing.

Optional variables:

| Variable | Purpose |
|----------|---------|
| `MCP_CA_BUNDLE` | Custom CA bundle path for HTTPS verification (Anthropic client) |
| `SSL_CERT_FILE` | Alternative SSL certificate file |
| `SHIRTY_API_KEY` | API key for the internal Shirty endpoint (Sandia-specific) |
| `SHIRTY_API_BASE` | Base URL for the Shirty endpoint (default: `https://shirty.sandia.gov/api/v1`) |

---

## Quick start

1. Set up your `.env` file with the appropriate API key.
2. Open `mcmc_mcp_test.ipynb` in Jupyter.
3. Edit the `MODEL` and `server_path` variables in the configuration cell.
4. Edit the `mcmc_prompt` to describe the run you want.
5. Run all cells. The agent will manage the MCMC workflow autonomously,
   and the final cells will extract output paths and plot chain traces.

---

## A note on LLM behaviour

The agentic workflow relies on an LLM to interpret the task description,
select hyperparameters, evaluate results, and decide when the chain has
converged. **Results will vary depending on the model chosen.** In practice:

- Different models have different tendencies for tool-call frequency,
  reasoning verbosity, and willingness to iterate.
- The same model may produce different sequences of tool calls across runs
  due to the inherent stochasticity of LLM sampling.
- Smaller or less capable models may struggle to follow multi-step
  instructions reliably, call tools in the wrong order, or fail to terminate
  the loop gracefully.
- In short, the agentic behaviour depends on the choice of LLM, the
  specificity of the prompt, and — not unlike MCMC itself — the phase of
  the moon.

This is expected and is part of what makes agentic workflows an active
research area. The prompt in the notebook can be tuned to guide the agent
more tightly if needed.

---

## Extending this proof of concept

The current implementation uses a synthetic 2-parameter linear model as a
placeholder. To adapt this to a real problem:

1. **Replace the model and likelihood** in `mcmc_run.py` — modify
   `_setup_lpinfo()` to use your own forward model, data, and log-posterior.
2. **Add diagnostics** — extend the MCP server tools to compute convergence
   statistics (e.g. R-hat, ESS) and expose them so the agent can reason about
   them.
3. **Tune the prompt** — the `mcmc_prompt` in the notebook is the primary
   lever for steering the agent's strategy.
4. **Add more tools** — any step in your workflow that can be expressed as a
   Python function can become an MCP tool, giving the agent more capabilities.
