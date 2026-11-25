# DKT Research Agent

An AI-assisted computational research platform for the **Double Kicked Top (DKT)** model, developed using Python, LangGraph, Julia, and Streamlit.

The project integrates classical nonlinear dynamics, quantum entanglement calculations, classical–quantum correspondence, numerical diagnostics, scientific visualization, and AI-assisted research workflows.

## Features

### 1. Classical DKT Dynamics

- Simulate classical DKT trajectories.
- Compare trajectories under different kicking parameters.
- Investigate phase-space structures and their dependence on the asymmetry parameter.
- Generate phase-space portraits and trajectory visualizations.
- Analyze fixed points and their stability.
- Perform parameter sweeps and compare trajectory measures.

### 2. Ergodicity and Mixing-Related Diagnostics

The research tools provide numerical investigations of the ergodic hierarchy, including:

- **Mean ergodicity:** trajectory time averages, observable fluctuations, and variance-based studies across initial conditions.
- **Correlation analysis:** observable autocorrelation functions and finite-time correlation diagnostics.
- **Weak-mixing-related diagnostics:** finite-window Cesàro averages of squared autocorrelations for nine Cartesian observables across multiple trajectories.
- **Strong-mixing-related diagnostics:** finite-partition alpha-coefficient estimates from pooled trajectory ensembles and sampled time origins.
- **Combined hierarchy workflow:** a single agent tool that reports the three finite-data diagnostic levels separately for matched DKT map parameters and multiple initial conditions.
- **Lyapunov analysis:** finite-time and long-time numerical estimates of maximal Lyapunov exponents.
- **Kolmogorov–Sinai (KS) entropy:** numerical estimates associated with the Lyapunov spectrum under applicable assumptions.
- **Parameter scans:** investigate how selected diagnostics vary with the average kicking parameter.

These calculations provide finite-time numerical evidence, not mathematical proofs of ergodicity, weak mixing, or strong mixing. The ergodicity-related study tests selected observables against uniform-sphere moments; it is not a general test of von Neumann mean ergodicity. The weak-mixing-related study examines only a finite set of observables and lags. The strong-mixing-related study estimates the alpha coefficient only for a specified finite spherical partition; it does not take the supremum over all measurable sets. The pooled trajectory/time-origin sample should be checked for approximate stationarity. Correlation decay alone is not sufficient to establish strong mixing.

### 3. Quantum Entanglement Dynamics

The Julia-backed quantum tools provide:

- Entanglement entropy as a function of kick number.
- Long-time-averaged entanglement entropy heatmaps over grids of spin-coherent initial states.
- Transient-discarded averaging.
- State-norm diagnostics and heatmap failure reporting.
- CSV outputs and visualizations.

The normalized subsystem entropy is

$
s_A = \frac{S_A}{\log_2[\min(Q+1,N-Q+1)]},
$

where $N=2s$ and $Q$ denotes the subsystem size used in the bipartition.

The evolving-state entanglement calculations are distinct from Floquet-eigenstate entanglement analysis.

For heatmaps, the transient convention is to discard kicks $0:T_0$ and average kicks $T_0+1:T$. Failed calculations remain missing values rather than being interpreted as zero.

### 4. Classical–Quantum Correspondence

The project includes a Julia-backed comparison of classical DKT trajectories and quantum spin-coherent-state evolution.

The comparison provides:

- Kick-resolved classical coordinates $X,Y,Z$.
- Normalized quantum spin expectations
  $\langle J_x\rangle/s$, $\langle J_y\rangle/s$, and $\langle J_z\rangle/s$.
- Component-wise mean absolute error (MAE).
- Root mean square error (RMSE).
- Maximum component-wise error.
- Vector-difference summaries.
- CSV output and Streamlit visualizations.

The initial quantum normalized-spin vector is checked against the specified classical initial point before the comparison is accepted.

Finite-time differences between a single classical trajectory and quantum expectation values are not, by themselves, a diagnostic of classical chaos. Entanglement entropy is treated as a separate observable.

### 5. AI-Assisted Research Interface

The LangGraph research agent provides access to simulation and analysis tools through natural-language interaction.

Capabilities include:

- Running authorized DKT simulations.
- Comparing parameter configurations.
- Analyzing existing trajectory CSV files.
- Running numerical research diagnostics.
- Performing Lyapunov and KS-entropy studies.
- Investigating correlation and ergodicity-related quantities.
- Running quantum entanglement calculations.
- Comparing classical and quantum dynamics.
- Searching existing research evidence.

The Streamlit interface provides an interactive environment for simulations, research workflows, plots, and downloadable numerical results.

## Scientific Parameter Convention

The DKT kicking parameters are defined as

$
k_r = \frac{k+k'}{2},
\qquad
k_\theta = \frac{k-k'}{2},
$

so that

$
k=k_r+k_\theta,
\qquad
k'=k_r-k_\theta.
$

The distinction between these parameters is central to the research:

- $k_r$ controls the degree of classical chaos and the regular-to-chaotic transition.
- $k_\theta$ changes the geometry, orientation, and structure of phase-space features. It can also break time-reversal symmetry.
- Increasing $k_\theta$ at fixed $k_r$ should not be interpreted as increasing the degree of classical chaos or changing the Lyapunov behavior.

A useful comparison is to investigate $k_\theta=0$ and $k_\theta=10^4$ while keeping $k_r$, initial conditions, $p$, and the number of kicks fixed.

## Project Structure

```text
DKT-Research-Agent/
│
├── app.py
├── README.md
├── requirements.txt
│
├── dkt_agent/
│   ├── agent.py
│   ├── tools.py
│   ├── langchain_tools.py
│   ├── classical_analysis.py
│   ├── evidence.py
│   └── research_tools.py
│
├── scripts/
│   └── julia/
│       ├── simulate_dkt.jl
│       ├── simulate_dkt_grid.jl
│       ├── analyze_dkt_fixed_points.jl
│       ├── compare_classical_quantum.jl
│       ├── dkt_lyapunov_ks.jl
│       ├── run_entanglement_dynamics.jl
│       └── run_entropy_heatmap.jl
│
├── tests/
│   └── python/
│       ├── test_agent_tool_flow.py
│       ├── test_classical_analysis.py
│       ├── test_classical_quantum_comparison.py
│       ├── test_dkt_tool.py
│       ├── test_advanced_diagnostics.py
│       ├── test_research_capabilities.py
│       └── test_research_workflows.py
│
└── QuantumKickedTop.jl/
    └── Julia simulation package (Git submodule)
```

## Requirements

- Python 3.10 or later.
- Julia (the project has been developed and tested with Julia 1.12).
- A Google Gemini API key for the AI research agent.
- Git, including Git submodule support.

Python dependencies are listed in `requirements.txt`.

## Installation

Clone the repository with its Julia submodule:

```bash
git clone --recurse-submodules https://github.com/avadhut1987/DKT-Research-Agent.git
cd DKT-Research-Agent
```

If the repository was cloned without submodules, initialize them:

```bash
git submodule update --init --recursive
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Instantiate the Julia project dependencies:

```bash
julia --project=QuantumKickedTop.jl -e 'using Pkg; Pkg.instantiate()'
```

Set the Google Gemini API key:

```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

Replace the placeholder with your own API key. Do not commit API keys or other credentials to the repository.

## Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Streamlit will display the local application URL in the terminal. Open that URL in your browser.

The application provides an interactive interface for DKT simulations, parameter studies, numerical diagnostics, and quantum research workflows.

## Running the Research Agent

The agent can also be launched from the terminal:

```bash
PYTHONPATH=. python -m dkt_agent.agent
```

Ensure that the virtual environment is activated and the Gemini API key is available in the environment.

## Julia Simulation Scripts

The Julia scripts in `scripts/julia/` provide standalone entry points for selected numerical workflows, including:

- Classical DKT trajectory simulation.
- Grid-based trajectory simulation.
- Fixed-point analysis.
- Classical–quantum comparison.
- Lyapunov exponent and KS-entropy diagnostics.
- Entanglement entropy dynamics.
- Entanglement entropy heatmap calculations.

Some scripts require input parameters or output paths. Consult the corresponding script for its expected arguments and configuration.

## Running Tests

Run the Python test suite from the repository root:

```bash
PYTHONPATH=. python -m pytest tests/python/
```

The tests cover agent-tool integration, classical analysis, classical–quantum comparison, research diagnostics, and research workflows.

## Suggested Research Workflows

### Classical dynamics

1. Select $k_r$, $k_\theta$, and the initial conditions.
2. Generate classical trajectories or phase-space portraits.
3. Compare trajectory structures for selected parameter configurations.
4. Examine Lyapunov and other numerical diagnostics where appropriate.

### Ergodicity and mixing-related investigations

1. Select observables and initial conditions.
2. Calculate finite-time trajectory averages and fluctuations.
3. Investigate observable autocorrelation functions.
4. For a multi-trajectory hierarchy study, pass at least two existing trajectories from distinct initial conditions under the same DKT map to diagnose_mixing_hierarchy.
5. Examine the selected-observable Birkhoff variance, finite-window Cesàro squared-autocorrelation envelopes, and finite-partition alpha coefficients separately.
6. Check ensemble stationarity, partition resolution, lag range, and sensitivity to initial-condition sampling.
7. Compare the behavior across parameter configurations and observation windows without converting finite diagnostics into asymptotic classifications.

### Quantum entanglement

1. Choose the quantum spin and subsystem partition.
2. Specify the initial spin-coherent state or grid of initial states.
3. Run the entanglement dynamics or heatmap calculation.
4. Specify the transient-discard interval and averaging interval.
5. Examine the resulting entropy curves, heatmaps, and numerical outputs.

### Classical–quantum correspondence

1. Specify matching classical and quantum initial conditions.
2. Select the DKT parameters and number of kicks.
3. Calculate classical coordinates and normalized quantum spin expectations.
4. Examine component-wise and vector error measures.
5. Investigate the finite-time correspondence without conflating expectation-value deviations with classical chaoticity.

## Scientific Interpretation and Limitations

The project is intended as a computational research assistant. Numerical outputs should be interpreted in the context of the model, assumptions, finite observation times, numerical precision, and sampling choices.

In particular:

- Finite-time trajectory statistics do not establish asymptotic ergodicity.
- Finite-window Cesàro diagnostics are numerical indicators, not proofs of weak mixing.
- Observed correlation decay over a finite interval does not prove strong mixing.
- Lyapunov and KS-entropy estimates depend on numerical methods, integration or iteration length, and applicable theoretical assumptions.
- Quantum entanglement measures and classical trajectory diagnostics describe different aspects of the dynamics.
- Classical–quantum correspondence should be assessed using clearly specified observables, initial-state alignment, and finite-time error measures.

The AI agent is designed to analyze available evidence and numerical results. It should not replace independent verification of calculations or justify unsupported physical conclusions.

## Troubleshooting

**Julia dependency errors**

Run:

```bash
julia --project=QuantumKickedTop.jl -e 'using Pkg; Pkg.instantiate()'
```

**Missing Gemini API key**

Set `GOOGLE_API_KEY` in the active terminal before launching the agent or Streamlit application.

**Python import errors**

Activate the virtual environment and reinstall the dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Git submodule issues**

Run:

```bash
git submodule update --init --recursive
```

**Streamlit port already in use**

Launch the application on another port:

```bash
streamlit run app.py --server.port 8502
```

## Development Status

The project is under active development. Its simulation, analysis, and research-agent capabilities are being extended and refined.

Numerical results and research conclusions should be checked against the underlying Julia and Python implementations, relevant theoretical assumptions, and independent calculations.

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Avadhut V. Purohit.

You are free to use, modify, distribute, and adapt this software in accordance with the terms of the MIT License.

See the [LICENSE](LICENSE) file for the complete license text.