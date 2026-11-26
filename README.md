# DKT Research Agent

An AI-assisted computational research platform for the **Double Kicked Top (DKT)** model, developed using Python, LangGraph, Julia, and Streamlit.

The project integrates classical nonlinear dynamics, quantum entanglement calculations, classical–quantum correspondence, ergodicity and mixing-related diagnostics, scientific visualization, reproducible numerical workflows, and AI-assisted research tools.

The platform is designed to support computational investigations of the DKT model, from generating classical trajectories and studying phase-space structures to examining the ergodic hierarchy and comparing classical and quantum dynamics.

## Features

### 1. Classical DKT Dynamics

The classical simulation tools provide:

- Simulation of classical DKT trajectories.
- Comparison of trajectories under different kicking parameters.
- Investigation of phase-space structures and their dependence on the asymmetry parameter.
- Generation of phase-space portraits and trajectory visualizations.
- Fixed-point analysis and stability investigations.
- Parameter sweeps and comparisons of trajectory measures.
- Export of trajectory data and associated metadata.

### 2. Ergodicity and Mixing-Related Diagnostics

The project provides numerical tools for investigating different levels of the ergodic hierarchy.

**Mean ergodicity**

- Finite-data evaluation of the von Neumann mean-ergodicity distance, $D_N$.
- Investigation of the mean-square distance between finite-time averages and the phase-space average.
- Evaluation over uniformly sampled initial conditions.
- Comparison across observables and DKT parameter configurations.

The diagnostic is designed to investigate whether finite-time averages approach the corresponding phase-space average in the sampled ensemble. A vanishing distance is consistent with mean ergodicity for the tested observable and sampling procedure; a nonzero plateau may indicate a nonconstant limiting contribution. Finite numerical data cannot establish the asymptotic property.

**Birkhoff-average and variance diagnostics**

- Calculation of finite-time trajectory averages.
- Variance of time averages across distinct initial conditions.
- Investigation of variance scaling with observation time.
- Comparison of selected Cartesian observables.

These diagnostics examine the behaviour of time averages along individual trajectories and across an ensemble of initial conditions.

**Correlation and weak-mixing-related diagnostics**

- Calculation of observable autocorrelation functions.
- Finite-time and finite-lag correlation analysis.
- Finite-window Cesàro averages of squared autocorrelations.
- Analysis of nine Cartesian observables and their correlations.
- Comparison across multiple trajectories and parameter configurations.

**Strong-mixing-related diagnostics**

- Numerical estimates of finite-partition alpha coefficients.
- Analysis of correlations between events separated by selected time lags.
- Use of pooled trajectory ensembles and sampled time origins.
- Investigation of the dependence on spherical partition resolution and lag range.

The alpha-coefficient calculation is performed for a specified finite spherical partition. It does not calculate the supremum over all measurable sets required by the mathematical definition of strong mixing.

**Combined ergodic-hierarchy workflow**

The project also provides an integrated diagnostic workflow that reports the following separately:

1. Mean-ergodicity-related diagnostics.
2. Birkhoff-average and variance behaviour.
3. Weak-mixing-related Cesàro correlation indicators.
4. Strong-mixing-related finite-partition alpha-coefficient estimates.

The diagnostics are intended to be interpreted together with their sampling assumptions and numerical limitations, not as automatic classifications of the underlying dynamics.

**Additional dynamical diagnostics**

- Maximal Lyapunov exponent estimates.
- Kolmogorov–Sinai (KS) entropy estimates under applicable assumptions.
- Parameter scans and comparisons of selected numerical diagnostics.

### 3. Reproducible Ergodic-Hierarchy Visual Dashboard

The Streamlit application includes a dedicated visual dashboard for generating, saving, selecting, and analysing DKT trajectory ensembles.

The workflow is designed to avoid manual trajectory-file preparation and repeated metadata entry.

**Generate and save a DKT trajectory ensemble**

Users can specify:

- Average kicking parameter, $k_r$.
- Asymmetry parameter, $k_\theta$.
- Rotation parameter, $p$.
- Angular-grid size.
- Number of kicks.
- Transient-discard interval.
- Maximum correlation lag.
- Spherical partition resolution.

The application generates an ensemble of distinct initial conditions under the selected DKT map, saves the trajectories and metadata, and proceeds to the numerical diagnostics.

The angular-grid size can be selected within the supported range of 5–50 points per direction. For example, a grid size of 5–8 produces 25–64 initial conditions.

**Automatic data management**

- Trajectory CSV files are generated automatically.
- Parameter-labelled filenames preserve the identity of each trajectory.
- An `ensemble_manifest.json` records the ensemble configuration and provenance.
- Generated ensembles are stored in separate run directories.
- Previously generated metadata-backed ensembles can be selected and analysed again.
- Individual ensembles can be downloaded as ZIP archives.
- Numerical results and diagnostic artifacts are stored separately from the original trajectory data.

**Dashboard visualizations and outputs**

The dashboard provides:

- Mean-ergodicity diagnostic results.
- Birkhoff-average variance versus analysed trajectory length.
- Normalized autocorrelation plots.
- Cesàro means of squared autocorrelations, selectable by observable.
- Finite-partition alpha-coefficient estimates versus lag.
- Numerical tables and downloadable JSON and CSV results.

The dashboard separates the generation of trajectory data from its subsequent analysis, allowing saved ensembles to be reused for further investigations.

**Individual trajectory compatibility**

Individual simulations generated through the `simulate_dkt` workflow are saved in nested run directories. Their trajectory CSV files use parameter-labelled filenames and are accompanied by a `trajectory_metadata.json` file.

The dashboard searches these directories recursively. It groups trajectories only when their map parameters match and their initial conditions are distinct.

Older trajectory files without metadata cannot always be assigned reliably to a specific DKT map configuration. The application therefore avoids guessing their parameters.

### 4. Quantum Entanglement Dynamics

The Julia-backed quantum tools provide:

- Entanglement entropy as a function of kick number.
- Long-time-averaged entanglement entropy heatmaps over grids of spin-coherent initial states.
- Transient-discarded averaging.
- State-norm diagnostics.
- Heatmap failure reporting.
- CSV outputs and visualizations.

The normalized subsystem entropy is

$$
s_A = \frac{S_A}{\log_2[\min(Q+1,N-Q+1)]},
$$

where $N=2s$ and $Q$ denotes the subsystem size used in the bipartition.

The evolving-state entanglement calculations are distinct from Floquet-eigenstate entanglement analysis.

For heatmaps, the transient convention is to discard kicks $0:T_0$ and average kicks $T_0+1:T$. Failed calculations remain missing values rather than being interpreted as zero.

### 5. Classical–Quantum Correspondence

The project includes a Julia-backed comparison of classical DKT trajectories and quantum spin-coherent-state evolution.

The comparison provides:

- Kick-resolved classical coordinates $X,Y,Z$.
- Normalized quantum spin expectations $\langle J_x\rangle/s$, $\langle J_y\rangle/s$, and $\langle J_z\rangle/s$.
- Component-wise mean absolute error (MAE).
- Root mean square error (RMSE).
- Maximum component-wise error.
- Vector-difference summaries.
- CSV output and Streamlit visualizations.

The initial quantum normalized-spin vector is checked against the specified classical initial point before the comparison is accepted.

Finite-time differences between a single classical trajectory and quantum expectation values are not, by themselves, a diagnostic of classical chaos. Entanglement entropy is treated as a separate observable.

### 6. AI-Assisted Research Interface

The application also includes a **Research Studio** for plan-first investigations. It translates a scientific question into a structured workflow, identifies suitable existing tools and required inputs, distinguishes read-only evidence retrieval from new numerical execution, and records authorization gates. Planning itself does not run calculations.

The research planner is designed to preserve the model-specific scientific constraints, including the distinction between the roles of $k_r$ and $k_\theta$, and the limitations of finite-data ergodic-hierarchy diagnostics. See [Phase 3 product and interface principles](docs/PHASE3_PRODUCT_DESIGN.md).

Research Studio now also supports persistent research campaigns. Each campaign records the plan, ordered stages, exact execution inputs, timestamps, results, errors and provenance, and can be resumed from the interface. Existing-evidence discovery is read-only; new numerical stages require explicit authorization and a separate execution action.

For fixed-$k_r$ geometry studies, the workflow includes both a matched two-trajectory comparison and a matched-ensemble comparison. The ensemble comparison reuses the same uniformly sampled initial conditions for both $k_\theta$ values and reports pooled and per-initial-condition Jensen–Shannon divergences on a common equal-area spherical partition. These are finite-sample distributional comparisons, not evidence of changed chaoticity or proof of distinct invariant measures.

For direction-dependent ergodicity questions, the planner can propose observable-resolved mean-ergodicity and Birkhoff-variance diagnostics for $X,Y,Z$ and selected quadratic/cross observables. The outputs must remain observable-specific. The current samplers use the full uniform sphere; region-conditioned local ergodicity is not yet implemented, and full-sphere results must not be presented as local-region conclusions.


The LangGraph research agent provides access to simulation and analysis tools through natural-language interaction.

Capabilities include:

- Running authorized DKT simulations.
- Comparing parameter configurations.
- Analysing existing trajectory CSV files.
- Running numerical research diagnostics.
- Performing Lyapunov and KS-entropy studies.
- Investigating mean-ergodicity-related quantities.
- Analysing Birkhoff averages and correlation functions.
- Running weak-mixing-related and strong-mixing-related numerical studies.
- Running quantum entanglement calculations.
- Comparing classical and quantum dynamics.
- Searching existing research evidence.

The Streamlit interface provides an interactive environment for simulations, research workflows, plots, and downloadable numerical results.

The agent is intended to assist with computational research and evidence-based analysis. Its conclusions should remain within the scope of the available numerical results and their assumptions.

## Scientific Parameter Convention

The DKT kicking parameters are defined as 

$$
k_r = \frac{k+k'}{2}, \qquad k_\theta = \frac{k-k'}{2},
$$

so that 

$$
k=k_r+k_\theta, \qquad k'=k_r-k_\theta.
$$

The distinction between these parameters is central to the research:

- $k_r$ controls the degree of classical chaos, including the regular-to-chaotic transition, Lyapunov behaviour, and KS entropy.
- $k_\theta$ changes the geometry, orientation, and structure of phase-space features. It can also break time-reversal symmetry.
- Increasing $k_\theta$ at fixed $k_r$ should not be interpreted as increasing the degree of classical chaos or changing the Lyapunov behaviour.

A useful comparison is to investigate $k_\theta=0$ and $k_\theta=10^4$ while keeping $k_r$, initial conditions, $p$, and the number of kicks fixed. This isolates the effects associated with the asymmetry parameter from changes in the average kicking strength.

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
│   ├── research_tools.py
│   ├── mixing_hierarchy.py
│   ├── mixing_tools.py
│   └── mean_ergodic_dashboard.py
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
│       ├── test_mixing_hierarchy.py
│       ├── test_research_capabilities.py
│       └── test_research_workflows.py
│
└── QuantumKickedTop.jl/
    └── Julia simulation package (Git submodule)
```

The structure above highlights the principal application, analysis, Julia, and test components. Additional modules and scripts may be present as the project develops.

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

The application provides an interactive environment for DKT simulations, parameter studies, the ergodic-hierarchy dashboard, numerical diagnostics, and quantum research workflows.

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

### Classical Dynamics

1. Select $k_r$, $k_\theta$, and the initial conditions.
2. Generate classical trajectories or phase-space portraits.
3. Compare trajectory structures for selected parameter configurations.
4. Examine Lyapunov and other numerical diagnostics where appropriate.

### Mean Ergodicity and Birkhoff-Average Investigations

1. Select the DKT map parameters and observables.
2. Evaluate the finite-data mean-ergodicity distance over sampled initial conditions.
3. Examine finite-time trajectory averages.
4. Investigate the variance of Birkhoff averages across distinct initial conditions.
5. Compare the results across observables, parameter configurations, and observation windows.
6. Interpret the numerical behaviour without treating finite-time results as proofs of asymptotic mean ergodicity or pointwise ergodicity.

### Weak-Mixing-Related Investigations

1. Select multiple trajectories generated under the same DKT map.
2. Calculate observable autocorrelation functions.
3. Examine finite-window Cesàro averages of squared autocorrelations.
4. Compare the results across observables and time windows.
5. Investigate sensitivity to initial-condition sampling and the available lag range.

### Strong-Mixing-Related Investigations

1. Generate or select an ensemble of trajectories under a common DKT map.
2. Specify the spherical partition resolution.
3. Calculate finite-partition alpha-coefficient estimates over selected time lags.
4. Examine sensitivity to partition resolution, lag range, and trajectory sampling.
5. Check approximate stationarity of the pooled trajectory/time-origin sample.
6. Interpret the results as finite-partition numerical indicators rather than a proof of strong mixing.

### Reproducible Ergodic-Hierarchy Dashboard Workflow

1. Open **Generate and save a DKT trajectory ensemble**.
2. Set $k_r$, $k_\theta$, and $p$.
3. Choose the angular-grid size and number of kicks.
4. Specify the transient-discard interval, maximum lag, and spherical partition.
5. Click **Generate, save & run diagnostics**.
6. Allow the application to generate the trajectory files, save the metadata and manifest, and calculate the diagnostics.
7. Examine the resulting plots and numerical tables.
8. Select a previously generated metadata-backed ensemble to repeat or extend the analysis.
9. Download the ensemble as a ZIP archive when required.

The generated ensemble and its diagnostic outputs are stored separately, preserving the original trajectory data and its provenance.

### Quantum Entanglement

1. Choose the quantum spin and subsystem partition.
2. Specify the initial spin-coherent state or grid of initial states.
3. Run the entanglement dynamics or heatmap calculation.
4. Specify the transient-discard interval and averaging interval.
5. Examine the resulting entropy curves, heatmaps, and numerical outputs.

### Classical–Quantum Correspondence

1. Specify matching classical and quantum initial conditions.
2. Select the DKT parameters and number of kicks.
3. Calculate classical coordinates and normalized quantum spin expectations.
4. Examine component-wise and vector error measures.
5. Investigate the finite-time correspondence without conflating expectation-value deviations with classical chaoticity.

## Scientific Interpretation and Limitations

The project is intended as a computational research assistant. Numerical outputs should be interpreted in the context of the model, assumptions, finite observation times, numerical precision, and sampling choices.

In particular:

- A finite-data mean-ergodicity distance is not a mathematical proof of von Neumann mean ergodicity.
- Finite-time Birkhoff averages and their ensemble variance do not establish asymptotic pointwise ergodicity.
- Finite-window Cesàro diagnostics are numerical indicators, not proofs of weak mixing.
- Observed correlation decay over a finite interval does not prove strong mixing.
- The finite-partition alpha coefficient is not the supremum over all measurable sets required by the mathematical definition of strong mixing.
- Pooled trajectory and time-origin samples used in alpha-coefficient estimates should be checked for approximate stationarity.
- Lyapunov and KS-entropy estimates depend on numerical methods, iteration length, and applicable theoretical assumptions.
- Quantum entanglement measures and classical trajectory diagnostics describe different aspects of the dynamics.
- Classical–quantum correspondence should be assessed using clearly specified observables, initial-state alignment, and finite-time error measures.

The AI agent is designed to analyse available evidence and numerical results. It should not replace independent verification of calculations or justify unsupported physical conclusions.

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

The project is under active development. Its simulation, analysis, visualization, and research-agent capabilities are being extended and refined.

The current platform brings together classical DKT dynamics, quantum entanglement, classical–quantum correspondence, and numerical investigations of the ergodic hierarchy within a reproducible computational workflow.

Numerical results and research conclusions should be checked against the underlying Julia and Python implementations, relevant theoretical assumptions, and independent calculations.

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Avadhut V. Purohit.

You are free to use, modify, distribute, and adapt this software in accordance with the terms of the MIT License.

See the [LICENSE](LICENSE) file for the complete license text.


## Periodic-orbit hierarchy

The numerical periodic-orbit search detects primitive DKT cycles over a configurable
period range, excludes lower-period repeats, deduplicates cyclic rotations, and
reports closure residuals and Floquet multipliers. See [the periodic-orbit guide](docs/periodic_orbits.md)
for the Julia command, Python API, Streamlit panel, output schema and numerical
limitations. Searches are finite and non-exhaustive; absence from a result is not
evidence of nonexistence. At fixed \(k_r\), \(k_\theta\) comparisons are interpreted
as orbit-geometry comparisons, not changes in classical chaoticity.
