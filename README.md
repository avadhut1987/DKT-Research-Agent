# DKT Research Agent

An AI-assisted research application for the **Double Kicked Top (DKT)** model, built using Python, LangGraph, Julia, and Streamlit.

The project combines classical DKT simulations, numerical analysis, visualization, and AI-assisted research interaction.

## Features

- Run classical DKT trajectory simulations.
- Compare DKT simulations under different parameter values.
- Investigate the effects of varying the asymmetry parameter \(k_\theta\) at fixed \(k_r\).
- Analyze classical fixed points and their stability.
- Visualize phase-space trajectories and structures.
- Search existing research evidence.
- Read and analyze existing CSV simulation results.
- Interact with an AI-powered DKT research assistant.
- Access the research agent through a Streamlit web interface or terminal.

## Research Objective

An important objective of this project is to investigate how increasing \(k_\theta\) affects phase-space structures while keeping \(k_r\) fixed.

For example, compare:

$
k_\theta=0
\qquad \text{and} \qquad
k_\theta=10^4.
$

For a meaningful comparison, keep the following parameters identical:

- Average kicking parameter $k_r$.
- Initial conditions $(\theta_0,\phi_0)$.
- Parameter $p$.
- Number of kicks.
- Numerical resolution, where applicable.

The distinction between the two parameters is important: \(k_r\) controls the degree of classical chaos, whereas $k_\theta$ changes the orientation and structure of phase-space features.

## Project Structure

```text
DKT-Research-Agent/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dkt_agent/
│   ├── agent.py
│   ├── tools.py
│   └── langchain_tools.py
│
├── scripts/
│   └── julia/
│       ├── simulate_dkt_grid.jl
│       └── analyze_dkt_fixed_points.jl
│
└── QuantumKickedTop.jl/
    └── Julia simulation package (Git submodule)
```

## Requirements

- Python 3.14 or a compatible version.
- Julia.
- Git.
- Google Gemini API key.
- Internet connection for installing dependencies and accessing the Gemini API.

Check your installations:

```bash
python --version
julia --version
git --version
```

## Installation

### 1. Clone the repository

Clone the repository together with its Julia submodule:

```bash
git clone --recurse-submodules https://github.com/avadhut1987/DKT-Research-Agent.git

cd DKT-Research-Agent
```

If you have already cloned the repository without its submodule:

```bash
git submodule update --init --recursive
```

### 2. Create the Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Configure the Google Gemini API key

The research agent uses the Google Gemini API. It accepts either `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

On Linux, configure one of the following environment variables:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

Alternatively:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Replace `YOUR_API_KEY` with your actual Gemini API key.

Do not commit API keys or other credentials to GitHub.

### 4. Initialize Julia dependencies

The classical simulation tools use the Julia package maintained in the `QuantumKickedTop.jl` submodule.

Initialize its dependencies:

```bash
julia --project=QuantumKickedTop.jl -e 'using Pkg; Pkg.instantiate()'
```

## Running the Application

There are two ways to use the DKT Research Agent.

### Option 1: Streamlit Web Application (Recommended)

Activate the Python environment:

```bash
source .venv/bin/activate
```

Launch the application:

```bash
streamlit run app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

The Streamlit interface provides access to the research agent and its available simulation and analysis tools.

### Option 2: Terminal-Based Research Agent

The original LangGraph research agent can also be launched directly from the terminal:

```bash
source .venv/bin/activate

PYTHONPATH=. python -m dkt_agent.agent
```

This starts an interactive research-agent session in the terminal.

## Running Julia Scripts Independently

Julia scripts can also be executed directly from the repository root.

For example, the classical DKT grid simulation:

```bash
julia --project=QuantumKickedTop.jl \
    scripts/julia/simulate_dkt_grid.jl
```

The script may require command-line arguments. Refer to its argument definitions before running it without arguments.

Fixed-point analysis:

```bash
julia --project=QuantumKickedTop.jl \
    scripts/julia/analyze_dkt_fixed_points.jl
```

## Suggested Research Workflow

To investigate the effects of increasing $k_\theta$:

1. Choose a fixed value of $k_r$.
2. Specify the initial conditions and number of kicks.
3. Run the simulation for $k_\theta=0$.
4. Repeat for larger values of $k_\theta$, keeping the other parameters unchanged.
5. Compare trajectories and phase-space structures.
6. Examine changes in stretching, rotation, and orientation.
7. Analyze the numerical results and interpret the observations.

The objective is to distinguish changes in phase-space geometry from changes in the underlying degree of classical chaos.

## Troubleshooting

### Python dependency errors

Ensure that the virtual environment is activated:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Julia dependency errors

Initialize the submodule and instantiate its dependencies:

```bash
git submodule update --init --recursive

julia --project=QuantumKickedTop.jl -e 'using Pkg; Pkg.instantiate()'
```

### Gemini API authentication errors

Check that `GOOGLE_API_KEY` is correctly configured:

```bash
echo "${GOOGLE_API_KEY:+API key is configured}"
```

This command checks whether the variable exists without printing its value.

### Streamlit port already in use

Use another port:

```bash
streamlit run app.py --server.port 8502
```

## Development Status

This project is under active development. Simulation capabilities, numerical analysis tools, visualizations, and AI research-agent functionality are evolving as the research progresses.
