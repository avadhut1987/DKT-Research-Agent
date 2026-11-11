# DKT Research Agent

A LangGraph-based research assistant for the Double Kicked Top model.

## Features
- Run DKT trajectory simulations when requested.
- Compare DKT simulations.
- Search existing research evidence.
- Read existing CSV results.

## Requirements
- Python 3.14 or compatible version
- Julia for simulation tools
- Google Gemini API key

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API key
```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

## Initialize Julia dependencies
```bash
julia --project=QuantumKickedTop.jl -e 'using Pkg; Pkg.instantiate()'
```

## Run the agent
```bash
PYTHONPATH=. python -m dkt_agent.agent
```
