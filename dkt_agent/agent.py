"""Gemini-powered LangGraph research agent for the Double Kicked Top."""

from __future__ import annotations

import os
import sys

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from dkt_agent.langchain_tools import (
    run_dkt_simulation,
    compare_dkt_simulations,
    run_dkt_comparison_study,
)


SYSTEM_PROMPT = """
You are a scientific research assistant specializing in the
Double Kicked Top (DKT), quantum chaos, classical dynamics,
quantum information, Floquet systems, and classical-quantum correspondence.

PHYSICAL MODEL

The DKT Floquet operator is

    U = exp[-i k' Jx^2/(2j)] exp[-i k Jz^2/(2j)] exp[-i p Jy].

The parameter convention is

    kr     = (k + kprime)/2
    ktheta = (k - kprime)/2

and therefore

    k      = kr + ktheta
    kprime = kr - ktheta.

The default rotation angle is p = pi/2.

AVAILABLE TOOLS

1. run_dkt_simulation:
   Executes one classical DKT trajectory using the Julia
   QuantumKickedTop.jl physics engine.

2. compare_dkt_simulations:
   Compares two previously generated trajectory CSV files.
   Calculates initial, final, maximum, mean, and RMS
   Euclidean separation and saves a comparison CSV.

3. run_dkt_comparison_study:
   Automatically runs two DKT simulations and compares their
   complete trajectories in one tool call.
   Returns both simulation results, comparison statistics,
   and all three CSV paths.

TOOL-SELECTION POLICY

1. When the user requests exactly two simulations with the same
   kr, initial condition, p, and number of kicks, but different
   ktheta values, prefer run_dkt_comparison_study.

2. Extract all supplied parameters accurately:
   kr, ktheta_a, ktheta_b, theta0, phi0, nsteps, and optionally p.

3. If the user does not specify p, use p = pi/2.

4. If the user does not specify optional parameters, use the
   comparison-study tool defaults where appropriate.

5. Do not ask the user to repeat parameters already supplied.

6. Use run_dkt_simulation for an individual simulation or when
   the requested simulations have different initial conditions
   or other parameters that cannot be represented by the
   comparison-study tool.

7. Use compare_dkt_simulations when the user wants to compare
   previously generated trajectories. Pass the exact CSV paths
   returned by the simulation tools.

8. For more than two simulations, call run_dkt_simulation
   separately for each requested parameter set. Compare only
   the requested pairs using compare_dkt_simulations.

9. Never claim a simulation or comparison succeeded unless
   the corresponding tool returned status = success.

REPORTING SIMULATION RESULTS

For every successful simulation, report:
- All input parameters.
- Initial and final Cartesian states.
- Number of trajectory rows.
- Maximum norm-conservation error.
- Exact trajectory CSV path.

The trajectory includes the initial state at step 0 and
therefore contains nsteps + 1 rows.

REPORTING COMPARISONS

When a comparison succeeds, report:
- Initial separation.
- Final separation.
- Maximum separation.
- Mean separation.
- RMS separation.
- Number of comparison rows.
- Exact comparison CSV path.

Also report both simulation CSV paths.

Use actual numerical values returned by the tools.
Do not estimate, fabricate, or silently modify results.

SCIENTIFIC INTERPRETATION

The Cartesian separation is

    d(n) = sqrt((XA(n)-XB(n))^2
              + (YA(n)-YB(n))^2
              + (ZA(n)-ZB(n))^2).

Explain what the computed separation indicates about the
two trajectories.

For trajectories on the unit sphere, their Euclidean
separation cannot exceed 2.

A large trajectory separation alone does not establish chaos,
a positive Lyapunov exponent, ergodicity, or mixing.

Do not infer Lyapunov exponents, KS entropy, mixing, or
ergodicity unless the corresponding diagnostic has actually
been computed.

Distinguish numerical observations from physical interpretation.
Be precise, transparent, and scientifically cautious.
"""


def build_agent():
    """Initialize Gemini and construct the LangGraph agent."""

    if not (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    ):
        raise EnvironmentError(
            "Gemini API key not found. Check the environment configuration."
        )

    model = init_chat_model(
        "gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0,
    )

    return create_agent(
        model=model,
        tools=[
            run_dkt_simulation,
            compare_dkt_simulations,
            run_dkt_comparison_study,
        ],
        system_prompt=SYSTEM_PROMPT,
    )


def read_user_request():
    """Read either a single-line or explicitly multiline request."""

    first_line = input("You: ").strip()

    if first_line.lower() != "/multi":
        return first_line

    print(
        "Enter your complete request. "
        "Type /send on a separate line when done."
    )

    lines = []

    while True:
        line = input("... ")

        if line.strip().lower() == "/send":
            break

        lines.append(line)

    return "\n".join(lines).strip()


def main():
    """Run an interactive DKT research-agent session."""

    try:
        agent = build_agent()
    except Exception as exc:
        print(f"Agent initialization failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print("=" * 65)
    print("DOUBLE KICKED TOP — RESEARCH AGENT")
    print("=" * 65)
    print("Model: Gemini 2.5 Flash")
    print("Physics engine: QuantumKickedTop.jl")
    print("Framework: LangGraph")
    print("Available tools: 3")
    print("\nEnter a research request, or type 'exit' to quit.")
    print("For multiline requests, type /multi and finish with /send.\n")

    conversation = []

    while True:
        try:
            user_input = read_user_request()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting research agent.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting research agent.")
            break

        conversation.append(
            {"role": "user", "content": user_input}
        )

        try:
            response = agent.invoke(
                {"messages": conversation}
            )

            conversation = response["messages"]

            last_message = conversation[-1]
            print("\nDKT Agent:", last_message.content, "\n")

        except Exception as exc:
            print(f"\nAgent error: {exc}\n")
            conversation.pop()


if __name__ == "__main__":
    main()
