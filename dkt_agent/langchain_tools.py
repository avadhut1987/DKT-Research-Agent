"""LangChain tools for the Double Kicked Top research agent."""

import json

from langchain_core.tools import tool

from dkt_agent.tools import (
    simulate_dkt,
    compare_dkt_trajectories,
    run_dkt_comparison_study as _run_dkt_comparison_study,
)


@tool
def run_dkt_simulation(
    kr: float,
    ktheta: float,
    theta0: float,
    phi0: float,
    nsteps: int,
    p: float = 1.5707963267948966,
) -> str:
    """
    Run a classical Double Kicked Top trajectory using the Julia physics engine.

    Parameters are given in radians where applicable.
    The DKT convention is k = kr + ktheta and kprime = kr - ktheta.
    The simulation starts at (theta0, phi0) and performs nsteps kicks.
    The default free-rotation angle is pi/2.

    Returns simulation parameters, initial and final states,
    norm-conservation error, and trajectory CSV path.
    """

    result = simulate_dkt(
        kr=kr,
        ktheta=ktheta,
        theta0=theta0,
        phi0=phi0,
        nsteps=nsteps,
        p=p,
        include_trajectory=False,
    )

    compact_result = {
        "status": result["status"],
        "parameters": result["parameters"],
        "trajectory_rows": result["trajectory_rows"],
        "initial_state": result["initial_state"],
        "final_state": result["final_state"],
        "initial_norm": result["initial_norm"],
        "final_norm": result["final_norm"],
        "max_norm_error": result["max_norm_error"],
        "trajectory_csv": result["trajectory_csv"],
    }

    return json.dumps(compact_result, indent=2)


@tool
def compare_dkt_simulations(
    trajectory_csv_1: str,
    trajectory_csv_2: str,
) -> str:
    """
    Compare two previously generated classical DKT trajectories.

    Supply the two trajectory CSV file paths returned by
    run_dkt_simulation.

    The trajectories must have matching step indices.

    Calculates the Euclidean distance between their Cartesian
    states at every kick and reports initial, final, maximum,
    mean, and RMS separation.

    Saves the kick-by-kick comparison data to a CSV file.
    """

    result = compare_dkt_trajectories(
        trajectory_csv_1=trajectory_csv_1,
        trajectory_csv_2=trajectory_csv_2,
    )

    return json.dumps(result, indent=2)


@tool
def run_dkt_comparison_study(
    kr: float = 1.0,
    ktheta_a: float = 0.0,
    ktheta_b: float = 10000.0,
    theta0: float = 0.75,
    phi0: float = -1.0,
    nsteps: int = 1000,
    p: float = 1.5707963267948966,
) -> str:
    """
    Run two DKT simulations and compare their complete trajectories.

    Both simulations use the same kr, initial condition, rotation
    angle, and number of kicks. Only ktheta differs.

    The tool automatically:
    1. Runs simulation A using ktheta_a.
    2. Runs simulation B using ktheta_b.
    3. Compares the complete trajectories.
    4. Saves both trajectories and the comparison to CSV files.

    Returns actual simulation results, trajectory separation
    statistics, and all three CSV paths.

    This tool does not infer or estimate numerical results.
    """

    result = _run_dkt_comparison_study(
        kr=kr,
        ktheta_a=ktheta_a,
        ktheta_b=ktheta_b,
        theta0=theta0,
        phi0=phi0,
        nsteps=nsteps,
        p=p,
    )

    # Keep the LangChain response concise while preserving
    # all important numerical results and output file paths.

    compact_result = {
        "status": result["status"],
        "study_parameters": result["study_parameters"],
        "simulation_a": {
            "status": result["simulation_a"]["status"],
            "parameters": result["simulation_a"]["parameters"],
            "trajectory_rows": result["simulation_a"]["trajectory_rows"],
            "initial_state": result["simulation_a"]["initial_state"],
            "final_state": result["simulation_a"]["final_state"],
            "initial_norm": result["simulation_a"]["initial_norm"],
            "final_norm": result["simulation_a"]["final_norm"],
            "max_norm_error": result["simulation_a"]["max_norm_error"],
            "trajectory_csv": result["simulation_a"]["trajectory_csv"],
        },
        "simulation_b": {
            "status": result["simulation_b"]["status"],
            "parameters": result["simulation_b"]["parameters"],
            "trajectory_rows": result["simulation_b"]["trajectory_rows"],
            "initial_state": result["simulation_b"]["initial_state"],
            "final_state": result["simulation_b"]["final_state"],
            "initial_norm": result["simulation_b"]["initial_norm"],
            "final_norm": result["simulation_b"]["final_norm"],
            "max_norm_error": result["simulation_b"]["max_norm_error"],
            "trajectory_csv": result["simulation_b"]["trajectory_csv"],
        },
        "comparison": result["comparison"],
        "csv_paths": result["csv_paths"],
    }

    return json.dumps(compact_result, indent=2)
