"""Python interface to the Julia QuantumKickedTop classical DKT simulator."""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JULIA_PROJECT = PROJECT_ROOT / "QuantumKickedTop.jl"
JULIA_SCRIPT = PROJECT_ROOT / "scripts" / "julia" / "simulate_dkt.jl"
RESULTS_DIR = PROJECT_ROOT / "results" / "trajectories"


def _validate_finite(name: str, value: float) -> float:
    """Convert a parameter to float and reject NaN or infinity."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")

    return value


def simulate_dkt(
    kr: float,
    ktheta: float,
    theta0: float,
    phi0: float,
    nsteps: int,
    p: float = math.pi / 2,
    timeout: int = 300,
    include_trajectory: bool = False,
) -> dict[str, Any]:
    """Run a classical DKT trajectory using the Julia physics engine.

    Parameters
    ----------
    kr, ktheta:
        DKT parameters, with k = kr + ktheta and kprime = kr - ktheta.
    theta0, phi0:
        Initial spherical coordinates in radians.
    nsteps:
        Number of kicks. The output contains nsteps + 1 rows,
        including the initial state at step zero.
    p:
        Rotation angle. Defaults to pi/2.
    timeout:
        Maximum simulation time in seconds.
    include_trajectory:
        If True, include all CSV rows in the returned dictionary.
        Keep False for large trajectories.

    Returns
    -------
    dict
        Simulation parameters, numerical diagnostics, output CSV path,
        and optionally the complete trajectory.
    """
    kr = _validate_finite("kr", kr)
    ktheta = _validate_finite("ktheta", ktheta)
    theta0 = _validate_finite("theta0", theta0)
    phi0 = _validate_finite("phi0", phi0)
    p = _validate_finite("p", p)

    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 1:
        raise ValueError("nsteps must be a positive integer.")

    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer.")

    if not JULIA_PROJECT.is_dir():
        raise FileNotFoundError(f"Julia project not found: {JULIA_PROJECT}")

    if not JULIA_SCRIPT.is_file():
        raise FileNotFoundError(f"Julia simulation script not found: {JULIA_SCRIPT}")

    julia = os.environ.get("JULIA") or shutil.which("julia")
    if not julia:
        raise FileNotFoundError(
            "Julia executable was not found. Install Julia or set the JULIA "
            "environment variable to its executable path."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Each call receives its own output directory, avoiding filename collisions.
    run_dir = Path(tempfile.mkdtemp(prefix="dkt_run_", dir=RESULTS_DIR))
    output_csv = run_dir / "trajectory.csv"

    command = [
        julia,
        f"--project={JULIA_PROJECT}",
        str(JULIA_SCRIPT),
        str(kr),
        str(ktheta),
        str(theta0),
        str(phi0),
        str(nsteps),
        str(output_csv),
        str(p),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"DKT simulation exceeded the {timeout}-second timeout."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "Julia DKT simulation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    if not output_csv.is_file():
        raise RuntimeError(
            "Julia returned successfully but did not create the trajectory CSV.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    # Parse the Julia script's key=value diagnostic output.
    diagnostics: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            diagnostics[key.strip()] = value.strip()

    # Read and validate the trajectory CSV.
    trajectory: list[dict[str, float | int]] = []

    with output_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_columns = {"step", "X", "Y", "Z", "norm"}

        if not reader.fieldnames or not expected_columns.issubset(reader.fieldnames):
            raise RuntimeError(
                "Trajectory CSV has an unexpected header. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            trajectory.append(
                {
                    "step": int(row["step"]),
                    "X": float(row["X"]),
                    "Y": float(row["Y"]),
                    "Z": float(row["Z"]),
                    "norm": float(row["norm"]),
                }
            )

    expected_rows = nsteps + 1
    if len(trajectory) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} trajectory rows, "
            f"but found {len(trajectory)}."
        )

    result: dict[str, Any] = {
        "status": "success",
        "parameters": {
            "kr": kr,
            "ktheta": ktheta,
            "k": kr + ktheta,
            "kprime": kr - ktheta,
            "p": p,
            "theta0": theta0,
            "phi0": phi0,
            "nsteps": nsteps,
        },
        "trajectory_rows": len(trajectory),
        "initial_state": trajectory[0],
        "final_state": trajectory[-1],
        "initial_norm": trajectory[0]["norm"],
        "final_norm": trajectory[-1]["norm"],
        "max_norm_error": max(
            abs(point["norm"] - 1.0) for point in trajectory
        ),
        "trajectory_csv": str(output_csv),
        "julia_stdout": completed.stdout.strip(),
    }

    if include_trajectory:
        result["trajectory"] = trajectory

    return result



def compare_dkt_trajectories(
    trajectory_csv_1: str,
    trajectory_csv_2: str,
) -> dict[str, Any]:
    """
    Compare two classical DKT trajectories using their Cartesian coordinates.

    Both CSV files must contain the columns:
    step, X, Y, Z, norm

    The trajectories must have matching step indices.
    The function calculates the Euclidean separation at every kick
    and saves the comparison data to a CSV file.
    """

    def load_trajectory(csv_path: str) -> list[dict[str, float]]:
        path = Path(csv_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        trajectory = []

        with path.open("r", newline="") as file:
            reader = csv.DictReader(file)

            required = {"step", "X", "Y", "Z"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"CSV must contain columns: {sorted(required)}"
                )

            for row in reader:
                point = {
                    "step": int(row["step"]),
                    "X": float(row["X"]),
                    "Y": float(row["Y"]),
                    "Z": float(row["Z"]),
                }

                if not all(
                    math.isfinite(point[key])
                    for key in ("X", "Y", "Z")
                ):
                    raise ValueError("Trajectory contains non-finite values.")

                trajectory.append(point)

        if not trajectory:
            raise ValueError(f"Trajectory is empty: {path}")

        return trajectory

    trajectory_1 = load_trajectory(trajectory_csv_1)
    trajectory_2 = load_trajectory(trajectory_csv_2)

    steps_1 = [point["step"] for point in trajectory_1]
    steps_2 = [point["step"] for point in trajectory_2]

    if steps_1 != steps_2:
        raise ValueError(
            "Trajectories must have identical step indices "
            "and the same number of kicks."
        )

    distances = []
    comparison_rows = []

    for point_1, point_2 in zip(trajectory_1, trajectory_2):
        dx = point_1["X"] - point_2["X"]
        dy = point_1["Y"] - point_2["Y"]
        dz = point_1["Z"] - point_2["Z"]

        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(distance)

        comparison_rows.append({
            "step": point_1["step"],
            "delta_X": dx,
            "delta_Y": dy,
            "delta_Z": dz,
            "distance": distance,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_dir = Path(
        tempfile.mkdtemp(prefix="dkt_comparison_", dir=RESULTS_DIR)
    )
    output_csv = comparison_dir / "trajectory_comparison.csv"

    with output_csv.open("w", newline="") as file:
        fieldnames = [
            "step",
            "delta_X",
            "delta_Y",
            "delta_Z",
            "distance",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    return {
        "status": "success",
        "trajectory_1": str(Path(trajectory_csv_1).expanduser().resolve()),
        "trajectory_2": str(Path(trajectory_csv_2).expanduser().resolve()),
        "number_of_steps": len(distances) - 1,
        "comparison_rows": len(distances),
        "initial_separation": distances[0],
        "final_separation": distances[-1],
        "maximum_separation": max(distances),
        "mean_separation": sum(distances) / len(distances),
        "rms_separation": math.sqrt(
            sum(distance**2 for distance in distances) / len(distances)
        ),
        "comparison_csv": str(output_csv.resolve()),
    }


def run_dkt_comparison_study(
    kr: float = 1.0,
    ktheta_a: float = 0.0,
    ktheta_b: float = 10000.0,
    theta0: float = 0.75,
    phi0: float = -1.0,
    nsteps: int = 1000,
    p: float = math.pi / 2,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run two DKT simulations and compare their complete trajectories.

    Both simulations use the same initial condition, rotation angle,
    kr, and number of kicks. Only ktheta differs.

    Returns
    -------
    dict
        Complete results for both simulations, trajectory comparison
        statistics, and the three generated CSV paths.
    """

    # Run simulation A.
    simulation_a = simulate_dkt(
        kr=kr,
        ktheta=ktheta_a,
        theta0=theta0,
        phi0=phi0,
        nsteps=nsteps,
        p=p,
        timeout=timeout,
    )

    if simulation_a.get("status") != "success":
        raise RuntimeError("Simulation A did not complete successfully.")

    # Run simulation B.
    simulation_b = simulate_dkt(
        kr=kr,
        ktheta=ktheta_b,
        theta0=theta0,
        phi0=phi0,
        nsteps=nsteps,
        p=p,
        timeout=timeout,
    )

    if simulation_b.get("status") != "success":
        raise RuntimeError("Simulation B did not complete successfully.")

    # Compare the complete trajectories using their actual CSV files.
    comparison = compare_dkt_trajectories(
        trajectory_csv_1=simulation_a["trajectory_csv"],
        trajectory_csv_2=simulation_b["trajectory_csv"],
    )

    if comparison.get("status") != "success":
        raise RuntimeError("Trajectory comparison did not complete successfully.")

    return {
        "status": "success",
        "study_parameters": {
            "kr": kr,
            "ktheta_a": ktheta_a,
            "ktheta_b": ktheta_b,
            "theta0": theta0,
            "phi0": phi0,
            "nsteps": nsteps,
            "p": p,
        },
        "simulation_a": simulation_a,
        "simulation_b": simulation_b,
        "comparison": comparison,
        "csv_paths": {
            "simulation_a": simulation_a["trajectory_csv"],
            "simulation_b": simulation_b["trajectory_csv"],
            "comparison": comparison["comparison_csv"],
        },
    }
