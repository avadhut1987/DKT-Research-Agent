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
FIXED_POINTS_SCRIPT = (
    PROJECT_ROOT / "scripts" / "julia" / "analyze_dkt_fixed_points.jl"
)
FIXED_POINTS_RESULTS_DIR = PROJECT_ROOT / "results" / "fixed_points"


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
        kr controls the degree of classical chaos. ktheta changes
        phase-space geometry but does not change the degree of chaos.
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



def simulate_dkt_grid(
    kr: float,
    ktheta: float,
    grid_size: int = 15,
    nsteps: int = 120,
    p: float = math.pi / 2,
    timeout: int = 300,
) -> dict[str, Any]:
    """Evolve a grid of DKT initial conditions and save every orbit point."""

    kr = _validate_finite("kr", kr)
    ktheta = _validate_finite("ktheta", ktheta)
    p = _validate_finite("p", p)

    if isinstance(grid_size, bool) or not isinstance(grid_size, int):
        raise ValueError("grid_size must be an integer between 5 and 50.")
    if not 5 <= grid_size <= 50:
        raise ValueError("grid_size must be between 5 and 50.")

    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 1:
        raise ValueError("nsteps must be a positive integer.")

    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer.")

    if not JULIA_PROJECT.is_dir():
        raise FileNotFoundError(f"Julia project not found: {JULIA_PROJECT}")

    grid_script = PROJECT_ROOT / "scripts" / "julia" / "simulate_dkt_grid.jl"
    if not grid_script.is_file():
        raise FileNotFoundError(
            f"Julia grid simulation script not found: {grid_script}"
        )

    julia = os.environ.get("JULIA") or shutil.which("julia")
    if not julia:
        raise FileNotFoundError(
            "Julia executable was not found. Install Julia or set the JULIA "
            "environment variable to its executable path."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="dkt_grid_", dir=RESULTS_DIR))
    output_csv = run_dir / "phase_space.csv"

    command = [
        julia,
        f"--project={JULIA_PROJECT}",
        str(grid_script),
        str(kr),
        str(ktheta),
        str(p),
        str(grid_size),
        str(nsteps),
        str(output_csv),
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
            f"DKT grid simulation exceeded the {timeout}-second timeout."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "Julia DKT grid simulation failed.\\n"
            f"Command: {' '.join(command)}\\n"
            f"STDOUT:\\n{completed.stdout}\\n"
            f"STDERR:\\n{completed.stderr}"
        )

    if not output_csv.is_file():
        raise RuntimeError(
            "Julia returned successfully but did not create the grid CSV.\\n"
            f"STDOUT:\\n{completed.stdout}\\n"
            f"STDERR:\\n{completed.stderr}"
        )

    expected_columns = {
        "grid_i", "grid_j", "kick", "theta0", "phi0",
        "theta", "phi", "X", "Y", "Z", "norm",
    }

    orbit_points: list[dict[str, float | int]] = []
    seen_samples: set[tuple[int, int, int]] = set()

    with output_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames or not expected_columns.issubset(
            reader.fieldnames
        ):
            raise RuntimeError(
                "Grid CSV has an unexpected header. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            point: dict[str, float | int] = {
                "grid_i": int(row["grid_i"]),
                "grid_j": int(row["grid_j"]),
                "kick": int(row["kick"]),
                "theta0": float(row["theta0"]),
                "phi0": float(row["phi0"]),
                "theta": float(row["theta"]),
                "phi": float(row["phi"]),
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "Z": float(row["Z"]),
                "norm": float(row["norm"]),
            }

            numeric_values = [
                value for key, value in point.items()
                if key not in ("grid_i", "grid_j", "kick")
            ]
            if not all(math.isfinite(value) for value in numeric_values):
                raise RuntimeError("Grid CSV contains non-finite values.")

            i = int(point["grid_i"])
            j = int(point["grid_j"])
            kick = int(point["kick"])

            if not (1 <= i <= grid_size and 1 <= j <= grid_size):
                raise RuntimeError(f"Invalid grid index: ({i}, {j}).")
            if not (0 <= kick <= nsteps):
                raise RuntimeError(f"Invalid kick index: {kick}.")

            sample_id = (i, j, kick)
            if sample_id in seen_samples:
                raise RuntimeError(f"Duplicate orbit sample: {sample_id}.")

            seen_samples.add(sample_id)
            orbit_points.append(point)

    expected_samples = grid_size**2 * (nsteps + 1)
    if len(orbit_points) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} orbit samples, "
            f"but found {len(orbit_points)}."
        )

    max_norm_error = max(
        abs(float(point["norm"]) - 1.0) for point in orbit_points
    )

    return {
        "status": "success",
        "parameters": {
            "kr": kr,
            "ktheta": ktheta,
            "k": kr + ktheta,
            "kprime": kr - ktheta,
            "p": p,
            "grid_size": grid_size,
            "initial_conditions": grid_size**2,
            "nsteps": nsteps,
        },
        "grid_points": len(orbit_points),
        "max_norm_error": max_norm_error,
        "phase_space_csv": str(output_csv),
        "orbit_points": orbit_points,
        "julia_stdout": completed.stdout.strip(),
    }



def find_dkt_fixed_points(
    kr: float,
    ktheta: float = 0.0,
    p: float = math.pi / 2,
    nseeds: int = 200,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    timeout: int = 300,
) -> dict[str, Any]:
    """Find DKT fixed points and classify their local linear stability.

    This is a finite numerical seed search, not a proof of completeness.
    """
    kr = _validate_finite("kr", kr)
    ktheta = _validate_finite("ktheta", ktheta)
    p = _validate_finite("p", p)
    tolerance = _validate_finite("tolerance", tolerance)

    for name, value, minimum in (
        ("nseeds", nseeds, 4),
        ("max_iterations", max_iterations, 1),
        ("timeout", timeout, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    if not JULIA_PROJECT.is_dir():
        raise FileNotFoundError(f"Julia project not found: {JULIA_PROJECT}")
    if not FIXED_POINTS_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Julia fixed-point script not found: {FIXED_POINTS_SCRIPT}"
        )

    julia = os.environ.get("JULIA") or shutil.which("julia")
    if not julia:
        raise FileNotFoundError(
            "Julia executable was not found. Install Julia or set JULIA."
        )

    FIXED_POINTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(
        prefix="dkt_fixed_points_", dir=FIXED_POINTS_RESULTS_DIR
    ))
    output_csv = run_dir / "fixed_points.csv"

    command = [
        julia,
        f"--project={JULIA_PROJECT}",
        str(FIXED_POINTS_SCRIPT),
        str(kr),
        str(ktheta),
        str(p),
        str(nseeds),
        str(tolerance),
        str(max_iterations),
        str(output_csv),
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
            f"Fixed-point analysis exceeded the {timeout}-second timeout."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "Julia fixed-point analysis failed.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if not output_csv.is_file():
        raise RuntimeError(
            "Julia did not create the fixed-point CSV.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    diagnostics = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            diagnostics[key.strip()] = value.strip()

    expected = {
        "fixed_point", "X", "Y", "Z", "residual", "iterations",
        "stability", "lambda1_real", "lambda1_imag",
        "lambda2_real", "lambda2_imag", "determinant",
    }
    float_columns = (
        "X", "Y", "Z", "residual", "lambda1_real", "lambda1_imag",
        "lambda2_real", "lambda2_imag", "determinant",
    )
    fixed_points = []

    with output_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise RuntimeError(
                f"Unexpected fixed-point CSV header: {reader.fieldnames}"
            )

        for row in reader:
            try:
                point = {
                    "fixed_point": int(row["fixed_point"]),
                    "iterations": int(row["iterations"]),
                    "stability": row["stability"],
                }
                point.update({key: float(row[key]) for key in float_columns})
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError("Invalid row in fixed-point CSV.") from exc

            if not all(math.isfinite(point[key]) for key in float_columns):
                raise RuntimeError("Fixed-point CSV contains non-finite values.")

            norm = math.sqrt(
                point["X"]**2 + point["Y"]**2 + point["Z"]**2
            )
            if abs(norm - 1.0) > 1e-7:
                raise RuntimeError(
                    f"Fixed point {point['fixed_point']} is not on the unit sphere."
                )
            if not point["stability"]:
                raise RuntimeError("Empty stability label in fixed-point CSV.")

            fixed_points.append(point)

    if not fixed_points:
        raise RuntimeError("No validated fixed points were returned.")

    return {
        "status": "success",
        "parameters": {
            "kr": kr,
            "ktheta": ktheta,
            "k": kr + ktheta,
            "kprime": kr - ktheta,
            "p": p,
            "nseeds": nseeds,
            "tolerance": tolerance,
            "max_iterations": max_iterations,
        },
        "converged_seeds": int(diagnostics.get("converged_seeds", "0")),
        "unique_fixed_points": len(fixed_points),
        "fixed_points": fixed_points,
        "fixed_points_csv": str(output_csv),
        "julia_stdout": completed.stdout.strip(),
    }


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
    The comparison measures trajectory and phase-space geometric
    differences, not a change in the degree of classical chaos.

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
