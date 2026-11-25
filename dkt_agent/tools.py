"""Python interface to the Julia QuantumKickedTop classical DKT simulator."""

from __future__ import annotations

import csv
import json
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
    def _token(value: float) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p") or "0"

    output_csv = run_dir / (
        f"trajectory_kr_{_token(kr)}_ktheta_{_token(ktheta)}_"
        f"p_{_token(p)}_theta0_{_token(theta0)}_phi0_{_token(phi0)}_"
        f"nsteps_{nsteps}.csv"
    )

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

    metadata = {
        "schema_version": 1,
        "generator": "dkt_agent.tools.simulate_dkt",
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
        "trajectory_csv": str(output_csv.relative_to(PROJECT_ROOT)),
        "trajectory_rows": len(trajectory),
        "max_norm_error": max(
            abs(point["norm"] - 1.0) for point in trajectory
        ),
    }
    metadata_path = run_dir / "trajectory_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
        "trajectory_metadata": str(metadata_path),
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
    comparison_mode: str = "fixed_kr",
    kr_a: float | None = None,
    kr_b: float | None = None,
    ktheta_fixed: float | None = None,
) -> dict[str, Any]:
    """Compare two DKT trajectories while varying either kr or ktheta.

    fixed_kr: same kr, different ktheta values.
    fixed_ktheta: same ktheta, different kr values.
    """

    if comparison_mode == "fixed_kr":
        parameter_a = {"kr": kr, "ktheta": ktheta_a}
        parameter_b = {"kr": kr, "ktheta": ktheta_b}

    elif comparison_mode == "fixed_ktheta":
        if kr_a is None or kr_b is None or ktheta_fixed is None:
            raise ValueError(
                "fixed_ktheta mode requires kr_a, kr_b, and ktheta_fixed."
            )
        parameter_a = {"kr": kr_a, "ktheta": ktheta_fixed}
        parameter_b = {"kr": kr_b, "ktheta": ktheta_fixed}

    else:
        raise ValueError(
            "comparison_mode must be 'fixed_kr' or 'fixed_ktheta'."
        )

    simulation_a = simulate_dkt(
        kr=parameter_a["kr"],
        ktheta=parameter_a["ktheta"],
        theta0=theta0,
        phi0=phi0,
        nsteps=nsteps,
        p=p,
        timeout=timeout,
    )

    if simulation_a.get("status") != "success":
        raise RuntimeError("Simulation A did not complete successfully.")

    simulation_b = simulate_dkt(
        kr=parameter_b["kr"],
        ktheta=parameter_b["ktheta"],
        theta0=theta0,
        phi0=phi0,
        nsteps=nsteps,
        p=p,
        timeout=timeout,
    )

    if simulation_b.get("status") != "success":
        raise RuntimeError("Simulation B did not complete successfully.")

    comparison = compare_dkt_trajectories(
        trajectory_csv_1=simulation_a["trajectory_csv"],
        trajectory_csv_2=simulation_b["trajectory_csv"],
    )

    if comparison.get("status") != "success":
        raise RuntimeError("Trajectory comparison did not complete successfully.")

    return {
        "status": "success",
        "study_parameters": {
            "comparison_mode": comparison_mode,
            "kr": kr if comparison_mode == "fixed_kr" else None,
            "ktheta_a": ktheta_a if comparison_mode == "fixed_kr" else ktheta_fixed,
            "ktheta_b": ktheta_b if comparison_mode == "fixed_kr" else ktheta_fixed,
            "kr_a": parameter_a["kr"],
            "kr_b": parameter_b["kr"],
            "ktheta_fixed": ktheta_fixed,
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


ENTANGLEMENT_RESULTS_DIR = PROJECT_ROOT / "results" / "quantum_entanglement"
ENTANGLEMENT_DYNAMICS_SCRIPT = PROJECT_ROOT / "scripts" / "julia" / "run_entanglement_dynamics.jl"
ENTANGLEMENT_HEATMAP_SCRIPT = PROJECT_ROOT / "scripts" / "julia" / "run_entropy_heatmap.jl"


def _run_entanglement_julia(script: Path, arguments: list[str], timeout: int, p: float) -> tuple[Path, str]:
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer.")
    if not JULIA_PROJECT.is_dir():
        raise FileNotFoundError(f"Julia project not found: {JULIA_PROJECT}")
    if not script.is_file():
        raise FileNotFoundError(f"Julia script not found: {script}")
    julia = os.environ.get("JULIA") or shutil.which("julia")
    if not julia:
        raise FileNotFoundError("Julia executable was not found.")
    ENTANGLEMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="entanglement_", dir=ENTANGLEMENT_RESULTS_DIR))
    output = run_dir / "results.csv"
    command = [julia, f"--project={JULIA_PROJECT}", str(script), *arguments, str(output), str(p)]
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                                   text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Entanglement calculation exceeded {timeout} seconds.") from exc
    if completed.returncode != 0:
        raise RuntimeError("Julia entanglement calculation failed.\\n"
                           f"STDOUT:\\n{completed.stdout}\\nSTDERR:\\n{completed.stderr}")
    if not output.is_file():
        raise RuntimeError("Julia completed without creating the expected CSV.")
    return output, completed.stdout


def run_entanglement_dynamics(
    s: float, kr: float, ktheta: float, theta0: float, phi0: float,
    nsteps: int, rdm_dim: int, p: float = math.pi / 2, timeout: int = 600,
) -> dict[str, Any]:
    """Compute normalized von Neumann entanglement entropy at kicks 0 through nsteps."""
    s, kr, ktheta, theta0, phi0, p = (
        _validate_finite(name, value) for name, value in
        (("s", s), ("kr", kr), ("ktheta", ktheta), ("theta0", theta0), ("phi0", phi0), ("p", p))
    )
    if s <= 0 or not math.isclose(2*s, round(2*s), abs_tol=1e-12):
        raise ValueError("s must be a positive integer or half-integer.")
    N = round(2*s)
    if isinstance(rdm_dim, bool) or not isinstance(rdm_dim, int) or not 1 <= rdm_dim < N:
        raise ValueError("rdm_dim must be an integer satisfying 1 <= rdm_dim < 2s.")
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 0:
        raise ValueError("nsteps must be a non-negative integer.")
    if not 0 <= theta0 <= math.pi or not -math.pi <= phi0 <= math.pi:
        raise ValueError("theta0 must be in [0, pi] and phi0 in [-pi, pi].")
    output, stdout = _run_entanglement_julia(
        ENTANGLEMENT_DYNAMICS_SCRIPT,
        [str(s), str(kr), str(ktheta), str(theta0), str(phi0),
         str(nsteps), str(rdm_dim)], timeout, p)
    rows = []
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"kick", "normalized_entropy", "state_norm"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise RuntimeError(f"Unexpected entanglement CSV header: {reader.fieldnames}")
        for row in reader:
            item = {"kick": int(row["kick"]),
                    "normalized_entropy": float(row["normalized_entropy"]),
                    "state_norm": float(row["state_norm"])}
            if not all(math.isfinite(v) for v in item.values()):
                raise RuntimeError("Entanglement CSV contains non-finite values.")
            if not -1e-10 <= item["normalized_entropy"] <= 1 + 1e-10:
                raise RuntimeError("Normalized entropy lies outside [0, 1].")
            rows.append(item)
    if len(rows) != nsteps + 1 or [r["kick"] for r in rows] != list(range(nsteps + 1)):
        raise RuntimeError("Entropy time series must contain exactly kicks 0 through nsteps.")
    return {"status": "success", "parameters": {"s": s, "N": N, "kr": kr,
            "ktheta": ktheta, "k": kr+ktheta, "kprime": kr-ktheta,
            "theta0": theta0, "phi0": phi0, "p": p, "nsteps": nsteps,
            "rdm_dim": rdm_dim, "transient": "not applied (time series)"},
            "rows": len(rows), "initial_entropy": rows[0]["normalized_entropy"],
            "final_entropy": rows[-1]["normalized_entropy"],
            "max_norm_error": max(abs(r["state_norm"]-1) for r in rows),
            "csv_path": str(output), "data": rows, "julia_stdout": stdout.strip()}


def run_entanglement_heatmap(
    s: float, kr: float, ktheta: float, nsteps: int, transient: int,
    rdm_dim: int, grid_size: int = 15, p: float = math.pi/2,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Compute normalized long-time entropy averages on a coherent-state grid."""
    s, kr, ktheta, p = (_validate_finite(k, v) for k, v in
                        (("s", s), ("kr", kr), ("ktheta", ktheta), ("p", p)))
    if s <= 0 or not math.isclose(2*s, round(2*s), abs_tol=1e-12):
        raise ValueError("s must be a positive integer or half-integer.")
    N = round(2*s)
    if isinstance(rdm_dim, bool) or not isinstance(rdm_dim, int) or not 1 <= rdm_dim < N:
        raise ValueError("rdm_dim must be an integer satisfying 1 <= rdm_dim < 2s.")
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 2:
        raise ValueError("nsteps must be an integer >= 2.")
    if isinstance(transient, bool) or not isinstance(transient, int) or not 0 <= transient < nsteps:
        raise ValueError("transient must satisfy 0 <= transient < nsteps.")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 2 <= grid_size <= 100:
        raise ValueError("grid_size must be between 2 and 100.")
    output, stdout = _run_entanglement_julia(
        ENTANGLEMENT_HEATMAP_SCRIPT,
        [str(s), str(kr), str(ktheta), str(nsteps), str(transient),
         str(rdm_dim), str(grid_size)], timeout, p)
    rows = []
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"theta", "phi", "normalized_average_entropy",
                    "early_late_abs_difference", "status"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise RuntimeError(f"Unexpected heatmap CSV header: {reader.fieldnames}")
        for row in reader:
            item = {"theta": float(row["theta"]), "phi": float(row["phi"]),
                    "normalized_average_entropy": float(row["normalized_average_entropy"])
                    if row["normalized_average_entropy"] not in ("NaN", "nan", "") else math.nan,
                    "early_late_abs_difference": float(row["early_late_abs_difference"])
                    if row["early_late_abs_difference"] not in ("NaN", "nan", "") else math.nan,
                    "status": row["status"]}
            if not math.isfinite(item["theta"]) or not math.isfinite(item["phi"]):
                raise RuntimeError("Heatmap coordinates must be finite.")
            if item["status"] == "success" and not 0 <= item["normalized_average_entropy"] <= 1:
                raise RuntimeError("Successful heatmap entropy must lie in [0, 1].")
            rows.append(item)
    expected_rows = grid_size**2
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} heatmap cells, found {len(rows)}.")
    failed = sum(row["status"] != "success" for row in rows)
    return {"status": "success" if failed == 0 else "partial",
            "parameters": {"s": s, "N": N, "kr": kr, "ktheta": ktheta,
            "k": kr+ktheta, "kprime": kr-ktheta, "p": p, "nsteps": nsteps,
            "transient_discarded": transient, "averaging_kicks": f"{transient+1}:{nsteps}",
            "post_transient_samples": nsteps-transient, "rdm_dim": rdm_dim,
            "grid_size": grid_size},
            "grid_points": len(rows), "successful_points": len(rows)-failed,
            "failed_points": failed, "csv_path": str(output), "data": rows,
            "julia_stdout": stdout.strip()}


# Classical-quantum correspondence workflow.
CLASSICAL_QUANTUM_RESULTS_DIR = PROJECT_ROOT / "results" / "classical_quantum"
CLASSICAL_QUANTUM_SCRIPT = (
    PROJECT_ROOT / "scripts" / "julia" / "compare_classical_quantum.jl"
)


def run_classical_quantum_comparison(
    s: float,
    kr: float,
    ktheta: float,
    theta0: float,
    phi0: float,
    nsteps: int,
    p: float = math.pi / 2,
    timeout: int = 900,
) -> dict[str, Any]:
    """Compare classical X,Y,Z with quantum <Jx>/s,<Jy>/s,<Jz>/s.

    Both evolutions start from the same spherical orientation. The quantum
    initial condition is the package's spin-coherent state. Entanglement
    entropy is a separate observable and is not used as a phase-space
    coordinate in this comparison.
    """
    s, kr, ktheta, theta0, phi0, p = (
        _validate_finite(name, value)
        for name, value in (
            ("s", s), ("kr", kr), ("ktheta", ktheta),
            ("theta0", theta0), ("phi0", phi0), ("p", p),
        )
    )
    if s <= 0 or not math.isclose(2 * s, round(2 * s), rel_tol=0, abs_tol=1e-12):
        raise ValueError("s must be a positive integer or half-integer.")
    if not 0 <= theta0 <= math.pi:
        raise ValueError("theta0 must lie in [0, pi].")
    if not -math.pi <= phi0 <= math.pi:
        raise ValueError("phi0 must lie in [-pi, pi].")
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 0:
        raise ValueError("nsteps must be a non-negative integer.")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer.")
    if not JULIA_PROJECT.is_dir():
        raise FileNotFoundError(f"Julia project not found: {JULIA_PROJECT}")
    if not CLASSICAL_QUANTUM_SCRIPT.is_file():
        raise FileNotFoundError(f"Julia comparison script not found: {CLASSICAL_QUANTUM_SCRIPT}")
    julia = os.environ.get("JULIA") or shutil.which("julia")
    if not julia:
        raise FileNotFoundError("Julia executable was not found.")

    CLASSICAL_QUANTUM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(
        prefix="cq_comparison_", dir=CLASSICAL_QUANTUM_RESULTS_DIR
    ))
    output_csv = run_dir / "classical_quantum_comparison.csv"
    command = [
        julia, f"--project={JULIA_PROJECT}", str(CLASSICAL_QUANTUM_SCRIPT),
        str(s), str(kr), str(ktheta), str(theta0), str(phi0),
        str(nsteps), str(output_csv), str(p),
    ]
    try:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Classical-quantum comparison exceeded {timeout} seconds."
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "Julia classical-quantum comparison failed.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if not output_csv.is_file():
        raise RuntimeError("Julia completed without creating the comparison CSV.")

    expected = {
        "kick", "classical_X", "classical_Y", "classical_Z",
        "quantum_X", "quantum_Y", "quantum_Z",
        "delta_X", "delta_Y", "delta_Z", "classical_norm",
        "quantum_spin_norm", "state_norm",
    }
    rows: list[dict[str, float | int]] = []
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise RuntimeError(f"Unexpected comparison CSV header: {reader.fieldnames}")
        for row in reader:
            item: dict[str, float | int] = {"kick": int(row["kick"])}
            for key in expected - {"kick"}:
                item[key] = float(row[key])
            if not all(math.isfinite(float(value)) for value in item.values()):
                raise RuntimeError("Comparison CSV contains non-finite values.")
            rows.append(item)

    if len(rows) != nsteps + 1 or [r["kick"] for r in rows] != list(range(nsteps + 1)):
        raise RuntimeError(f"Expected exactly kicks 0 through {nsteps} in the CSV.")

    component_metrics: dict[str, dict[str, float]] = {}
    for component in ("X", "Y", "Z"):
        errors = [float(row[f"delta_{component}"]) for row in rows]
        component_metrics[component] = {
            "mae": sum(abs(value) for value in errors) / len(errors),
            "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "max_abs_error": max(abs(value) for value in errors),
        }

    vector_errors = [
        math.sqrt(sum(float(row[f"delta_{axis}"]) ** 2 for axis in ("X", "Y", "Z")))
        for row in rows
    ]
    stdout_values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            stdout_values[key.strip()] = value.strip()

    return {
        "status": "success",
        "parameters": {
            "s": s, "hilbert_dimension": int(round(2 * s)) + 1,
            "kr": kr, "ktheta": ktheta, "k": kr + ktheta,
            "kprime": kr - ktheta, "p": p,
            "theta0": theta0, "phi0": phi0, "nsteps": nsteps,
        },
        "rows": len(rows),
        "initial_alignment_error": float(stdout_values.get(
            "initial_alignment_error", vector_errors[0]
        )),
        "final_vector_difference": vector_errors[-1],
        "maximum_vector_difference": max(vector_errors),
        "mean_vector_difference": sum(vector_errors) / len(vector_errors),
        "vector_rmse": math.sqrt(sum(value * value for value in vector_errors) / len(vector_errors)),
        "component_metrics": component_metrics,
        "max_classical_norm_error": float(stdout_values.get(
            "max_classical_norm_error", max(abs(float(r["classical_norm"]) - 1) for r in rows)
        )),
        "max_quantum_state_norm_error": float(stdout_values.get(
            "max_state_norm_error", max(abs(float(r["state_norm"]) - 1) for r in rows)
        )),
        "csv_path": str(output_csv),
        "julia_stdout": completed.stdout.strip(),
    }
