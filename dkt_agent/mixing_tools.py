"""LangChain wrapper for finite-data ergodic-hierarchy diagnostics."""

from __future__ import annotations

import csv
import json
import csv
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dkt_agent.tools import simulate_dkt_grid

from langchain_core.tools import tool

from dkt_agent.classical_analysis import load_trajectory
from dkt_agent.mixing_hierarchy import run_mixing_hierarchy_from_trajectories

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = (PROJECT_ROOT / "results" / "trajectories").resolve()
DIAGNOSTICS_ROOT = PROJECT_ROOT / "results" / "diagnostics"


def _resolve_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each trajectory_csvs entry must be a non-empty path.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(TRAJECTORY_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Only CSV files inside results/trajectories/ can be analysed. "
            "Use the exact path returned by a DKT simulation."
        ) from exc
    if not resolved.is_file() or resolved.suffix.lower() != ".csv":
        raise ValueError("Each trajectory must be an existing CSV file.")
    return resolved


def _number_token(value: float) -> str:
    """Create a readable, filesystem-safe numerical filename token."""
    token = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if token in ("", "-0"):
        token = "0"
    return token.replace("-", "m").replace(".", "p")


def generate_dkt_ensemble(
    kr: float,
    ktheta: float = 0.0,
    p: float = 1.5707963267948966,
    grid_size: int = 5,
    nsteps: int = 1000,
) -> dict:
    """Generate, split, and persist a same-map ensemble from the Julia grid simulator.

    The grid simulator evolves distinct angular-grid initial conditions under
    one common DKT map. Each orbit is saved as a parameter-labelled CSV and
    an ensemble manifest records the exact map parameters and initial state.
    """
    for name, value in (("kr", kr), ("ktheta", ktheta), ("p", p)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if name == "kr":
            kr = value
        elif name == "ktheta":
            ktheta = value
        else:
            p = value

    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 5 <= grid_size <= 8:
        raise ValueError("grid_size must be an integer from 5 to 8 (25–64 trajectories).")
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or not 1 <= nsteps <= 5_000:
        raise ValueError("nsteps must be an integer from 1 to 5,000.")

    generated = simulate_dkt_grid(
        kr=kr, ktheta=ktheta, p=p, grid_size=grid_size, nsteps=nsteps
    )
    orbit_points = generated.get("orbit_points", [])
    expected_trajectories = grid_size * grid_size
    expected_rows = nsteps + 1
    grouped: dict[tuple[int, int], list[dict]] = {}
    for point in orbit_points:
        key = (int(point["grid_i"]), int(point["grid_j"]))
        grouped.setdefault(key, []).append(point)

    if len(grouped) != expected_trajectories:
        raise RuntimeError(
            f"Expected {expected_trajectories} distinct initial conditions; "
            f"received {len(grouped)}."
        )

    TRAJECTORY_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"ensemble_kr_{_number_token(kr)}_ktheta_{_number_token(ktheta)}_"
        f"p_{_number_token(p)}_grid_{grid_size}_nsteps_{nsteps}_"
    )
    ensemble_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=TRAJECTORY_ROOT))
    trajectory_records = []

    for (grid_i, grid_j), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["kick"]))
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"Grid point ({grid_i}, {grid_j}) has {len(rows)} rows; "
                f"expected {expected_rows}."
            )
        theta0 = float(rows[0]["theta0"])
        phi0 = float(rows[0]["phi0"])
        filename = (
            f"trajectory_{grid_i:02d}_{grid_j:02d}_"
            f"kr_{_number_token(kr)}_ktheta_{_number_token(ktheta)}_"
            f"p_{_number_token(p)}_theta0_{_number_token(theta0)}_"
            f"phi0_{_number_token(phi0)}_nsteps_{nsteps}.csv"
        )
        output_path = ensemble_dir / filename
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["step", "X", "Y", "Z", "norm"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "step": int(row["kick"]),
                    "X": float(row["X"]),
                    "Y": float(row["Y"]),
                    "Z": float(row["Z"]),
                    "norm": float(row["norm"]),
                })
        trajectory_records.append({
            "grid_i": grid_i,
            "grid_j": grid_j,
            "theta0": theta0,
            "phi0": phi0,
            "file": str(output_path.relative_to(PROJECT_ROOT)),
            "rows": len(rows),
        })

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "QuantumKickedTop.jl simulate_dkt_grid.jl",
        "sampling": "midpoint angular grid in theta and uniform grid in phi",
        "parameters": {
            "kr": kr,
            "ktheta": ktheta,
            "k": kr + ktheta,
            "kprime": kr - ktheta,
            "p": p,
            "grid_size": grid_size,
            "n_trajectories": expected_trajectories,
            "nsteps": nsteps,
        },
        "raw_grid_csv": generated.get("phase_space_csv"),
        "trajectory_files": trajectory_records,
        "max_norm_error": generated.get("max_norm_error"),
    }
    manifest_path = ensemble_dir / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "status": "success",
        "manifest_path": str(manifest_path),
        "ensemble_directory": str(ensemble_dir),
        "n_trajectories": expected_trajectories,
        "nsteps": nsteps,
        "max_norm_error": generated.get("max_norm_error"),
        "parameters": manifest["parameters"],
        "trajectory_files": [record["file"] for record in trajectory_records],
    }


def discover_saved_ensembles() -> dict[str, dict]:
    """Read saved ensemble manifests, newest first, without guessing CSV provenance."""
    ensembles: dict[str, dict] = {}
    if not TRAJECTORY_ROOT.is_dir():
        return ensembles
    manifests = sorted(
        TRAJECTORY_ROOT.glob("ensemble_*/ensemble_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest["trajectory_files"]
            parameters = manifest["parameters"]
            if len(records) < 2 or not all(
                (PROJECT_ROOT / record["file"]).is_file() for record in records
            ):
                continue
            label = (
                f"{manifest_path.parent.name} · "
                f"kr={parameters['kr']:g}, kθ={parameters['ktheta']:g}, "
                f"p={parameters['p']:g} · {len(records)} trajectories"
            )
            ensembles[label] = {
                "manifest": manifest,
                "manifest_path": str(manifest_path),
            }
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return ensembles


@tool
def diagnose_mixing_hierarchy(
    trajectory_csvs: list[str],
    transient: int = 0,
    max_lag: int = 1000,
    n_z: int = 2,
    n_phi: int = 4,
) -> str:
    """Run finite-data ergodicity-, weak-mixing-, and strong-mixing-related diagnostics.

    Requires at least two existing DKT trajectory CSVs from distinct initial
    conditions. The caller must ensure all files use identical map parameters;
    parameter metadata are not inferred from trajectory coordinates. Reports selected-observable
    Birkhoff-average variance, Cesàro means of squared autocorrelations, and
    finite-partition alpha-coefficient estimates. No asymptotic classification
    or proof of ergodicity/mixing is made.
    """
    if not isinstance(trajectory_csvs, list) or not 2 <= len(trajectory_csvs) <= 64:
        raise ValueError("trajectory_csvs must contain 2–64 trajectory paths.")
    paths = [_resolve_path(value) for value in trajectory_csvs]
    if len(set(paths)) != len(paths):
        raise ValueError("trajectory_csvs must refer to distinct files.")
    if isinstance(transient, bool) or not isinstance(transient, int) or transient < 0:
        raise ValueError("transient must be a non-negative integer.")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or not 2 <= max_lag <= 5000:
        raise ValueError("max_lag must be between 2 and 5000.")
    if isinstance(n_z, bool) or not isinstance(n_z, int) or n_z < 1:
        raise ValueError("n_z must be a positive integer.")
    if isinstance(n_phi, bool) or not isinstance(n_phi, int) or n_phi < 1:
        raise ValueError("n_phi must be a positive integer.")
    if n_z * n_phi > 16:
        raise ValueError("n_z*n_phi must not exceed 16 (exact alpha calculation).")

    trajectories = [load_trajectory(path) for path in paths]
    result = run_mixing_hierarchy_from_trajectories(
        trajectories,
        transient=transient,
        max_lag=max_lag,
        n_z=n_z,
        n_phi=n_phi,
    )
    DIAGNOSTICS_ROOT.mkdir(parents=True, exist_ok=True)
    out = DIAGNOSTICS_ROOT / "mixing_hierarchy"
    out.mkdir(parents=True, exist_ok=True)
    # Keep each run isolated and never overwrite previous research artifacts.
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="finite_hierarchy_", dir=out))
    result["input_trajectory_csvs"] = [str(path) for path in paths]
    result["parameters"] = {
        "transient_discarded": transient,
        "max_lag": max_lag,
        "n_z": n_z,
        "n_phi": n_phi,
    }
    diag = result["finite_data_diagnostics"]
    ergodic = diag["ergodicity_related"]
    with (out / "birkhoff_variance.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(ergodic["rows"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ergodic["rows"])
    with (out / "weak_mixing_cesaro.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["observable", "lag", "normalized_autocorrelation", "cesaro_mean_squared_autocorrelation"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in diag["weak_mixing_related"]["observables"].items():
            correlations = values["normalized_autocorrelation"]
            cesaro = values["cesaro_mean_squared_autocorrelation"]
            for lag, correlation in enumerate(correlations):
                writer.writerow({
                    "observable": name,
                    "lag": lag,
                    "normalized_autocorrelation": correlation,
                    "cesaro_mean_squared_autocorrelation": (None if lag == 0 else cesaro[lag - 1]),
                })
    with (out / "strong_mixing_partition_alpha.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = diag["strong_mixing_related"]["lags"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = out / "mixing_hierarchy_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["artifacts"] = {
        "directory": str(out),
        "summary_json": str(summary_path),
        "birkhoff_variance_csv": str(out / "birkhoff_variance.csv"),
        "weak_mixing_cesaro_csv": str(out / "weak_mixing_cesaro.csv"),
        "strong_mixing_partition_alpha_csv": str(out / "strong_mixing_partition_alpha.csv"),
    }
    # Persist artifact paths in the JSON file itself.
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json.dumps(result, indent=2)
