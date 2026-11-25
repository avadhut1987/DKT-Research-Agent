"""LangChain wrapper for finite-data ergodic-hierarchy diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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


@tool
def diagnose_mixing_hierarchy(
    trajectory_csvs: list[str],
    transient: int = 0,
    max_lag: int = 1000,
    n_z: int = 2,
    n_phi: int = 4,
) -> str:
    """Run finite-data ergodicity-, weak-mixing-, and strong-mixing-related diagnostics.

    Requires at least two existing DKT trajectory CSVs generated from distinct
    initial conditions under the same map parameters. Reports selected-observable
    Birkhoff-average variance, Cesàro means of squared autocorrelations, and
    finite-partition alpha-coefficient estimates. No asymptotic classification
    or proof of ergodicity/mixing is made.
    """
    if not isinstance(trajectory_csvs, list) or not 2 <= len(trajectory_csvs) <= 24:
        raise ValueError("trajectory_csvs must contain 2–24 trajectory paths.")
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
        "strong_mixing_partition_alpha_csv": str(out / "strong_mixing_partition_alpha.csv"),
    }
    # Persist artifact paths in the JSON file itself.
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json.dumps(result, indent=2)
