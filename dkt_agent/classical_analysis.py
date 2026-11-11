"""Numerical diagnostics for distinguishing 1D and 2D DKT orbit support.

The module estimates:
  1. Correlation dimension from temporally admissible sampled pairs.
  2. Box-counting dimension using shifted Cartesian grids.
  3. Jensen-Shannon divergence between empirical measures on equal-area
     spherical partitions.

These are finite-data diagnostics, not mathematical proofs of ergodicity.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

def _validate_points(points: np.ndarray) -> np.ndarray:
    """Validate a finite array of three-dimensional trajectory points."""
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3).")
    if len(points) == 0:
        raise ValueError("points must not be empty.")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values.")

    return points


def _validate_output_label(label: str, name: str = "label") -> None:
    """Reject labels that could escape the output directory."""
    if (
        not isinstance(label, str)
        or not label
        or label in {".", ".."}
        or Path(label).name != label
        or "/" in label
        or "\\" in label
    ):
        raise ValueError(f"Invalid {name}: {label!r}")


def load_trajectory(path: str | Path) -> np.ndarray:
    """Load X,Y,Z columns from a DKT trajectory CSV."""
    data = np.genfromtxt(
        path, delimiter=",", names=True, dtype=float, encoding="utf-8"
    )
    points = np.column_stack((data["X"], data["Y"], data["Z"]))

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Unexpected trajectory shape in {path}: {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"Non-finite coordinates found in {path}")

    norms = np.linalg.norm(points, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-8:
        raise ValueError(
            f"Trajectory is not on the unit sphere: max norm error "
            f"{np.max(np.abs(norms - 1.0)):.3e}"
        )
    return points


def _sample_admissible_pair_distances(
    points: np.ndarray,
    theiler: int,
    n_pairs: int,
    seed: int,
) -> np.ndarray:
    """Sample pairs uniformly from index pairs with |i-j| > theiler."""
    points = _validate_points(points)
    n = len(points)

    if isinstance(n_pairs, bool) or not isinstance(n_pairs, (int, np.integer)):
        raise ValueError("n_pairs must be a positive integer.")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be a positive integer.")

    if isinstance(theiler, bool) or not isinstance(theiler, (int, np.integer)):
        raise ValueError("theiler must be a non-negative integer.")
    if n < 3:
        raise ValueError("At least three trajectory points are required.")
    if theiler < 0 or theiler >= n - 1:
        raise ValueError("Theiler window must be between 0 and n-2.")

    rng = np.random.default_rng(seed)
    distances = np.empty(n_pairs, dtype=float)
    filled = 0

    while filled < n_pairs:
        batch = max(4 * (n_pairs - filled), 10000)
        i = rng.integers(0, n, size=batch)
        j = rng.integers(0, n, size=batch)
        keep = np.abs(i - j) > theiler
        i, j = i[keep], j[keep]

        take = min(len(i), n_pairs - filled)
        if take == 0:
            continue

        delta = points[i[:take]] - points[j[:take]]
        distances[filled:filled + take] = np.linalg.norm(delta, axis=1)
        filled += take

    return distances


def correlation_dimension(
    points: np.ndarray,
    *,
    theiler: int = 20,
    n_pairs: int = 1_000_000,
    seed: int = 1234,
    radii: np.ndarray | None = None,
    fit_c_range: tuple[float, float] = (1e-3, 0.1),
) -> dict:
    """Estimate D2 using a sampled correlation integral.

    C(r) is estimated over pairs whose temporal separation exceeds the
    Theiler window. The returned slope is a fit over the specified C(r)
    range; inspect the full curve and local slopes before interpreting it.
    """
    points = _validate_points(points)

    if radii is None:
        radii = np.logspace(-4, np.log10(1.5), 48)
    radii = np.asarray(radii, dtype=float)

    if radii.ndim != 1 or len(radii) < 1:
        raise ValueError("radii must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("radii must be finite and positive.")
    if np.any(np.diff(radii) <= 0):
        raise ValueError("radii must be strictly increasing.")

    if len(fit_c_range) != 2:
        raise ValueError("fit_c_range must contain two values.")

    c_min, c_max = map(float, fit_c_range)
    if (
        not np.isfinite(c_min)
        or not np.isfinite(c_max)
        or c_min <= 0
        or c_max > 1
        or c_min >= c_max
    ):
        raise ValueError("fit_c_range must satisfy 0 < min < max <= 1.")

    distances = _sample_admissible_pair_distances(
        points, theiler, n_pairs, seed
    )
    distances.sort()

    counts = np.searchsorted(distances, radii, side="right")
    corr = counts / len(distances)

    valid = (
        (corr >= c_min)
        & (corr <= c_max)
        & (corr > 0)
        & np.isfinite(corr)
    )

    fit = None
    if np.count_nonzero(valid) >= 4:
        slope, intercept = np.polyfit(
            np.log(radii[valid]), np.log(corr[valid]), 1
        )
        predicted = slope * np.log(radii[valid]) + intercept
        residual = np.log(corr[valid]) - predicted
        ss_res = np.sum(residual**2)
        ss_tot = np.sum(
            (np.log(corr[valid]) - np.mean(np.log(corr[valid])))**2
        )
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        fit = {
            "D2": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "n_fit_points": int(np.count_nonzero(valid)),
            "fit_c_range": [c_min, c_max],
        }

    # Five-point local slopes help reveal whether a stable scaling region exists.
    local_slopes = np.full(len(radii), np.nan)
    log_r = np.log(radii)
    log_c = np.full(len(corr), np.nan)
    positive = corr > 0
    log_c[positive] = np.log(corr[positive])

    for i in range(2, len(radii) - 2):
        idx = slice(i - 2, i + 3)
        if np.all(np.isfinite(log_c[idx])):
            local_slopes[i] = np.polyfit(log_r[idx], log_c[idx], 1)[0]

    return {
        "radii": radii,
        "correlation_integral": corr,
        "local_slopes": local_slopes,
        "fit": fit,
        "theiler": int(theiler),
        "n_pairs": int(n_pairs),
    }


def occupied_cell_scaling(
    points: np.ndarray,
    *,
    epsilons: np.ndarray | None = None,
    n_shifts: int = 8,
    seed: int = 1234,
    fit_count_range: tuple[int, float] = (10, 0.5),
) -> dict:
    """Estimate box-counting scaling using shifted 3D Cartesian grids."""
    points = _validate_points(points)
    n = len(points)

    if isinstance(n_shifts, bool) or not isinstance(n_shifts, (int, np.integer)):
        raise ValueError("n_shifts must be a positive integer.")
    if n_shifts <= 0:
        raise ValueError("n_shifts must be a positive integer.")

    if epsilons is None:
        epsilons = np.logspace(np.log10(0.005), np.log10(0.5), 24)
    epsilons = np.asarray(epsilons, dtype=float)

    if epsilons.ndim != 1 or len(epsilons) < 1:
        raise ValueError("epsilons must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(epsilons)) or np.any(epsilons <= 0):
        raise ValueError("epsilons must be finite and positive.")
    if np.any(np.diff(epsilons) <= 0):
        raise ValueError("epsilons must be strictly increasing.")

    if len(fit_count_range) != 2:
        raise ValueError("fit_count_range must contain two values.")

    lower_count, upper_fraction = map(float, fit_count_range)
    if (
        not np.isfinite(lower_count)
        or not np.isfinite(upper_fraction)
        or lower_count <= 0
        or upper_fraction <= 0
        or upper_fraction > 1
    ):
        raise ValueError(
            "fit_count_range must satisfy lower > 0 and 0 < upper_fraction <= 1."
        )
    rng = np.random.default_rng(seed)
    mean_counts = []

    for eps in epsilons:
        counts = []
        for _ in range(n_shifts):
            # Independent offsets reduce sensitivity to grid placement.
            offset = rng.uniform(0.0, eps, size=3)
            cells = np.floor((points + 1.0 + offset) / eps).astype(np.int64)
            counts.append(len(np.unique(cells, axis=0)))
        mean_counts.append(np.mean(counts))

    mean_counts = np.asarray(mean_counts)

    upper_count = upper_fraction * n

    valid = (
        (mean_counts >= lower_count)
        & (mean_counts <= upper_count)
        & (mean_counts > 0)
    )

    fit = None
    if np.count_nonzero(valid) >= 4:
        # N(eps) ~ eps^(-D); slope of log N versus log(1/eps) is D.
        slope, intercept = np.polyfit(
            np.log(1.0 / epsilons[valid]),
            np.log(mean_counts[valid]),
            1,
        )
        predicted = (
            slope * np.log(1.0 / epsilons[valid]) + intercept
        )
        observed = np.log(mean_counts[valid])
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        fit = {
            "D_box": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "n_fit_points": int(np.count_nonzero(valid)),
            "fit_count_range": [lower_count, upper_count],
        }

    return {
        "epsilons": epsilons,
        "mean_occupied_cells": mean_counts,
        "fit": fit,
        "n_shifts": int(n_shifts),
    }


def equal_area_measure(
    points: np.ndarray,
    *,
    n_z: int = 40,
    n_phi: int = 80,
) -> np.ndarray:
    """Empirical probability on equal-area (uniform z, uniform phi) cells."""
    points = _validate_points(points)

    for name, value in (("n_z", n_z), ("n_phi", n_phi)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    z = np.clip(points[:, 2], -1.0, 1.0)

    phi = np.arctan2(points[:, 1], points[:, 0])

    hist, _, _ = np.histogram2d(
        z,
        phi,
        bins=(n_z, n_phi),
        range=((-1.0, 1.0), (-np.pi, np.pi)),
    )
    prob = hist.ravel().astype(float)
    total = prob.sum()
    if total == 0:
        raise ValueError("Empty empirical measure.")
    return prob / total


def empirical_measure_comparison(
    trajectories: dict[str, np.ndarray],
    *,
    n_z: int = 40,
    n_phi: int = 80,
) -> dict:
    """Pairwise Jensen-Shannon divergences between trajectory measures."""
    labels = list(trajectories)
    measures = {
        label: equal_area_measure(
            trajectories[label], n_z=n_z, n_phi=n_phi
        )
        for label in labels
    }

    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            b = labels[j]
            # Squared Jensen-Shannon distance is the JS divergence.
            value = jensenshannon(
                measures[a], measures[b], base=2.0
            ) ** 2
            matrix[i, j] = matrix[j, i] = value

    return {
        "labels": labels,
        "js_divergence": matrix,
        "n_z": int(n_z),
        "n_phi": int(n_phi),
        "measures": measures,
    }


def _save_trajectory_plots(
    label: str,
    corr: dict,
    box: dict,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    r = corr["radii"]
    c = corr["correlation_integral"]
    mask = c > 0
    axes[0].loglog(r[mask], c[mask], "o-", ms=3)
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("C(r)")
    axes[0].set_title(f"Correlation integral: {label}")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].semilogx(
        r, corr["local_slopes"], "o-", ms=3, label="local slope"
    )
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("local D2")
    axes[1].set_title("Local correlation-dimension slope")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_correlation_dimension.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(
        1.0 / box["epsilons"],
        box["mean_occupied_cells"],
        "o-",
        ms=4,
    )
    ax.set_xlabel("1 / cell width")
    ax.set_ylabel("Mean occupied cells")
    ax.set_title(f"Occupied-cell scaling: {label}")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_box_scaling.png", dpi=180)
    plt.close(fig)


def run_ergodicity_diagnostics(
    trajectory_files: dict[str, str | Path],
    output_dir: str | Path,
    *,
    theiler: int = 20,
    n_pairs: int = 1_000_000,
    seed: int = 1234,
    n_z: int = 40,
    n_phi: int = 80,
    measure_groups: dict[str, list[str]] | None = None,
) -> dict:
    """Run geometric diagnostics and optional within-parameter measure comparisons.

    measure_groups maps a parameter-group label to trajectory labels.
    Only trajectories explicitly grouped together are compared.
    """
    if not trajectory_files:
        raise ValueError("trajectory_files must not be empty.")

    for label in trajectory_files:
        _validate_output_label(label, "trajectory label")

    if measure_groups is not None:
        for group, labels in measure_groups.items():
            _validate_output_label(group, "measure-group label")

            if not isinstance(labels, list) or not labels:
                raise ValueError(
                    f"Measure group {group!r} must contain a non-empty list."
                )

            if len(labels) != len(set(labels)):
                raise ValueError(
                    f"Measure group {group!r} contains duplicate labels."
                )

            for label in labels:
                if label not in trajectory_files:
                    raise KeyError(
                        f"Unknown trajectory label in group {group!r}: {label!r}"
                    )

    trajectories = {
        label: load_trajectory(path)
        for label, path in trajectory_files.items()
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"trajectories": {}, "parameters": {
        "theiler": theiler,
        "n_pairs": n_pairs,
        "seed": seed,
        "n_z": n_z,
        "n_phi": n_phi,
    }}

    for index, (label, points) in enumerate(trajectories.items()):
        print(f"Analysing {label}: {len(points)} points")

        corr = correlation_dimension(
            points,
            theiler=theiler,
            n_pairs=n_pairs,
            seed=seed + index,
        )
        box = occupied_cell_scaling(
            points,
            seed=seed + index,
        )

        results["trajectories"][label] = {
            "n_points": len(points),
            "correlation_dimension": corr["fit"],
            "box_dimension": box["fit"],
        }
        _save_trajectory_plots(label, corr, box, output_dir)

        np.savetxt(
            output_dir / f"{label}_correlation_curve.csv",
            np.column_stack((
                corr["radii"],
                corr["correlation_integral"],
                corr["local_slopes"],
            )),
            delimiter=",",
            header="radius,C_r,local_slope",
            comments="",
        )
        np.savetxt(
            output_dir / f"{label}_box_curve.csv",
            np.column_stack((
                box["epsilons"],
                box["mean_occupied_cells"],
            )),
            delimiter=",",
            header="epsilon,mean_occupied_cells",
            comments="",
        )

    results["empirical_measure_comparison"] = {}

    if measure_groups is not None:
        for group, labels in measure_groups.items():
            missing = [label for label in labels if label not in trajectories]
            if missing:
                raise KeyError(
                    f"Unknown trajectory labels in group {group}: {missing}"
                )

            group_trajectories = {
                label: trajectories[label] for label in labels
            }
            measures = empirical_measure_comparison(
                group_trajectories, n_z=n_z, n_phi=n_phi
            )

            results["empirical_measure_comparison"][group] = {
                "labels": measures["labels"],
                "js_divergence": measures["js_divergence"].tolist(),
                "n_z": measures["n_z"],
                "n_phi": measures["n_phi"],
            }

            np.savetxt(
                output_dir / f"{group}_empirical_measure_js.csv",
                measures["js_divergence"],
                delimiter=",",
                header="Pairwise Jensen-Shannon divergence; labels: "
                + ", ".join(measures["labels"]),
                comments="",
            )

    with open(output_dir / "diagnostics_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
