"""Finite-sample diagnostics for the ergodic hierarchy.

This module deliberately reports numerical diagnostics rather than classifying a
finite data set as ergodic, weakly mixing, or strongly mixing.  In particular,
strong mixing is assessed only on a specified finite partition and an ensemble
of trajectories; no finite computation proves an asymptotic property.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

OBSERVABLE_NAMES = ("X", "Y", "Z", "X2", "Y2", "Z2", "XY", "YZ", "ZX")
AREA_MEANS = np.array(
    [0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
    dtype=float,
)


def _validate_ensemble(trajectories: Iterable[np.ndarray]) -> list[np.ndarray]:
    ensemble = [np.asarray(item, dtype=float) for item in trajectories]
    if len(ensemble) < 2:
        raise ValueError("At least two trajectories from distinct initial conditions are required.")
    for index, points in enumerate(ensemble):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
            raise ValueError(f"Trajectory {index + 1} must have shape (T, 3), T >= 4.")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"Trajectory {index + 1} contains non-finite coordinates.")
        norms = np.linalg.norm(points, axis=1)
        if np.max(np.abs(norms - 1.0)) > 1e-6:
            raise ValueError(f"Trajectory {index + 1} is not on the unit sphere.")
    return ensemble


def cartesian_observables(points: np.ndarray) -> np.ndarray:
    """Return columns X,Y,Z,X²,Y²,Z²,XY,YZ,ZX."""
    points = np.asarray(points, dtype=float)
    x, y, z = points.T
    return np.column_stack((x, y, z, x*x, y*y, z*z, x*y, y*z, z*x))


def _checkpoints(n_samples: int, requested: int = 8) -> list[int]:
    if n_samples < 4:
        raise ValueError("At least four post-transient samples are required.")
    powers = [2**k for k in range(2, int(np.log2(n_samples)) + 1)]
    if not powers or powers[-1] != n_samples:
        powers.append(n_samples)
    if len(powers) > requested:
        indices = np.unique(np.linspace(0, len(powers) - 1, requested).round().astype(int))
        powers = [powers[i] for i in indices]
    return powers


def mean_ergodic_ensemble_diagnostic(
    trajectories: Iterable[np.ndarray], *, transient: int = 0, n_checkpoints: int = 8
) -> dict:
    """Compare finite Birkhoff averages across initial conditions with sphere moments.

    This is a selected-observable, finite-ensemble consistency diagnostic.  It
    is not a general test of von Neumann mean ergodicity or a proof of ergodicity.
    """
    ensemble = _validate_ensemble(trajectories)
    if isinstance(transient, bool) or not isinstance(transient, int) or transient < 0:
        raise ValueError("transient must be a non-negative integer.")
    n = min(len(points) for points in ensemble) - transient
    if n < 4:
        raise ValueError("Transient leaves fewer than four common samples.")
    checkpoints = _checkpoints(n, n_checkpoints)
    rows = []
    for count in checkpoints:
        means = np.stack(
            [cartesian_observables(points[transient:transient + count]).mean(axis=0)
             for points in ensemble],
            axis=0,
        )
        ensemble_mean = means.mean(axis=0)
        variance = means.var(axis=0, ddof=1)
        for q, name in enumerate(OBSERVABLE_NAMES):
            rows.append({
                "analyzed_steps": int(count),
                "observable": name,
                "ensemble_mean": float(ensemble_mean[q]),
                "ensemble_variance": float(variance[q]),
                "ensemble_standard_deviation": float(np.sqrt(variance[q])),
                "uniform_sphere_reference_mean": float(AREA_MEANS[q]),
                "squared_bias_from_reference": float((ensemble_mean[q] - AREA_MEANS[q])**2),
            })
    return {
        "n_trajectories": len(ensemble),
        "common_samples_per_trajectory": int(n),
        "transient_discarded": transient,
        "checkpoints": checkpoints,
        "reference_measure": "uniform spherical-area measure",
        "observables": list(OBSERVABLE_NAMES),
        "rows": rows,
        "interpretation": (
            "Decay of across-initial-condition Birkhoff-average variance and "
            "decreasing bias are finite-sample consistency indicators for the "
            "listed observables, not a proof of ergodicity."
        ),
    }


def _pooled_autocorrelation(
    trajectories: list[np.ndarray],
    observable_index: int,
    invariant_mean: float,
    transient: int,
    n_samples: int,
    max_lag: int,
) -> np.ndarray:
    """Estimate invariant-measure correlations by pooling ensemble/time origins.

    The centering constant is the known uniform-sphere mean, not an orbit-wise
    sample mean. This retains correlations caused by invariant components.
    """
    centered = [
        cartesian_observables(points[transient:transient+n_samples])[:, observable_index]
        - float(invariant_mean)
        for points in trajectories
    ]
    variance = float(np.mean(np.concatenate([values * values for values in centered])))
    if variance <= np.finfo(float).eps:
        return np.full(max_lag + 1, np.nan)
    rho = np.empty(max_lag + 1, dtype=float)
    rho[0] = 1.0
    for lag in range(1, max_lag + 1):
        products = np.concatenate([
            values[:-lag] * values[lag:] for values in centered
        ])
        rho[lag] = float(np.mean(products) / variance)
    return rho


def weak_mixing_cesaro_diagnostic(
    trajectories: Iterable[np.ndarray], *, transient: int = 0, max_lag: int = 1000
) -> dict:
    """Compute pooled finite-record Cesàro means of squared correlations.

    For weak mixing, the corresponding Cesàro limit is zero for every
    mean-zero L² observable. Here only nine selected observables and finite
    records are examined. Correlations are estimated by pooling trajectories
    and time origins and are centered on exact uniform-sphere means.
    """
    ensemble = _validate_ensemble(trajectories)
    if isinstance(transient, bool) or not isinstance(transient, int) or transient < 0:
        raise ValueError("transient must be a non-negative integer.")
    n = min(len(points) for points in ensemble) - transient
    if n < 8:
        raise ValueError("Transient leaves fewer than eight common samples.")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 2:
        raise ValueError("max_lag must be an integer >= 2.")
    lag = min(max_lag, (n - 1) // 2)
    by_observable = {}
    for q, name in enumerate(OBSERVABLE_NAMES):
        rho = _pooled_autocorrelation(
            ensemble, q, AREA_MEANS[q], transient, n, lag
        )
        rho2 = rho[1:] ** 2
        cesaro = np.cumsum(rho2) / np.arange(1, lag + 1)
        by_observable[name] = {
            "normalized_autocorrelation": [
                None if not np.isfinite(v) else float(v) for v in rho
            ],
            "cesaro_mean_squared_autocorrelation": [
                None if not np.isfinite(v) else float(v) for v in cesaro
            ],
            "final_cesaro_value": (
                float(cesaro[-1]) if np.isfinite(cesaro[-1]) else None
            ),
            "finite_record_status": (
                "constant_or_unresolved" if not np.all(np.isfinite(rho)) else "computed"
            ),
        }
    return {
        "n_trajectories": len(ensemble),
        "common_samples_per_trajectory": int(n),
        "transient_discarded": transient,
        "max_lag_used": int(lag),
        "correlation_estimator": (
            "pooled ensemble/time-origin covariance centered on exact uniform-sphere "
            "observable means, normalized by pooled lag-zero variance"
        ),
        "observables": by_observable,
        "interpretation": (
            "A decreasing finite-window Cesàro mean is consistent with weak-mixing "
            "behaviour for that tested observable. A nonzero finite-window value "
            "does not by itself disprove weak mixing; finite-time effects, "
            "nonstationarity, and sampling noise remain."
        ),
    }


def _partition_labels(points: np.ndarray, n_z: int, n_phi: int) -> np.ndarray:
    z = np.clip(points[:, 2], -1.0, 1.0)
    phi = np.arctan2(points[:, 1], points[:, 0])
    iz = np.minimum(((z + 1.0) * 0.5 * n_z).astype(int), n_z - 1)
    ip = np.floor((phi + np.pi) * n_phi / (2.0 * np.pi)).astype(int)
    ip = np.clip(ip, 0, n_phi - 1)
    return iz * n_phi + ip


def finite_partition_alpha(delta: np.ndarray) -> float:
    """Exact alpha coefficient for the finite algebra generated by a partition.

    delta[i,j] = P(A_i intersect T^-n A_j) - P(A_i)P(A_j).
    The maximization is over all unions of partition cells in both arguments.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ValueError("delta must be a square matrix.")
    k = delta.shape[0]
    if k > 16:
        raise ValueError("Exact finite-partition alpha calculation is limited to 16 cells.")
    best = 0.0
    for mask in range(1, 1 << k):
        selected = [i for i in range(k) if (mask >> i) & 1]
        vector = delta[selected, :].sum(axis=0)
        candidate = max(float(np.maximum(vector, 0.0).sum()),
                        float(np.maximum(-vector, 0.0).sum()))
        if candidate > best:
            best = candidate
    return float(best)


def strong_mixing_partition_diagnostic(
    trajectories: Iterable[np.ndarray],
    *,
    transient: int = 0,
    n_z: int = 3,
    n_phi: int = 4,
    max_lag: int = 64,
) -> dict:
    """Estimate finite-partition alpha coefficients from an initial-condition ensemble.

    The ensemble is pooled over trajectories and available time origins. This
    requires a defensible approximation to the invariant ensemble; uniform
    initial sampling alone does not guarantee stationarity after transients.
    """
    ensemble = _validate_ensemble(trajectories)
    if isinstance(transient, bool) or not isinstance(transient, int) or transient < 0:
        raise ValueError("transient must be a non-negative integer.")
    for name, value in (("n_z", n_z), ("n_phi", n_phi)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    k = n_z * n_phi
    if k > 16:
        raise ValueError("n_z*n_phi must not exceed 16 for exact finite-partition alpha.")
    n = min(len(points) for points in ensemble) - transient
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer.")
    lag_max = min(max_lag, n - 2)
    if lag_max < 1:
        raise ValueError("Insufficient post-transient samples for lag analysis.")
    labels = [_partition_labels(points[transient:transient+n], n_z, n_phi)
              for points in ensemble]
    lag_values = sorted(set([1, 2, 4, 8, 16, 32, 64, lag_max]))
    lag_values = [lag for lag in lag_values if lag <= lag_max]
    rows = []
    for lag in lag_values:
        origin = np.concatenate([lab[:-lag] for lab in labels])
        future = np.concatenate([lab[lag:] for lab in labels])
        total = len(origin)
        joint = np.bincount(origin * k + future, minlength=k*k).reshape(k, k) / total
        p_origin = np.bincount(origin, minlength=k) / total
        p_future = np.bincount(future, minlength=k) / total
        delta = joint - np.outer(p_origin, p_future)
        rows.append({
            "lag": int(lag),
            "finite_partition_alpha": finite_partition_alpha(delta),
            "max_absolute_cell_pair_deviation": float(np.max(np.abs(delta))),
            "pooled_pairs": int(total),
            "occupied_origin_cells": int(np.count_nonzero(p_origin)),
            "occupied_future_cells": int(np.count_nonzero(p_future)),
        })
    return {
        "n_trajectories": len(ensemble),
        "common_samples_per_trajectory": int(n),
        "transient_discarded": transient,
        "partition": {
            "type": "equal-area spherical cells (uniform z bins and uniform azimuth bins)",
            "n_z": n_z,
            "n_phi": n_phi,
            "n_cells": k,
        },
        "lags": rows,
        "ensemble_assumption": (
            "The pooled trajectory/time-origin sample should approximate an "
            "invariant ensemble. Check stationarity and initial-condition sampling."
        ),
        "interpretation": (
            "These are finite-partition, finite-lag alpha-coefficient estimates. "
            "They do not take the supremum over all measurable sets and cannot "
            "establish strong mixing of the full dynamical system."
        ),
    }


def run_mixing_hierarchy_from_trajectories(
    trajectories: Iterable[np.ndarray],
    *,
    transient: int = 0,
    max_lag: int = 1000,
    n_z: int = 3,
    n_phi: int = 4,
) -> dict:
    """Run the three finite-data diagnostics without returning a hierarchy verdict."""
    ensemble = _validate_ensemble(trajectories)
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 2:
        raise ValueError("max_lag must be an integer >= 2.")
    ergodic = mean_ergodic_ensemble_diagnostic(ensemble, transient=transient)
    weak = weak_mixing_cesaro_diagnostic(
        ensemble, transient=transient, max_lag=max_lag
    )
    strong = strong_mixing_partition_diagnostic(
        ensemble, transient=transient, n_z=n_z, n_phi=n_phi,
        max_lag=min(max_lag, 64),
    )
    return {
        "status": "diagnostic_only",
        "finite_data_diagnostics": {
            "ergodicity_related": ergodic,
            "weak_mixing_related": weak,
            "strong_mixing_related": strong,
        },
        "hierarchy_verdict": None,
        "scientific_limit": (
            "No finite collection of trajectories, observables, lags, or partition "
            "cells establishes ergodicity, weak mixing, or strong mixing as an "
            "asymptotic property. The three diagnostics must be interpreted separately."
        ),
    }
