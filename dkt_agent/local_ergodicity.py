"""Region-conditioned, direction-resolved finite-data ergodicity diagnostics.

A local mean-ergodicity statistic is meaningful only relative to a specified
region carrying an invariant conditional measure. This module therefore
screens forward and backward region leakage before interpreting Birkhoff
averages relative to the region-conditioned area mean.

The screen is statistical, not a proof of invariance. A passing screen licenses
only a finite-data, candidate-level interpretation; it never proves local
ergodicity, pointwise ergodicity, or mixing.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from dkt_agent.mean_ergodicity import DEFAULT_BATTERY, dkt_step, parse_observable

RESULTS_ROOT = Path("results/diagnostics/direction_dependent_local_ergodicity")
REGION_TYPES = ("full_sphere", "spherical_cap", "latitude_band", "longitude_sector")
OBSERVABLES = DEFAULT_BATTERY


def _validate_region(region_type: str, *, center_theta: float, center_phi: float,
                     angular_radius: float, theta_min: float, theta_max: float,
                     phi_min: float, phi_max: float) -> None:
    if region_type not in REGION_TYPES:
        raise ValueError(f"region_type must be one of {REGION_TYPES}")
    values = (center_theta, center_phi, angular_radius, theta_min, theta_max, phi_min, phi_max)
    if not all(np.isfinite(v) for v in values):
        raise ValueError("All region parameters must be finite.")
    if region_type == "spherical_cap":
        if not 0.0 < angular_radius <= math.pi:
            raise ValueError("angular_radius must lie in (0, pi].")
        if not 0.0 <= center_theta <= math.pi:
            raise ValueError("center_theta must lie in [0, pi].")
    elif region_type == "latitude_band":
        if not (0.0 <= theta_min < theta_max <= math.pi):
            raise ValueError("Require 0 <= theta_min < theta_max <= pi.")
    elif region_type == "longitude_sector":
        width = (phi_max - phi_min) % (2.0 * math.pi)
        if width <= 1e-12:
            raise ValueError("phi_min and phi_max must define a nonzero sector.")
        if width >= 2.0 * math.pi - 1e-12:
            raise ValueError("Use full_sphere for a complete longitude range.")


def region_area_fraction(region_type: str, *, center_theta: float = 0.75,
                         center_phi: float = -1.0, angular_radius: float = 0.35,
                         theta_min: float = 0.3, theta_max: float = 1.3,
                         phi_min: float = -1.0, phi_max: float = 1.0) -> float:
    """Exact spherical-area fraction for the supported geometric region types."""
    if region_type == "full_sphere":
        return 1.0
    if region_type == "spherical_cap":
        return (1.0 - math.cos(angular_radius)) / 2.0
    if region_type == "latitude_band":
        return (math.cos(theta_min) - math.cos(theta_max)) / 2.0
    if region_type == "longitude_sector":
        return ((phi_max - phi_min) % (2.0 * math.pi)) / (2.0 * math.pi)
    raise ValueError(f"Unsupported region_type: {region_type}")


def sample_region(region_type: str, n: int, rng: np.random.Generator, *,
                  center_theta: float = 0.75, center_phi: float = -1.0,
                  angular_radius: float = 0.35, theta_min: float = 0.3,
                  theta_max: float = 1.3, phi_min: float = -1.0,
                  phi_max: float = 1.0) -> np.ndarray:
    """Draw points uniformly with respect to spherical area in the selected region."""
    _validate_region(region_type, center_theta=center_theta, center_phi=center_phi,
                     angular_radius=angular_radius, theta_min=theta_min,
                     theta_max=theta_max, phi_min=phi_min, phi_max=phi_max)
    if n < 1:
        raise ValueError("Sample size must be positive.")
    if region_type in {"full_sphere", "longitude_sector"}:
        z = rng.uniform(-1.0, 1.0, n)
    elif region_type == "latitude_band":
        z = rng.uniform(math.cos(theta_max), math.cos(theta_min), n)
    else:
        cos_gamma = rng.uniform(math.cos(angular_radius), 1.0, n)
        gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        psi = rng.uniform(0.0, 2.0 * math.pi, n)
        st, ct = math.sin(center_theta), math.cos(center_theta)
        sp, cp = math.sin(center_phi), math.cos(center_phi)
        axis = np.array([st * cp, st * sp, ct])
        e_theta = np.array([ct * cp, ct * sp, -st])
        e_phi = np.array([-sp, cp, 0.0])
        points = (cos_gamma[:, None] * axis[None, :]
                  + (np.sin(gamma) * np.cos(psi))[:, None] * e_theta[None, :]
                  + (np.sin(gamma) * np.sin(psi))[:, None] * e_phi[None, :])
        return points

    if region_type == "longitude_sector":
        width = (phi_max - phi_min) % (2.0 * math.pi)
        phi = (phi_min % (2.0 * math.pi)) + rng.uniform(0.0, width, n)
    else:
        phi = rng.uniform(-math.pi, math.pi, n)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def region_membership(points: np.ndarray, region_type: str, *,
                      center_theta: float = 0.75, center_phi: float = -1.0,
                      angular_radius: float = 0.35, theta_min: float = 0.3,
                      theta_max: float = 1.3, phi_min: float = -1.0,
                      phi_max: float = 1.0, tolerance: float = 1e-12) -> np.ndarray:
    """Return the geometric membership mask for Cartesian unit-sphere points."""
    points = np.asarray(points, dtype=float)
    if region_type == "full_sphere":
        return np.ones(len(points), dtype=bool)
    if region_type == "spherical_cap":
        st, ct = math.sin(center_theta), math.cos(center_theta)
        axis = np.array([st * math.cos(center_phi), st * math.sin(center_phi), ct])
        return points @ axis >= math.cos(angular_radius) - tolerance
    if region_type == "latitude_band":
        theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
        return (theta >= theta_min - tolerance) & (theta <= theta_max + tolerance)
    if region_type == "longitude_sector":
        phi = np.arctan2(points[:, 1], points[:, 0])
        start = phi_min % (2.0 * math.pi)
        width = (phi_max - phi_min) % (2.0 * math.pi)
        offset = (phi - start) % (2.0 * math.pi)
        return offset <= width + tolerance
    raise ValueError(f"Unsupported region_type: {region_type}")


def dkt_inverse_step(points: np.ndarray, k: float, kp: float, p: float) -> np.ndarray:
    """Exact inverse of dkt_step's rotate-y, twist-z, twist-x composition."""
    x, y, z = np.asarray(points, dtype=float).T
    # Undo the final twist about x; x is unchanged by this twist.
    angle_x = kp * x
    c, s = np.cos(angle_x), np.sin(angle_x)
    y2, z2 = y * c + z * s, -y * s + z * c
    # Undo the preceding twist about z; z is unchanged by this twist.
    angle_z = k * z2
    c, s = np.cos(angle_z), np.sin(angle_z)
    x1, y1 = x * c + y2 * s, -x * s + y2 * c
    # Undo the initial rotation about y.
    cp, sp = math.cos(p), math.sin(p)
    x0, z0 = x1 * cp - z2 * sp, x1 * sp + z2 * cp
    return np.column_stack((x0, y1, z0))


def _step_points(points: np.ndarray, k: float, kp: float, p: float) -> np.ndarray:
    x, y, z = dkt_step(points[:, 0], points[:, 1], points[:, 2], k, kp, p)
    return np.column_stack((x, y, z))


def _wilson_upper(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    """Upper endpoint of a two-sided 95% Wilson interval for a binomial rate."""
    if trials <= 0:
        return 1.0
    phat = successes / trials
    den = 1.0 + z * z / trials
    centre = phat + z * z / (2.0 * trials)
    half = z * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials))
    return min(1.0, (centre + half) / den)


def _observable_values(points: np.ndarray, observables) -> np.ndarray:
    x, y, z = points.T
    return np.column_stack([observable.func(x, y, z) for observable in observables])


def _tail_slope(checkpoints: np.ndarray, values: np.ndarray) -> float | None:
    mask = (checkpoints >= checkpoints[-1] / 10.0) & np.isfinite(values) & (values > 0.0)
    if np.count_nonzero(mask) < 4:
        return None
    slope = np.polyfit(np.log(checkpoints[mask]), np.log(values[mask]), 1)[0]
    return float(-slope)


def run_direction_dependent_local_ergodicity(
    k_r: float,
    k_theta: float = 0.0,
    p: float = math.pi / 2,
    region_type: str = "spherical_cap",
    center_theta: float = 0.75,
    center_phi: float = -1.0,
    angular_radius: float = 0.35,
    theta_min: float = 0.3,
    theta_max: float = 1.3,
    phi_min: float = -1.0,
    phi_max: float = 1.0,
    observables: Sequence[str] = OBSERVABLES,
    n_kicks: int = 4096,
    n_initial_conditions: int = 1200,
    n_reference_samples: int = 12000,
    n_invariance_samples: int = 5000,
    invariance_tolerance: float = 0.005,
    seed: int = 0,
    results_root: str | None = None,
) -> dict:
    """Screen a chosen spherical region and compute direction-resolved averages.

    The region's conditional area measure is used as the reference measure.
    Local mean-ergodicity interpretation is gated by forward/backward leakage
    screens and a trajectory-retention screen. Passing is not proof of
    invariance; failing blocks local-ergodicity interpretation.
    """
    numeric = (k_r, k_theta, p, invariance_tolerance)
    if not all(np.isfinite(v) for v in numeric):
        raise ValueError("Map parameters and invariance_tolerance must be finite.")
    if not 0.0 < invariance_tolerance < 0.25:
        raise ValueError("invariance_tolerance must lie in (0, 0.25).")
    if n_kicks < 16 or n_kicks > 100000:
        raise ValueError("n_kicks must lie between 16 and 100000.")
    if n_initial_conditions < 100 or n_initial_conditions > 10000:
        raise ValueError("n_initial_conditions must lie between 100 and 10000.")
    if n_reference_samples < 1000 or n_reference_samples > 100000:
        raise ValueError("n_reference_samples must lie between 1000 and 100000.")
    if n_invariance_samples < 500 or n_invariance_samples > 100000:
        raise ValueError("n_invariance_samples must lie between 500 and 100000.")
    if n_initial_conditions * n_kicks > 100_000_000:
        raise ValueError("n_initial_conditions × n_kicks is too large for one run.")
    names = list(dict.fromkeys(str(v).lower().replace(" ", "") for v in observables))
    if not names:
        raise ValueError("Select at least one observable.")
    if len(names) > 9 or any(name not in OBSERVABLES for name in names):
        raise ValueError(f"Supported observables are {OBSERVABLES}.")
    _validate_region(region_type, center_theta=center_theta, center_phi=center_phi,
                     angular_radius=angular_radius, theta_min=theta_min,
                     theta_max=theta_max, phi_min=phi_min, phi_max=phi_max)

    k, kp = k_r + k_theta, k_r - k_theta
    rng = np.random.default_rng(seed)
    region_args = dict(center_theta=center_theta, center_phi=center_phi,
                       angular_radius=angular_radius, theta_min=theta_min,
                       theta_max=theta_max, phi_min=phi_min, phi_max=phi_max)

    # Independent samples: one for the conditional reference moments and one
    # for the forward/backward invariance screen.
    reference = sample_region(region_type, n_reference_samples, rng, **region_args)
    parsed_observables = [parse_observable(name) for name in names]
    reference_values = _observable_values(reference, parsed_observables)
    reference_means = reference_values.mean(axis=0)
    reference_variances = reference_values.var(axis=0, ddof=1)
    reference_mean_se = reference_values.std(axis=0, ddof=1) / math.sqrt(n_reference_samples)

    screen_points = sample_region(region_type, n_invariance_samples, rng, **region_args)
    forward = _step_points(screen_points, k, kp, p)
    backward = dkt_inverse_step(screen_points, k, kp, p)
    forward_out = int((~region_membership(forward, region_type, **region_args)).sum())
    backward_out = int((~region_membership(backward, region_type, **region_args)).sum())

    # A separate ensemble supplies conditional Birkhoff averages. We track
    # whether any sampled orbit leaves the proposed region at any kick.
    initial = sample_region(region_type, n_initial_conditions, rng, **region_args)
    points = initial.copy()
    ever_out = np.zeros(n_initial_conditions, dtype=bool)
    max_norm_error = float(np.max(np.abs(np.sum(points * points, axis=1) - 1.0)))
    sums = np.zeros((n_initial_conditions, len(names)), dtype=float)
    checkpoints = np.unique(np.round(np.geomspace(1, n_kicks, 32)).astype(int))
    d_values = np.empty((len(checkpoints), len(names)), dtype=float)
    d_errors = np.empty_like(d_values)
    normalized = np.empty_like(d_values)
    checkpoint_index = 0
    for kick in range(1, n_kicks + 1):
        sums += _observable_values(points, parsed_observables)
        points = _step_points(points, k, kp, p)
        max_norm_error = max(
            max_norm_error,
            float(np.max(np.abs(np.sum(points * points, axis=1) - 1.0))),
        )
        ever_out |= ~region_membership(points, region_type, **region_args)
        if kick == checkpoints[checkpoint_index]:
            averages = sums / kick
            delta2 = (averages - reference_means[None, :]) ** 2
            # Remove the leading variance contribution from the independent
            # Monte-Carlo estimate of the regional mean (delta method).
            raw = delta2.mean(axis=0)
            corrected = np.maximum(0.0, raw - reference_variances / n_reference_samples)
            d_values[checkpoint_index] = corrected
            d_errors[checkpoint_index] = delta2.std(axis=0, ddof=1) / math.sqrt(n_initial_conditions)
            normalized[checkpoint_index] = np.divide(
                corrected, reference_variances,
                out=np.full_like(corrected, np.nan), where=reference_variances > 1e-14
            )
            checkpoint_index += 1
            if checkpoint_index == len(checkpoints):
                break

    orbit_out = int(ever_out.sum())
    upper_forward = _wilson_upper(forward_out, n_invariance_samples)
    upper_backward = _wilson_upper(backward_out, n_invariance_samples)
    upper_orbit = _wilson_upper(orbit_out, n_initial_conditions)
    screen_pass = max(upper_forward, upper_backward, upper_orbit) <= invariance_tolerance
    invariance = {
        "status": "finite_sample_invariance_screen_passed" if screen_pass else "invariance_not_supported_at_selected_tolerance",
        "tolerance": invariance_tolerance,
        "confidence_level": 0.95,
        "forward_one_step": {"escapes": forward_out, "samples": n_invariance_samples,
                             "escape_fraction": forward_out / n_invariance_samples,
                             "wilson_95_upper_bound": upper_forward},
        "backward_one_step": {"escapes": backward_out, "samples": n_invariance_samples,
                              "escape_fraction": backward_out / n_invariance_samples,
                              "wilson_95_upper_bound": upper_backward},
        "trajectory_retention": {"orbits_with_any_escape": orbit_out, "orbits": n_initial_conditions,
                                 "fraction": orbit_out / n_initial_conditions,
                                 "wilson_95_upper_bound": upper_orbit},
        "interpretation": (
            "The proposed region passed finite-sample forward, backward and multi-kick leakage screens. "
            "This is a candidate-region screen, not a proof that the region is invariant."
            if screen_pass else
            "The proposed region did not pass the selected finite-sample leakage screen. "
            "The conditional Birkhoff statistics below are descriptive for initial points sampled in this geometric region; "
            "they must not be interpreted as local mean-ergodicity statistics for an invariant subsystem."
        ),
    }

    per_observable = {}
    for j, name in enumerate(names):
        per_observable[name] = {
            "regional_mean_estimate": float(reference_means[j]),
            "regional_mean_standard_error": float(reference_mean_se[j]),
            "regional_variance_estimate": float(reference_variances[j]),
            "D_N_over_regional_variance": [
                float(v) if np.isfinite(v) else None for v in normalized[:, j]
            ],
            "D_N_estimate": [float(v) for v in d_values[:, j]],
            "D_N_standard_error_across_initial_conditions": [float(v) for v in d_errors[:, j]],
            "descriptive_tail_decay_exponent": _tail_slope(checkpoints, normalized[:, j]),
            "finite_data_note": (
                "Conditional regional L2 indicator candidate: interpret only together with the passed finite-sample invariance screen."
                if screen_pass else
                "Descriptive Birkhoff-average spread only; no local invariant-measure interpretation is supported by the leakage screen."
            ),
        }

    final_scores = {name: (values["D_N_over_regional_variance"][-1]
                           if values["D_N_over_regional_variance"] else None)
                    for name, values in per_observable.items()}
    ordered = sorted(
        ((name, value) for name, value in final_scores.items() if value is not None),
        key=lambda item: item[1], reverse=True
    )
    directional_summary = {
        "largest_finite_time_normalized_spread": ordered[:3],
        "smallest_finite_time_normalized_spread": ordered[-3:],
        "note": "This ranks finite-time observable-wise spreads only. It does not classify an observable as ergodic or non-ergodic."
    }

    parameters = {
        "k_r": float(k_r), "k_theta": float(k_theta), "k": float(k), "k_prime": float(kp),
        "p": float(p), "region_type": region_type, **{key: float(value) for key, value in region_args.items()},
        "region_area_fraction_of_sphere": region_area_fraction(region_type, **region_args),
        "coordinate_convention": "theta from +Z; phi=atan2(Y,X); radians",
        "observables": names, "n_kicks": int(n_kicks),
        "n_initial_conditions": int(n_initial_conditions),
        "n_reference_samples": int(n_reference_samples),
        "n_invariance_samples": int(n_invariance_samples),
        "invariance_tolerance": float(invariance_tolerance), "seed": int(seed),
    }
    out_dir = Path(results_root or RESULTS_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = out_dir / f"local_ergodicity_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    csv_path = run_dir / "directional_local_birkhoff_spread.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n_kicks", "observable", "D_N", "D_N_standard_error", "D_N_over_regional_variance"])
        for i, n in enumerate(checkpoints):
            for j, name in enumerate(names):
                writer.writerow([int(n), name, f"{d_values[i,j]:.12e}",
                                 f"{d_errors[i,j]:.12e}", f"{normalized[i,j]:.12e}"])

    invariance_path = run_dir / "region_invariance_screen.json"
    invariance_path.write_text(json.dumps(invariance, indent=2), encoding="utf-8")
    summary = {
        "status": "success",
        "diagnostic": "Direction-dependent, region-conditioned finite-data ergodicity investigation",
        "parameters": parameters,
        "region_invariance_screen": invariance,
        "local_measure_interpretation": (
            "candidate_region_only; conditional area measure is a working hypothesis pending stronger invariance validation"
            if screen_pass else
            "blocked_by_region_leakage; only descriptive conditional-start Birkhoff statistics are reported"
        ),
        "observables": per_observable,
        "directional_summary": directional_summary,
        "numerical_checks": {"maximum_unit_sphere_norm_error": max_norm_error},
        "scientific_interpretation": [
            "The region-conditioned area measure is appropriate as an invariant measure only if the selected region is invariant under the map; finite-sample leakage screens cannot prove this.",
            "The observable-wise D_N values are finite-data L2 indicators relative to the estimated conditional area means, not proofs of mean or pointwise ergodicity.",
            "Different finite-time spreads across X, Y, Z or quadratic observables are direction-/observable-dependent statistics, not categorical ergodicity verdicts.",
            "Mean ergodicity, Birkhoff/pointwise ergodicity, weak mixing and strong mixing remain distinct properties.",
            "At fixed k_r, changing k_theta must be interpreted as changing phase-space geometry, not the degree of classical chaos, Lyapunov behaviour or KS entropy."
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "directory": str(run_dir),
            "summary_json": str(run_dir / "local_ergodicity_summary.json"),
            "directional_birkhoff_csv": str(csv_path),
            "region_invariance_json": str(invariance_path),
        },
    }
    summary_path = run_dir / "local_ergodicity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for j, name in enumerate(names):
            ax.loglog(checkpoints, np.clip(normalized[:, j], 1e-16, None), marker="o", ms=3, label=name)
        ax.set_xlabel("Number of kicks")
        ax.set_ylabel(r"$D_N(f)/\mathrm{Var}_{R}(f)$")
        ax.set_title("Region-conditioned Birkhoff-average spread by observable")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        plot_path = run_dir / "directional_local_birkhoff_spread.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        summary["artifacts"]["directional_birkhoff_plot"] = str(plot_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except ImportError:
        pass
    return summary


def format_for_agent(result: dict) -> str:
    """Compact, scientifically qualified summary for the research assistant."""
    params = result["parameters"]
    screen = result["region_invariance_screen"]
    lines = [
        "Direction-dependent local ergodicity diagnostic (finite data)",
        f"Map: k_r={params['k_r']:g}, k_theta={params['k_theta']:g}, p={params['p']:g}",
        f"Region: {params['region_type']} · {screen['status']}",
        f"Forward escape fraction={screen['forward_one_step']['escape_fraction']:.4g} "
        f"(95% Wilson upper bound {screen['forward_one_step']['wilson_95_upper_bound']:.4g})",
        f"Backward escape fraction={screen['backward_one_step']['escape_fraction']:.4g} "
        f"(95% Wilson upper bound {screen['backward_one_step']['wilson_95_upper_bound']:.4g})",
        f"Multi-kick orbit escape fraction={screen['trajectory_retention']['fraction']:.4g} "
        f"(95% Wilson upper bound {screen['trajectory_retention']['wilson_95_upper_bound']:.4g})",
    ]
    for name, values in result["observables"].items():
        final = values["D_N_over_regional_variance"][-1]
        slope = values["descriptive_tail_decay_exponent"]
        lines.append(f"{name}: final D_N/Var_R={final:.4g}; descriptive tail exponent={slope}")
    lines.append("Interpretation: " + screen["interpretation"])
    lines.extend(result["scientific_interpretation"])
    lines.append("Artifacts: " + json.dumps(result["artifacts"], ensure_ascii=False))
    return "\n".join(lines)


try:
    from langchain_core.tools import tool

    @tool
    def diagnose_direction_dependent_local_ergodicity(
        k_r: float,
        k_theta: float,
        region_type: str,
        p: float = math.pi / 2,
        center_theta: float = 0.75,
        center_phi: float = -1.0,
        angular_radius: float = 0.35,
        theta_min: float = 0.3,
        theta_max: float = 1.3,
        phi_min: float = -1.0,
        phi_max: float = 1.0,
        observables: list[str] | None = None,
        n_kicks: int = 4096,
        n_initial_conditions: int = 1200,
        n_reference_samples: int = 12000,
        n_invariance_samples: int = 5000,
        invariance_tolerance: float = 0.005,
        seed: int = 0,
    ) -> str:
        """Test direction-dependent Birkhoff-average statistics in an explicitly defined spherical region.

        Supported regions: full_sphere, spherical_cap, latitude_band, longitude_sector.
        Angles are radians; theta is measured from +Z; phi=atan2(Y,X).
        A local-measure interpretation is gated by forward/backward one-step
        leakage and multi-kick retention screens. A passing finite-sample screen
        is not proof of invariance or ergodicity. Use multiple observables
        (x,y,z,x2,y2,z2,xy,yz,xz)."""
        result = run_direction_dependent_local_ergodicity(
            k_r=k_r, k_theta=k_theta, p=p, region_type=region_type,
            center_theta=center_theta, center_phi=center_phi,
            angular_radius=angular_radius, theta_min=theta_min, theta_max=theta_max,
            phi_min=phi_min, phi_max=phi_max,
            observables=observables or DEFAULT_BATTERY, n_kicks=n_kicks,
            n_initial_conditions=n_initial_conditions,
            n_reference_samples=n_reference_samples,
            n_invariance_samples=n_invariance_samples,
            invariance_tolerance=invariance_tolerance, seed=seed,
        )
        return json.dumps(result, indent=2, ensure_ascii=False)

except ImportError:  # pragma: no cover
    diagnose_direction_dependent_local_ergodicity = None
