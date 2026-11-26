"""Safe, agent-facing classical research diagnostics."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from dkt_agent.tools import simulate_dkt as _simulate_dkt

from dkt_agent.classical_analysis import (
    empirical_measure_comparison,
    load_trajectory,
    run_ergodicity_diagnostics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = (PROJECT_ROOT / "results" / "trajectories").resolve()
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
DIAGNOSTICS_ROOT = PROJECT_ROOT / "results" / "diagnostics"


def _resolve_trajectory_csv(value: str) -> Path:
    """Resolve an existing trajectory CSV without allowing arbitrary reads."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A trajectory CSV path is required.")

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Trajectory CSV not found: {value}") from exc

    try:
        resolved.relative_to(TRAJECTORY_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Only CSV files inside results/trajectories/ can be analysed. "
            "Use the exact path returned by a DKT simulation."
        ) from exc

    if not resolved.is_file() or resolved.suffix.lower() != ".csv":
        raise ValueError("The selected trajectory must be an existing CSV file.")

    return resolved


def _new_output_dir(prefix: str) -> Path:
    DIAGNOSTICS_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=DIAGNOSTICS_ROOT))


@tool
def diagnose_classical_trajectory(
    trajectory_csv: str,
    theiler: int = 20,
    n_pairs: int = 200000,
    seed: int = 1234,
) -> str:
    """Run finite-data geometric diagnostics on an existing DKT trajectory.

    Estimates correlation dimension and shifted-grid box-counting dimension,
    and saves the correlation curve, local slopes, occupied-cell curve, plots,
    and a JSON summary. This does not prove ergodicity or mixing.
    """
    path = _resolve_trajectory_csv(trajectory_csv)

    for name, value, minimum in (
        ("theiler", theiler, 0),
        ("n_pairs", n_pairs, 1000),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if n_pairs > 2_000_000:
        raise ValueError("n_pairs must not exceed 2,000,000.")

    output_dir = _new_output_dir("trajectory_diagnostics_")
    result = run_ergodicity_diagnostics(
        {"trajectory": str(path)},
        output_dir,
        theiler=theiler,
        n_pairs=n_pairs,
        seed=seed,
    )
    summary = result["trajectories"]["trajectory"]

    return json.dumps(
        {
            "status": "success",
            "trajectory_csv": str(path),
            "n_points": summary["n_points"],
            "correlation_dimension": summary["correlation_dimension"],
            "box_dimension": summary["box_dimension"],
            "parameters": result["parameters"],
            "artifacts": {
                "output_directory": str(output_dir),
                "summary_json": str(output_dir / "diagnostics_summary.json"),
                "correlation_curve_csv": str(output_dir / "trajectory_correlation_curve.csv"),
                "box_curve_csv": str(output_dir / "trajectory_box_curve.csv"),
                "correlation_plot": str(output_dir / "trajectory_correlation_dimension.png"),
                "box_plot": str(output_dir / "trajectory_box_scaling.png"),
            },
            "interpretation_limits": [
                "These are finite-trajectory geometric scaling diagnostics.",
                "A fitted slope depends on the selected scaling range and must be checked against the full curve and local slopes.",
                "Neither estimate proves ergodicity, mixing, or an invariant-measure statement.",
            ],
        },
        indent=2,
    )


@tool
def compare_trajectory_measures(
    trajectory_csvs: list[str],
    n_z: int = 40,
    n_phi: int = 80,
) -> str:
    """Compare empirical measures of 2–10 DKT trajectories.

    Uses a common equal-area spherical partition and pairwise base-2
    Jensen–Shannon divergences. This compares finite sampled distributions;
    it does not establish ergodicity, mixing, or chaos.
    """
    if not isinstance(trajectory_csvs, list) or not 2 <= len(trajectory_csvs) <= 10:
        raise ValueError("trajectory_csvs must contain between 2 and 10 paths.")
    if isinstance(n_z, bool) or not isinstance(n_z, int) or n_z < 2:
        raise ValueError("n_z must be an integer >= 2.")
    if isinstance(n_phi, bool) or not isinstance(n_phi, int) or n_phi < 2:
        raise ValueError("n_phi must be an integer >= 2.")

    paths = [_resolve_trajectory_csv(value) for value in trajectory_csvs]
    if len(set(paths)) != len(paths):
        raise ValueError("The trajectory CSV paths must be distinct.")

    labels = [f"trajectory_{index + 1}" for index in range(len(paths))]
    trajectories = {
        label: load_trajectory(path)
        for label, path in zip(labels, paths)
    }
    comparison = empirical_measure_comparison(
        trajectories, n_z=n_z, n_phi=n_phi
    )

    output_dir = _new_output_dir("measure_comparison_")
    matrix_path = output_dir / "jensen_shannon_divergence.csv"
    matrix = comparison["js_divergence"].tolist()
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])

    return json.dumps(
        {
            "status": "success",
            "trajectory_files": [str(path) for path in paths],
            "labels": labels,
            "jensen_shannon_divergence_base2": matrix,
            "partition": {"n_z": n_z, "n_phi": n_phi},
            "csv_path": str(matrix_path),
            "interpretation_limits": [
                "The matrix compares empirical finite-sample distributions on the selected common partition.",
                "Comparisons are most interpretable when trajectories use compatible observation windows and sampling conventions.",
                "A difference in empirical distributions is not by itself evidence of a change in chaoticity, ergodicity, or mixing.",
            ],
        },
        indent=2,
    )


@tool
def diagnose_observable_timeseries(
    trajectory_csv: str,
    transient: int = 0,
    max_lag: int = 200,
    block_sizes: list[int] | None = None,
) -> str:
    """Compute finite-record Birkhoff averages, block-mean fluctuations and autocorrelations.

    Observables: X,Y,Z,X²,Y²,Z²,XY,YZ,ZX. The CSV includes kick zero;
    transient discards rows 0 through transient inclusive. These finite-data
    diagnostics do not prove ergodicity or mixing.
    """
    import numpy as np
    path = _resolve_trajectory_csv(trajectory_csv)
    points = load_trajectory(path)
    x, y, z = points.T
    series = {
        'X': x, 'Y': y, 'Z': z, 'X2': x*x, 'Y2': y*y, 'Z2': z*z,
        'XY': x*y, 'YZ': y*z, 'ZX': z*x,
    }
    n_total = len(points)
    if isinstance(transient, bool) or not isinstance(transient, int) or not 0 <= transient < n_total - 1:
        raise ValueError(f'transient must be in [0, {n_total - 2}] so at least one post-transient sample remains.')
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or not 1 <= max_lag <= 5000:
        raise ValueError('max_lag must be between 1 and 5000.')
    series = {key: value[transient + 1:] for key, value in series.items()}
    n = n_total - transient - 1
    max_lag = min(max_lag, n - 1)
    if block_sizes is None:
        block_sizes = [v for v in (10, 20, 50, 100, 200, 500, 1000) if v <= n]
        if not block_sizes and n >= 2:
            block_sizes = [2]
    if not isinstance(block_sizes, list) or not block_sizes:
        raise ValueError('block_sizes must be a non-empty list.')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 or v > n for v in block_sizes) or len(set(block_sizes)) != len(block_sizes):
        raise ValueError('block_sizes must be distinct integers between 2 and the analyzed sample count.')

    averages, fluctuations, correlations = {}, {}, {}
    for name, values in series.items():
        averages[name] = {
            'mean': float(np.mean(values)),
            'standard_deviation': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'samples': int(len(values)),
        }
        block_rows = []
        for size in sorted(block_sizes):
            n_blocks = len(values) // size
            if n_blocks < 2:
                continue
            means = values[:n_blocks * size].reshape(n_blocks, size).mean(axis=1)
            block_rows.append({
                'block_size': size, 'blocks': int(n_blocks),
                'mean_of_block_means': float(np.mean(means)),
                'variance_of_block_means': float(np.var(means, ddof=1)),
                'standard_error': float(np.std(means, ddof=1) / np.sqrt(n_blocks)),
            })
        fluctuations[name] = block_rows
        if name in {'X', 'Y', 'Z'}:
            centered = values - np.mean(values)
            var = float(np.dot(centered, centered) / len(centered))
            if var <= np.finfo(float).eps:
                correlations[name] = [1.0] + [None] * max_lag
            else:
                correlations[name] = [
                    float(np.dot(centered[:len(centered)-lag or None], centered[lag:]) /
                          (len(centered)-lag) / var)
                    for lag in range(max_lag + 1)
                ]

    out = _new_output_dir('observable_diagnostics_')
    result = {
        'status': 'success', 'trajectory_csv': str(path),
        'transient_discarded': transient, 'samples_analyzed': n,
        'kick_convention': 'Discard rows 0 through transient inclusive; average rows transient+1 onward.',
        'birkhoff_time_averages': averages,
        'block_fluctuations': fluctuations,
        'autocorrelation': {'max_lag': max_lag, 'normalization': 'C(0)=1', 'observables': correlations},
        'artifact_directory': str(out),
        'interpretation_limits': [
            'These are finite-record time averages, not infinite-time Birkhoff limits.',
            'Block variance depends on available record length and selected block sizes.',
            'Finite-lag autocorrelation decay alone does not prove mixing.',
            'No global ergodicity or invariant-measure claim is inferred.',
        ],
    }
    (out / 'observable_diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return json.dumps(result, indent=2)


@tool
def summarize_research_csv(relative_path: str, max_rows: int = 200000) -> str:
    """Read-only descriptive summary of a CSV inside results/: schema, missingness and numeric statistics."""
    import csv, math
    import numpy as np
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError('relative_path is required.')
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 200000:
        raise ValueError('max_rows must be between 1 and 200000.')
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError('Use a path relative to results/.')
    if requested.parts and requested.parts[0] == RESULTS_ROOT.name:
        requested = Path(*requested.parts[1:])
    path = (RESULTS_ROOT / requested).resolve()
    try:
        path.relative_to(RESULTS_ROOT)
    except ValueError as exc:
        raise ValueError('CSV must remain inside results/.') from exc
    if path.suffix.lower() != '.csv' or not path.is_file():
        raise ValueError('An existing CSV inside results/ is required.')
    if path.stat().st_size > 50_000_000:
        raise ValueError('CSV exceeds 50 MB.')
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError('CSV has no header.')
        columns = list(reader.fieldnames)
        numeric = {c: [] for c in columns}
        missing = {c: 0 for c in columns}
        non_numeric = {c: 0 for c in columns}
        rows = 0
        for row in reader:
            if rows >= max_rows:
                break
            rows += 1
            for col in columns:
                raw = row.get(col)
                if raw is None or not str(raw).strip():
                    missing[col] += 1
                    continue
                try:
                    value = float(raw)
                    if math.isfinite(value): numeric[col].append(value)
                    else: missing[col] += 1
                except (TypeError, ValueError):
                    non_numeric[col] += 1
    stats = {}
    for col, vals in numeric.items():
        arr = np.asarray(vals, dtype=float)
        stats[col] = ({'numeric_values': int(len(arr)), 'mean': float(arr.mean()),
                       'standard_deviation': float(arr.std(ddof=1)) if len(arr) > 1 else None,
                       'minimum': float(arr.min()), 'median': float(np.median(arr)),
                       'maximum': float(arr.max())} if len(arr) else {'numeric_values': 0})
    return json.dumps({
        'status': 'success', 'path': path.relative_to(PROJECT_ROOT).as_posix(),
        'rows_analyzed': rows, 'possibly_truncated': rows == max_rows,
        'columns': columns, 'missing_or_nonfinite_values': missing,
        'non_numeric_values': non_numeric, 'numeric_statistics': stats,
        'limitations': ['Descriptive statistics do not validate data-generation methods.',
                        'Column meanings are not inferred.',
                        'If truncated, statistics cover only the first max_rows records.'],
    }, indent=2)


@tool
def run_dkt_parameter_sweep(
    parameter: str, values: list[float], kr: float = 1.0, ktheta: float = 0.0,
    theta0: float = 0.75, phi0: float = -1.0, nsteps: int = 2000,
    p: float = 1.5707963267948966, n_z: int = 30, n_phi: int = 60,
) -> str:
    """Run a controlled one-parameter sweep (2–8 values) and compare empirical phase-space measures.

    Holds the initial condition and all unselected parameters fixed. Reports
    trajectories, norm errors and pairwise base-2 Jensen-Shannon divergences.
    Does not calculate Lyapunov exponents or prove chaos, ergodicity or mixing.
    """
    import math
    if parameter not in {'kr', 'ktheta'}:
        raise ValueError("parameter must be 'kr' or 'ktheta'.")
    if not isinstance(values, list) or not 2 <= len(values) <= 8:
        raise ValueError('values must contain 2–8 distinct finite numbers.')
    clean = [float(v) for v in values]
    if not all(math.isfinite(v) for v in clean) or len(set(clean)) != len(clean):
        raise ValueError('Sweep values must be finite and distinct.')
    for name, value in [('kr', kr), ('ktheta', ktheta), ('theta0', theta0), ('phi0', phi0), ('p', p)]:
        if not math.isfinite(float(value)): raise ValueError(f'{name} must be finite.')
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or not 1 <= nsteps <= 200000:
        raise ValueError('nsteps must be between 1 and 200000.')
    if isinstance(n_z, bool) or not isinstance(n_z, int) or not 2 <= n_z <= 100:
        raise ValueError('n_z must be between 2 and 100.')
    if isinstance(n_phi, bool) or not isinstance(n_phi, int) or not 2 <= n_phi <= 200:
        raise ValueError('n_phi must be between 2 and 200.')
    runs, trajectories = [], {}
    for i, value in enumerate(clean):
        run_kr = value if parameter == 'kr' else float(kr)
        run_ktheta = value if parameter == 'ktheta' else float(ktheta)
        sim = _simulate_dkt(kr=run_kr, ktheta=run_ktheta, theta0=float(theta0),
                            phi0=float(phi0), nsteps=nsteps, p=float(p), include_trajectory=False)
        label = f'{parameter}_{i+1}'
        trajectories[label] = load_trajectory(sim['trajectory_csv'])
        runs.append({
            'label': label, 'varied_value': value, 'parameters': sim['parameters'],
            'trajectory_rows': sim['trajectory_rows'], 'initial_state': sim['initial_state'],
            'final_state': sim['final_state'], 'max_norm_error': sim['max_norm_error'],
            'trajectory_csv': sim['trajectory_csv'],
        })
    comparison = empirical_measure_comparison(trajectories, n_z=n_z, n_phi=n_phi)
    out = _new_output_dir('parameter_sweep_')
    matrix_path = out / 'empirical_measure_js.csv'
    labels, matrix = comparison['labels'], comparison['js_divergence'].tolist()
    with matrix_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(['trajectory', *labels])
        for label, row in zip(labels, matrix): writer.writerow([label, *row])
    return json.dumps({
        'status': 'success', 'sweep': {'parameter': parameter, 'values': clean,
        'fixed_kr': float(kr), 'fixed_ktheta': float(ktheta), 'theta0': float(theta0),
        'phi0': float(phi0), 'nsteps': nsteps, 'p': float(p)},
        'runs': runs, 'empirical_measure_comparison': {
            'labels': labels, 'jensen_shannon_divergence_base2': matrix,
            'partition': {'n_z': n_z, 'n_phi': n_phi}, 'csv_path': str(matrix_path)},
        'interpretation_limits': [
            'Only one parameter is varied; other settings and initial conditions are fixed.',
            'Jensen-Shannon divergence compares finite empirical distributions on the selected partition.',
            'A ktheta sweep at fixed kr probes geometry/distributional differences, not degree of chaos.',
            'A kr sweep does not by itself establish a chaos transition; no Lyapunov exponent is calculated.',
        ],
    }, indent=2)



def _finite_float(name: str, value: float) -> float:
    import math
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _run_julia_lyapunov(
    *, kr: float, ktheta: float, theta0: float, phi0: float, p: float,
    nsteps: int, transient: int, output_csv: Path, timeout: int = 900,
) -> list[dict]:
    """Call QuantumKickedTop's existing tangent-plane QR Lyapunov routine."""
    import csv
    import os
    import shutil
    import subprocess

    julia = os.environ.get("JULIA") or shutil.which("julia")
    if not julia:
        raise FileNotFoundError("Julia executable not found; install Julia or set JULIA.")
    project = PROJECT_ROOT / "QuantumKickedTop.jl"
    script = PROJECT_ROOT / "scripts" / "julia" / "dkt_lyapunov_ks.jl"
    if not project.is_dir() or not script.is_file():
        raise FileNotFoundError("Julia project or dkt_lyapunov_ks.jl script is missing.")
    command = [
        julia, f"--project={project}", str(script), str(kr), str(ktheta),
        str(theta0), str(phi0), str(p), str(nsteps), str(transient), str(output_csv),
    ]
    try:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"QR Lyapunov calculation exceeded {timeout} seconds.") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "Julia QR Lyapunov calculation failed.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if not output_csv.is_file():
        raise RuntimeError("Julia completed without producing the Lyapunov CSV.")
    rows = []
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "analyzed_steps", "lambda1", "lambda2", "sum_exponents",
            "ks_pesin_estimate",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Unexpected Lyapunov CSV header: {reader.fieldnames}")
        for row in reader:
            parsed = {
                "analyzed_steps": int(row["analyzed_steps"]),
                "lambda1": float(row["lambda1"]),
                "lambda2": float(row["lambda2"]),
                "sum_exponents": float(row["sum_exponents"]),
                "ks_pesin_estimate": float(row["ks_pesin_estimate"]),
            }
            import math
            if not all(math.isfinite(v) for k, v in parsed.items() if k != "analyzed_steps"):
                raise RuntimeError("Lyapunov CSV contains non-finite values.")
            rows.append(parsed)
    if not rows:
        raise RuntimeError("Lyapunov CSV contains no estimates.")
    return rows


@tool
def diagnose_lyapunov_ks(
    kr: float, ktheta: float = 0.0, theta0: float = 0.75,
    phi0: float = -1.0, p: float = 1.5707963267948966,
    nsteps: int = 20000, transient: int = 1000, timeout: int = 900,
) -> str:
    """Estimate the two tangent-plane Lyapunov exponents and finite-time Pesin KS entropy.

    Uses QuantumKickedTop.jl's analytic tangent map and periodic QR
    orthonormalization. Reports multiple finite-time windows. The KS quantity
    is a finite-time sum of positive exponents (Pesin estimate), not a proof
    of asymptotic KS entropy or global chaos.
    """
    import json
    for name, value in (("kr", kr), ("ktheta", ktheta), ("theta0", theta0), ("phi0", phi0), ("p", p)):
        _finite_float(name, value)
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or not 100 <= nsteps <= 500000:
        raise ValueError("nsteps must be an integer between 100 and 500000.")
    if isinstance(transient, bool) or not isinstance(transient, int) or not 0 <= transient <= nsteps - 100:
        raise ValueError("transient must leave at least 100 analyzed kicks.")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 3600:
        raise ValueError("timeout must be an integer between 1 and 3600 seconds.")
    out = _new_output_dir("lyapunov_ks_")
    csv_path = out / "lyapunov_ks_checkpoints.csv"
    checkpoints = _run_julia_lyapunov(
        kr=float(kr), ktheta=float(ktheta), theta0=float(theta0), phi0=float(phi0),
        p=float(p), nsteps=nsteps, transient=transient,
        output_csv=csv_path, timeout=timeout,
    )
    final = checkpoints[-1]
    result = {
        "status": "success",
        "parameters": {
            "kr": float(kr), "ktheta": float(ktheta), "k": float(kr) + float(ktheta),
            "kprime": float(kr) - float(ktheta), "theta0": float(theta0),
            "phi0": float(phi0), "p": float(p), "nsteps": nsteps,
            "transient_discarded": transient,
            "analyzed_steps": nsteps - transient,
        },
        "method": "analytic tangent Jacobian on the sphere with periodic QR orthonormalization",
        "lyapunov_exponents_nats_per_kick": [final["lambda1"], final["lambda2"]],
        "sum_of_exponents_nats_per_kick": final["sum_exponents"],
        "area_preservation_consistency_check": {
            "expected_sum": 0.0,
            "measured_sum": final["sum_exponents"],
            "absolute_residual": abs(final["sum_exponents"]),
            "interpretation": "For this area-preserving spherical map, the two asymptotic exponents should sum to zero; the finite-time residual is a numerical diagnostic, not a correction."
        },
        "finite_time_pesin_ks_estimate_nats_per_kick": final["ks_pesin_estimate"],
        "finite_time_checkpoints": checkpoints,
        "artifacts": {"directory": str(out), "csv": str(csv_path)},
        "interpretation_limits": [
            "These are finite-time, initial-condition-specific Lyapunov estimates.",
            "Checkpoint drift indicates that the estimate may not yet be stable with observation time.",
            "The KS value is the finite-time sum of positive exponents (Pesin estimate); asymptotic equality requires the relevant dynamical and measure-theoretic assumptions.",
            "A positive finite-time exponent does not establish global chaos or a global regular-to-chaotic boundary.",
            "At fixed kr, changing ktheta must not be interpreted as changing the degree of classical chaos; finite-time variations can reflect orbit and sampling effects."
        ],
    }
    (out / "lyapunov_ks_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json.dumps(result, indent=2)


def _autocorrelation_fft(values, max_lag: int):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional series with at least two samples.")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or not 0 <= max_lag < len(values):
        raise ValueError("max_lag must be an integer in [0, n-1].")
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered) / len(centered))
    if variance <= np.finfo(float).eps:
        return np.concatenate(([1.0], np.full(max_lag, np.nan))), "constant_observable"
    nfft = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=nfft)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:max_lag + 1]
    raw /= np.arange(len(centered), len(centered) - max_lag - 1, -1)
    return raw / raw[0], "ok"


@tool
def diagnose_mixing_correlations(
    trajectory_csv: str, transient: int = 0, max_lag: int = 2000,
    threshold: float = 0.05,
) -> str:
    """Compute finite-record normalized autocorrelation functions for nine Cartesian observables.

    Reports first zero crossings and sustained threshold crossings as
    descriptive correlation-decay diagnostics. No mixing classification is
    inferred from finite data.
    """
    import csv
    import json
    import numpy as np
    path = _resolve_trajectory_csv(trajectory_csv)
    points = load_trajectory(path)
    if isinstance(transient, bool) or not isinstance(transient, int) or not 0 <= transient < len(points) - 2:
        raise ValueError("transient must leave at least two post-transient samples.")
    values = points[transient + 1:]
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer.")
    max_lag = min(max_lag, (len(values) - 1) // 2, 5000)
    threshold = _finite_float("threshold", threshold)
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1.")
    x, y, z = values.T
    series = {"X": x, "Y": y, "Z": z, "X2": x*x, "Y2": y*y, "Z2": z*z,
              "XY": x*y, "YZ": y*z, "ZX": z*x}
    curves, summaries = {}, {}
    for name, sequence in series.items():
        rho, status = _autocorrelation_fft(sequence, max_lag)
        finite = np.isfinite(rho)
        first_zero = next((i for i in range(1, len(rho)) if finite[i] and rho[i] <= 0), None)
        sustained = None
        run = 5
        for i in range(1, max(1, len(rho) - run + 1)):
            if np.all(finite[i:i+run]) and np.all(np.abs(rho[i:i+run]) <= threshold):
                sustained = i
                break
        curves[name] = [float(v) if np.isfinite(v) else None for v in rho]
        summaries[name] = {
            "status": status, "first_zero_crossing_lag": first_zero,
            "first_five_lag_threshold_crossing": sustained,
            "threshold": threshold,
        }
    out = _new_output_dir("mixing_correlations_")
    curve_path = out / "autocorrelation_curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lag", *series.keys()])
        for lag in range(max_lag + 1):
            writer.writerow([lag, *[curves[name][lag] for name in series]])
    result = {
        "status": "success", "trajectory_csv": str(path),
        "transient_discarded": transient, "samples_analyzed": len(values),
        "max_lag": max_lag, "normalization": "unbiased finite-record covariance divided by lag-zero covariance; rho(0)=1",
        "correlation_decay_summaries": summaries,
        "autocorrelation_preview_first_12_lags": {name: values[:12] for name, values in curves.items()},
        "artifacts": {"directory": str(out), "curves_csv": str(curve_path)},
        "interpretation_limits": [
            "Autocorrelation functions are estimated from one finite deterministic trajectory.",
            "A finite-lag decrease or threshold crossing is not proof of asymptotic correlation decay.",
            "Mixing is an asymptotic property and cannot be established by this finite-record diagnostic alone.",
            "Oscillatory correlations and sticky or intermittent dynamics can make simple decay-time summaries misleading."
        ],
    }
    (out / "mixing_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json.dumps(result, indent=2)


@tool
def run_ergodicity_variance_study(
    kr: float, ktheta: float = 0.0, n_trajectories: int = 8,
    nsteps: int = 10000, transient: int = 1000, seed: int = 20261124,
    p: float = 1.5707963267948966,
) -> str:
    """Compare finite-time Birkhoff averages across uniform-sphere initial conditions.

    For the area-preserving DKT map, the uniform spherical-area averages of
    X,Y,Z,X²,Y²,Z²,XY,YZ,ZX are known exactly. The study estimates the
    across-initial-condition variance of finite-time averages at increasing
    observation lengths. This is finite-ensemble evidence, not a proof of
    ergodicity or global invariant-component structure.
    """
    import csv
    import json
    import numpy as np
    for name, value in (("kr", kr), ("ktheta", ktheta), ("p", p)):
        _finite_float(name, value)
    if isinstance(n_trajectories, bool) or not isinstance(n_trajectories, int) or not 3 <= n_trajectories <= 24:
        raise ValueError("n_trajectories must be between 3 and 24.")
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or not 100 <= nsteps <= 50000:
        raise ValueError("nsteps must be between 100 and 50000.")
    if isinstance(transient, bool) or not isinstance(transient, int) or not 0 <= transient < nsteps - 1:
        raise ValueError("transient must satisfy 0 <= transient < nsteps-1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    n_analyzed = nsteps - transient
    checkpoints = [2**k for k in range(5, int(np.log2(n_analyzed)) + 1) if 2**k <= n_analyzed]
    checkpoints = sorted(set(checkpoints + [n_analyzed]))
    rng = np.random.default_rng(seed)
    labels = ["X", "Y", "Z", "X2", "Y2", "Z2", "XY", "YZ", "ZX"]
    reference = np.array([0., 0., 0., 1/3, 1/3, 1/3, 0., 0., 0.])
    all_means = np.empty((n_trajectories, len(checkpoints), len(labels)), dtype=float)
    initial_conditions = []
    run_metadata = []
    for index in range(n_trajectories):
        z0 = float(rng.uniform(-1.0, 1.0))
        phi0 = float(rng.uniform(-np.pi, np.pi))
        theta0 = float(np.arccos(z0))
        sim = _simulate_dkt(
            kr=float(kr), ktheta=float(ktheta), theta0=theta0, phi0=phi0,
            nsteps=nsteps, p=float(p), include_trajectory=False,
        )
        points = load_trajectory(sim["trajectory_csv"])[transient + 1:]
        x, y, z = points.T
        observables = np.column_stack((x, y, z, x*x, y*y, z*z, x*y, y*z, z*x))
        for j, count in enumerate(checkpoints):
            all_means[index, j, :] = np.mean(observables[:count], axis=0)
        initial_conditions.append({"trajectory": index + 1, "theta0": theta0, "phi0": phi0})
        run_metadata.append({
            "trajectory": index + 1, "trajectory_csv": sim["trajectory_csv"],
            "trajectory_rows": sim["trajectory_rows"], "max_norm_error": sim["max_norm_error"],
        })
    out = _new_output_dir("ergodicity_variance_")
    csv_path = out / "birkhoff_variance_scaling.csv"
    summary_rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["analyzed_steps", "observable", "ensemble_mean", "ensemble_variance",
                  "ensemble_standard_deviation", "reference_area_mean", "squared_bias_from_reference"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for j, count in enumerate(checkpoints):
            for q, name in enumerate(labels):
                samples = all_means[:, j, q]
                mean = float(np.mean(samples))
                variance = float(np.var(samples, ddof=1))
                row = {
                    "analyzed_steps": count, "observable": name,
                    "ensemble_mean": mean, "ensemble_variance": variance,
                    "ensemble_standard_deviation": float(np.sqrt(variance)),
                    "reference_area_mean": float(reference[q]),
                    "squared_bias_from_reference": float((mean-reference[q])**2),
                }
                writer.writerow(row)
                summary_rows.append(row)
    result = {
        "status": "success", "parameters": {
            "kr": float(kr), "ktheta": float(ktheta), "p": float(p),
            "n_trajectories": n_trajectories, "nsteps": nsteps,
            "transient_discarded": transient, "seed": seed,
            "initial_condition_sampling": "independent uniform spherical-area samples: z uniform on [-1,1], phi uniform on [-pi,pi]",
            "analyzed_steps_checkpoints": checkpoints,
        },
        "reference_measure": "uniform spherical-area measure; exact moments for the nine listed Cartesian observables",
        "initial_conditions": initial_conditions, "simulation_runs": run_metadata,
        "variance_scaling": summary_rows,
        "artifacts": {"directory": str(out), "csv": str(csv_path)},
        "interpretation_limits": [
            "The variance is across the finite set of sampled initial conditions, not a formal confidence interval.",
            "Decay of finite-time Birkhoff-average variance is consistent with ergodic behaviour for the tested observables and ensemble but does not prove ergodicity.",
            "Non-decay can reflect finite-time effects, regular components, sticky trajectories, or insufficient ensemble sampling.",
            "The reference values are spherical-area averages; the map's area-preserving structure is assumed."
        ],
    }
    (out / "ergodicity_variance_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json.dumps(result, indent=2)


@tool
def run_lyapunov_parameter_scan(
    parameter: str, values: list[float], kr: float = 1.0,
    ktheta: float = 0.0, theta0: float = 0.75, phi0: float = -1.0,
    p: float = 1.5707963267948966, nsteps: int = 20000,
    transient: int = 1000, timeout: int = 900,
) -> str:
    """Run an initial-condition-specific finite-time Lyapunov/KS scan over kr or ktheta.

    Varies one parameter only. A kr scan can locate candidate changes in
    orbit-specific finite-time exponents for follow-up; it does not determine
    a global critical transition. A ktheta scan at fixed kr is a geometry/
    consistency comparison, not a chaos-control scan.
    """
    import csv
    import json
    if parameter not in {"kr", "ktheta"}:
        raise ValueError("parameter must be 'kr' or 'ktheta'.")
    if not isinstance(values, list) or not 2 <= len(values) <= 12:
        raise ValueError("values must contain 2–12 distinct finite numbers.")
    clean = [_finite_float("sweep value", v) for v in values]
    if len(set(clean)) != len(clean):
        raise ValueError("values must be distinct.")
    for name, value in (("kr", kr), ("ktheta", ktheta), ("theta0", theta0), ("phi0", phi0), ("p", p)):
        _finite_float(name, value)
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or not 100 <= nsteps <= 500000:
        raise ValueError("nsteps must be between 100 and 500000.")
    if isinstance(transient, bool) or not isinstance(transient, int) or not 0 <= transient <= nsteps - 100:
        raise ValueError("transient must leave at least 100 analyzed kicks.")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 3600:
        raise ValueError("timeout must be an integer between 1 and 3600 seconds.")
    out = _new_output_dir("lyapunov_parameter_scan_")
    rows = []
    for index, value in enumerate(clean):
        run_kr = value if parameter == "kr" else float(kr)
        run_ktheta = value if parameter == "ktheta" else float(ktheta)
        checkpoint_path = out / f"{parameter}_{index+1}_checkpoints.csv"
        checkpoints = _run_julia_lyapunov(
            kr=run_kr, ktheta=run_ktheta, theta0=float(theta0), phi0=float(phi0),
            p=float(p), nsteps=nsteps, transient=transient,
            output_csv=checkpoint_path, timeout=timeout,
        )
        final = checkpoints[-1]
        rows.append({
            "varied_value": value, "kr": run_kr, "ktheta": run_ktheta,
            "lambda1": final["lambda1"], "lambda2": final["lambda2"],
            "sum_exponents": final["sum_exponents"],
            "ks_pesin_estimate": final["ks_pesin_estimate"],
            "checkpoint_csv": str(checkpoint_path),
        })
    summary_path = out / "lyapunov_parameter_scan.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["varied_value", "kr", "ktheta", "lambda1", "lambda2", "sum_exponents", "ks_pesin_estimate", "checkpoint_csv"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "success", "parameter": parameter, "values": clean,
        "fixed_parameters": {"kr": float(kr), "ktheta": float(ktheta),
                             "theta0": float(theta0), "phi0": float(phi0),
                             "p": float(p), "nsteps": nsteps, "transient": transient},
        "results": rows, "summary_csv": str(summary_path), "artifact_directory": str(out),
        "interpretation_limits": [
            "The scan uses one initial condition and finite-time Lyapunov estimates at each parameter value.",
            "A kr scan can suggest orbit-specific regime changes but cannot by itself identify a global critical kr or phase-space-wide transition.",
            "A ktheta scan at fixed kr probes numerical/trajectory dependence associated with phase-space geometry; it must not be interpreted as ktheta changing the degree of classical chaos.",
            "Check finite-time checkpoint drift and repeat with additional initial conditions before drawing broader conclusions."
        ],
    }
    (out / "lyapunov_parameter_scan_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json.dumps(result, indent=2)


@tool
def run_dkt_geometry_ensemble_comparison(
    kr: float,
    ktheta_values: list[float],
    n_trajectories: int = 8,
    nsteps: int = 2000,
    p: float = 1.5707963267948966,
    seed: int = 20261126,
    n_z: int = 30,
    n_phi: int = 60,
) -> str:
    """Compare matched ensembles at two ktheta values while holding kr and p fixed.

    The same reproducible uniform-sphere initial conditions are used for both
    parameter values. It reports pooled empirical phase-space measure
    Jensen-Shannon divergence and per-initial-condition divergences. This is a
    finite-sample geometry/distribution comparison, not a chaos or ergodicity
    test and not a proof that invariant measures differ.
    """
    import csv
    import json
    import math
    import numpy as np

    if not math.isfinite(float(kr)) or not math.isfinite(float(p)):
        raise ValueError("kr and p must be finite.")
    if not isinstance(ktheta_values, list) or len(ktheta_values) != 2:
        raise ValueError("ktheta_values must contain exactly two values.")
    values = [float(value) for value in ktheta_values]
    if not all(math.isfinite(value) for value in values) or values[0] == values[1]:
        raise ValueError("ktheta_values must be distinct finite values.")
    if isinstance(n_trajectories, bool) or not isinstance(n_trajectories, int) or not 2 <= n_trajectories <= 24:
        raise ValueError("n_trajectories must be between 2 and 24.")
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or not 100 <= nsteps <= 50000:
        raise ValueError("nsteps must be between 100 and 50000.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if isinstance(n_z, bool) or not isinstance(n_z, int) or not 2 <= n_z <= 100:
        raise ValueError("n_z must be between 2 and 100.")
    if isinstance(n_phi, bool) or not isinstance(n_phi, int) or not 2 <= n_phi <= 200:
        raise ValueError("n_phi must be between 2 and 200.")

    rng = np.random.default_rng(seed)
    labels = [f"ktheta={values[0]:g}", f"ktheta={values[1]:g}"]
    pooled = {label: [] for label in labels}
    initial_conditions = []
    runs = []
    matched_divergences = []

    for index in range(n_trajectories):
        z0 = float(rng.uniform(-1.0, 1.0))
        phi0 = float(rng.uniform(-np.pi, np.pi))
        theta0 = float(np.arccos(z0))
        initial_conditions.append({
            "ensemble_index": index + 1, "theta0": theta0, "phi0": phi0,
        })
        pair_points = {}
        for value_index, value in enumerate(values):
            sim = _simulate_dkt(
                kr=float(kr), ktheta=value, theta0=theta0, phi0=phi0,
                nsteps=nsteps, p=float(p), include_trajectory=False,
            )
            points = load_trajectory(sim["trajectory_csv"])
            pooled[labels[value_index]].append(points)
            pair_points[labels[value_index]] = points
            runs.append({
                "ensemble_index": index + 1,
                "ktheta": value,
                "parameters": sim["parameters"],
                "trajectory_rows": sim["trajectory_rows"],
                "max_norm_error": sim["max_norm_error"],
                "trajectory_csv": sim["trajectory_csv"],
            })
        pair_result = empirical_measure_comparison(
            pair_points, n_z=n_z, n_phi=n_phi
        )
        matched_divergences.append({
            "ensemble_index": index + 1,
            "theta0": theta0,
            "phi0": phi0,
            "js_divergence_base2": float(pair_result["js_divergence"][0, 1]),
        })

    pooled_points = {
        label: np.concatenate(arrays, axis=0)
        for label, arrays in pooled.items()
    }
    pooled_result = empirical_measure_comparison(
        pooled_points, n_z=n_z, n_phi=n_phi
    )
    out = _new_output_dir("geometry_ensemble_comparison_")
    matched_csv = out / "matched_initial_condition_js.csv"
    with matched_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["ensemble_index", "theta0", "phi0", "js_divergence_base2"]
        )
        writer.writeheader()
        writer.writerows(matched_divergences)

    result = {
        "status": "success",
        "parameters": {
            "kr": float(kr), "ktheta_values": values, "p": float(p),
            "n_trajectories": n_trajectories, "nsteps": nsteps,
            "seed": seed, "n_z": n_z, "n_phi": n_phi,
            "initial_condition_sampling": "matched independent uniform-sphere samples: z uniform on [-1,1], phi uniform on [-pi,pi]",
            "samples_per_parameter": int(n_trajectories * (nsteps + 1)),
        },
        "initial_conditions": initial_conditions,
        "simulation_runs": runs,
        "pooled_empirical_measure_comparison": {
            "labels": labels,
            "jensen_shannon_divergence_base2": pooled_result["js_divergence"].tolist(),
            "partition": {"n_z": n_z, "n_phi": n_phi},
        },
        "matched_initial_condition_divergences": matched_divergences,
        "artifacts": {
            "directory": str(out),
            "matched_divergence_csv": str(matched_csv),
        },
        "interpretation_limits": [
            "The same initial conditions, kick count, kr and p are used for both ktheta values.",
            "The pooled and per-initial-condition Jensen-Shannon divergences compare finite empirical distributions on the selected equal-area partition.",
            "Results depend on ensemble size, observation window, seed and partition resolution.",
            "A nonzero divergence is evidence of different sampled trajectory distributions, not proof of distinct invariant measures.",
            "At fixed kr, differences are interpreted as phase-space geometry/distribution changes; they must not be attributed to changes in classical chaoticity, Lyapunov behaviour or KS entropy.",
            "Uniform full-sphere initial-condition sampling is used; this tool does not establish properties restricted to a local phase-space region.",
        ],
    }
    with (out / "geometry_ensemble_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return json.dumps(result, indent=2)
