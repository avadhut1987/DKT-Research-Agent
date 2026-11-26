"""
Agent-facing layer for the von Neumann (L2) mean-ergodicity diagnostic.

* run_mean_ergodic_diagnostic(...)  -> plain, JSON-serialisable dict (no LLM,
  no Julia, numpy/matplotlib only).  Saves JSON + CSV + PNG artifacts under
  results/diagnostics/mean_ergodicity/<parameter-labelled folder>/.
* mean_ergodicity_langchain_tool    -> LangChain tool wrapping the function
  (created only if langchain_core is importable).

Scientific wording follows the project README: outputs are finite-data
indicators, not proofs of (mean-)ergodicity.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from dkt_agent.mean_ergodicity import (
    DEFAULT_BATTERY,
    mean_ergodic_analysis,
    mean_ergodic_battery,
    render,
    render_battery,
)

RESULTS_ROOT = Path("results/diagnostics/mean_ergodicity")

CAVEAT = (
    "Finite-data indicator only. D_N = ||A_N f - <f>||^2_L2 is estimated by Monte-Carlo "
    "over uniformly sampled initial conditions (with standard errors) for the listed "
    "observables; a decaying D_N for some f does not establish mean ergodicity for all "
    "f, and a plateau is evidence of a non-constant L2 limit for that f only. "
    "Check N, M and seed sensitivity before drawing conclusions."
)


def _f(x):
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else x


def _label(kr, kt, p, N, M, seed):
    return f"kr{kr:g}_kt{kt:g}_p{p:.4f}_N{N}_M{M}_seed{seed}"


def save_mean_ergodic_artifacts(
    k_r: float,
    k_theta: float,
    observables: Sequence[str],
    n_kicks: int,
    n_initial_conditions: int,
    p: float,
    seed: int,
    results: dict,
    summary: dict,
    results_root: Optional[str] = None,
) -> dict:
    """Build a serialisable summary and save artifacts from already-computed results.

    This lets the Streamlit panel save its in-memory calculation without
    running the comparatively expensive trajectory ensemble a second time.
    """
    per_obs = {}
    for o, r in results.items():
        D, se = r.D_final
        per_obs[o] = dict(
            D_final=_f(D), D_standard_error=_f(se), D_over_variance=_f(summary["rel_D"][o]),
            tail_exponent_alpha=_f(r.alpha), tail_exponent_error=_f(r.alpha_err),
            verdict=r.verdict, observable_mean=_f(r.obs.mean), observable_variance=_f(r.obs.var),
        )
    out = dict(
        diagnostic="von Neumann mean ergodicity (L2)",
        parameters=dict(k_r=k_r, k_theta=k_theta, k=k_r + k_theta, k_prime=k_r - k_theta,
                        p=p, n_kicks=n_kicks, n_initial_conditions=n_initial_conditions,
                        seed=seed),
        observables=per_obs,
        worst_observable=summary["worst_observable"],
        worst_D_over_variance=_f(summary["worst_rel_D"]),
        all_observables_decay=bool(summary["all_decay"]),
        caveat=CAVEAT,
        created=datetime.now().isoformat(timespec="seconds"),
        artifacts={},
    )
    d = Path(results_root or RESULTS_ROOT) / _label(
        k_r, k_theta, p, n_kicks, n_initial_conditions, seed
    )
    d.mkdir(parents=True, exist_ok=True)
    for o, r in results.items():
        render(r, str(d / f"mean_ergodicity_{o}.png"))
        with open(d / f"D_N_{o}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["N", "D_direct", "D_standard_error", "D_from_autocorrelation"])
            for row in zip(r.Ns, r.D, r.D_se, r.D_corr):
                w.writerow([int(row[0])] + [f"{v:.10e}" for v in row[1:]])
    if len(results) > 1:
        render_battery({f"k_r={k_r:g}, k_θ={k_theta:g}": summary["rel_D"]},
                       str(d / "battery_summary.png"))
    names = {q.name for q in d.iterdir()} | {"mean_ergodicity_summary.json"}
    out["artifacts"] = {"directory": str(d), "files": sorted(names)}
    with open(d / "mean_ergodicity_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def run_mean_ergodic_diagnostic(
    k_r: float,
    k_theta: float = 0.0,
    observables: Sequence[str] = ("z",),
    n_kicks: int = 2 ** 14,
    n_initial_conditions: int = 3000,
    p: float = math.pi / 2,
    seed: int = 0,
    save: bool = True,
    results_root: Optional[str] = None,
) -> dict:
    """Run the mean-ergodicity diagnostic for one (k_r, k_theta, p) configuration.

    Several observables (e.g. the 9 monomials of DEFAULT_BATTERY) are strongly
    recommended: a single observable can be blind to non-ergodicity by symmetry.
    Observable names: 'x','y','z','x2','y2','z2','xy','yz','xz', ...
    """
    observables = list(observables)
    results, summary = mean_ergodic_battery(
        kr=k_r, ktheta=k_theta, observables=observables,
        N=n_kicks, M=n_initial_conditions, seed=seed, p=p,
    )
    if not save:
        # Use the same summary schema without creating output files.
        out = save_mean_ergodic_artifacts(
            k_r, k_theta, observables, n_kicks, n_initial_conditions, p, seed,
            results, summary, results_root=results_root,
        )
        out["artifacts"] = {}
        return out
    return save_mean_ergodic_artifacts(
        k_r, k_theta, observables, n_kicks, n_initial_conditions, p, seed,
        results, summary, results_root=results_root,
    )

def format_for_agent(result: dict) -> str:
    """Compact text for the LLM: numbers, verdicts, caveat."""
    pr = result["parameters"]
    lines = [f"Mean-ergodicity (L2) diagnostic: k_r={pr['k_r']:g}, k_theta={pr['k_theta']:g}, "
             f"p={pr['p']:.4f}, N={pr['n_kicks']}, M={pr['n_initial_conditions']}, seed={pr['seed']}"]
    for o, v in result["observables"].items():
        lines.append(f"  f={o}: D_N/Var(f)={v['D_over_variance']:.3e} | {v['verdict']}")
    lines.append(f"Worst observable: {result['worst_observable']} "
                 f"(D_N/Var = {result['worst_D_over_variance']:.3e}); "
                 f"all decay: {result['all_observables_decay']}")
    if result["artifacts"]:
        lines.append(f"Artifacts: {result['artifacts']['directory']}")
    lines.append("NOTE: " + result["caveat"])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Optional LangChain wrapper (does not break imports if LangChain is absent)
# --------------------------------------------------------------------------
try:  # pragma: no cover - depends on the user's environment
    from langchain_core.tools import tool as _lc_tool

    @_lc_tool
    def diagnose_mean_ergodicity(
        k_r: float,
        k_theta: float = 0.0,
        observables: Optional[list[str]] = None,
        n_kicks: int = 16384,
        n_initial_conditions: int = 3000,
        p: float = math.pi / 2,
        seed: int = 0,
    ) -> str:
        """Finite-data von Neumann (L2) mean-ergodicity diagnostic for the classical
        double kicked top with kicks k = k_r + k_theta, k' = k_r - k_theta.

        Estimates D_N = ||A_N f - <f>||^2 over uniformly sampled initial conditions
        for several Cartesian observables (default: x, y, z, x2, y2, z2, xy, yz, xz)
        and reports the decay exponent, a decay/plateau indicator per observable and
        the worst-case observable. Use it to compare e.g. k_theta=0 with k_theta=1e4
        at fixed k_r. Results are numerical indicators, not proofs."""
        res = run_mean_ergodic_diagnostic(
            k_r, k_theta, observables or DEFAULT_BATTERY, n_kicks,
            n_initial_conditions, p, seed)
        return format_for_agent(res)

    mean_ergodicity_langchain_tool = diagnose_mean_ergodicity
except ImportError:  # LangChain not installed in this environment
    mean_ergodicity_langchain_tool = None
