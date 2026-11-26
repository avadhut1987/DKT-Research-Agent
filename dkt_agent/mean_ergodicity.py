"""
mean_ergodicity.py -- L2 (von Neumann) mean-ergodicity diagnostics for the
classical double kicked top (DKT) map on the unit sphere.

Why this is different from the Birkhoff (pointwise) test
--------------------------------------------------------
For a measure-preserving map T on (S^2, mu) the von Neumann theorem says that
the time average A_N f = (1/N) sum_{n<N} f o T^n ALWAYS converges in L2(mu)
to P f, the projection of f onto T-invariant functions.  So "mean ergodic" in
the operational sense used for the DKT means: the L2 limit is the *constant*
<f>.  The quantity that detects this is

    D_N(f) = || A_N f - <f> ||^2_{L2(mu)} = E_{x0 ~ mu}[ (A_N f(x0) - <f>)^2 ]

    * D_N -> 0                      : T is mean ergodic w.r.t. f (limit constant)
    * D_N -> D_inf > 0 (plateau)    : A_N f -> Pf != <f>; D_inf = ||Pf - <f>||^2
                                      measures the non-ergodic (invariant) part.

This is weaker than Birkhoff (a.e. convergence) and is estimated here with an
unbiased Monte-Carlo average over uniformly drawn initial conditions, with a
standard error, so the plateau/decay can be judged against statistical noise.

Two independent estimators are provided and cross-checked:
  (1) direct   : sample mean of (A_N f(x0) - <f>)^2
  (2) spectral : D_N = (1/N)[C(0) + 2 sum_{k=1}^{N-1} (1 - k/N) C(k)]
                 from the autocorrelation C(k) = <f0 . f0 o T^k>, f0 = f - <f>.

Conventions
-----------
Floquet operator  U = exp(-i k'/2j Jx^2) exp(-i k/2j Jz^2) exp(-i pi/2 Jy).
Classical map on X = J/j (apply right to left): rotate pi/2 about y, twist
about z by angle k*z, twist about x by angle k'*x.  Orientation conventions
do not affect ergodic properties.  If your project uses (k_r, k_theta), pass
your own step function (or convert to (k, k') before calling) -- this module
does not assume the transformation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Classical map
# --------------------------------------------------------------------------

def dkt_step(x, y, z, k: float, kp: float, p: float = np.pi / 2):
    """One period of the classical DKT map (vectorised, area preserving).
    p is the rotation angle about y (Floquet factor exp(-i p Jy)); p = pi/2 default."""
    if p == np.pi / 2:
        x, y, z = z, y, -x                   # exact R_y(pi/2)
    else:
        c0, s0 = np.cos(p), np.sin(p)
        x, z = x * c0 + z * s0, -x * s0 + z * c0
    a = k * z                                # twist about z
    c, s = np.cos(a), np.sin(a)
    x, y = x * c - y * s, x * s + y * c
    a = kp * x                               # twist about x
    c, s = np.cos(a), np.sin(a)
    y, z = y * c - z * s, y * s + z * c
    return x, y, z


def kr_ktheta_to_kkp(kr: float, ktheta: float):
    """Paper's Eq. (5) inverted: k = k_r + k_theta, k' = k_r - k_theta."""
    return kr + ktheta, kr - ktheta


def kkp_to_kr_ktheta(k: float, kp: float):
    """Purohit & Bhosale, PRE 112, 014217, Eq. (5): k_r=(k+k')/2, k_theta=(k-k')/2."""
    return 0.5 * (k + kp), 0.5 * (k - kp)


def sample_sphere(M: int, rng: np.random.Generator):
    """M points uniform on S^2 (uniform in cos(theta), phi)."""
    z = rng.uniform(-1.0, 1.0, M)
    phi = rng.uniform(0.0, 2.0 * np.pi, M)
    r = np.sqrt(1.0 - z * z)
    return r * np.cos(phi), r * np.sin(phi), z

# --------------------------------------------------------------------------
# Observables (monomials x^a y^b z^c with exact sphere average)
# --------------------------------------------------------------------------

def _dfact(n: int) -> int:
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


@dataclass
class Observable:
    name: str
    func: Callable
    mean: float          # <f> w.r.t. uniform measure on S^2
    var: float = np.nan  # Var(f) w.r.t. uniform measure (if known)


def monomial(a: int, b: int, c: int, name: Optional[str] = None) -> Observable:
    """f = x^a y^b z^c with analytic mean and variance on the sphere."""
    def mom(a, b, c):
        if a % 2 or b % 2 or c % 2:
            return 0.0
        return _dfact(a - 1) * _dfact(b - 1) * _dfact(c - 1) / _dfact(a + b + c + 1)
    m = mom(a, b, c)
    v = mom(2 * a, 2 * b, 2 * c) - m * m
    return Observable(name or f"x^{a} y^{b} z^{c}",
                      lambda x, y, z: x**a * y**b * z**c, m, v)


def parse_observable(spec) -> Observable:
    """'z', 'x2', 'xy', 'x2y2', ... or an Observable / (a,b,c) tuple."""
    if isinstance(spec, Observable):
        return spec
    if isinstance(spec, tuple):
        return monomial(*spec)
    exps = {"x": 0, "y": 0, "z": 0}
    toks = re.findall(r"([xyz])(\d*)", spec)
    if not toks or "".join(a + b for a, b in toks) != spec:
        raise ValueError(f"cannot parse observable {spec!r}; use e.g. 'z', 'x2', 'xy'")
    for ax, p in toks:
        exps[ax] += int(p) if p else 1
    return monomial(exps["x"], exps["y"], exps["z"], name=spec)

# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------

@dataclass
class MeanErgodicResult:
    label: str
    obs: Observable
    N: int
    M: int
    Ns: np.ndarray                 # checkpoints
    D: np.ndarray                  # direct estimate of D_N
    D_se: np.ndarray               # its standard error
    D_corr: np.ndarray             # correlation-based estimate of D_N
    C: np.ndarray                  # autocorrelation C(k), k = 0..N-1
    dev: np.ndarray                # (M, n_ck) A_N f(x0) - <f>
    x0: np.ndarray                 # (M, 3) initial conditions
    alpha: float = np.nan          # D_N ~ N^-alpha over the tail
    alpha_err: float = np.nan
    verdict: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def D_final(self):
        return self.D[-1], self.D_se[-1]


def _checkpoints(N: int, n: int) -> np.ndarray:
    ck = np.unique(np.round(np.logspace(0, np.log10(N), n)).astype(int))
    return ck[ck >= 1]


def _fit_tail(Ns, D, se, tail_decades):
    mask = (Ns >= Ns[-1] / 10 ** tail_decades) & (D > 0)
    if mask.sum() < 4:
        return np.nan, np.nan
    lx, ly = np.log(Ns[mask]), np.log(D[mask])
    sig = np.clip(se[mask] / D[mask], 1e-3, None)
    p, cov = np.polyfit(lx, ly, 1, w=1.0 / sig, cov="unscaled")
    return -p[0], float(np.sqrt(cov[0, 0]))


def _verdict(alpha, alpha_err, D_last, se_last):
    """Heuristic classification -- thresholds are conventions, not theorems."""
    if np.isnan(alpha):
        return "inconclusive (too few usable points)"
    if alpha - 2 * alpha_err > 0.4:
        return (f"DECAY  D_N ~ N^-{alpha:.2f}: finite-data indicator consistent with "
                f"an L2 limit equal to the constant <f> for this observable")
    if alpha + 2 * alpha_err < 0.15:
        return (f"PLATEAU  D_N -> {D_last:.3g} +/- {se_last:.1g}: finite-data indicator that "
                f"A_N f tends to a NON-constant invariant function for this observable")
    return (f"INCONCLUSIVE  local exponent {alpha:.2f} +/- {alpha_err:.2f}; "
            f"slow decay or transient -- increase N and M")


def mean_ergodic_analysis(
    k: float = 3.0,
    kp: float = 1.0,
    obs="z",
    N: int = 2 ** 14,
    M: int = 4000,
    n_checkpoints: int = 45,
    seed: int = 0,
    step: Optional[Callable] = None,
    tail_decades: float = 1.0,
    mem_budget: float = 2e7,
    label: Optional[str] = None,
    kr: Optional[float] = None,
    ktheta: Optional[float] = None,
    p: float = np.pi / 2,
) -> MeanErgodicResult:
    """Estimate D_N(f) = ||A_N f - <f>||^2_{L2} and the exponent of its decay.

    step : optional callable (x, y, z) -> (x, y, z) to override the built-in
           DKT map.
    p : rotation angle about y in exp(-i p Jy) (default pi/2).
    kr, ktheta : if given, override (k, k') using k = k_r + k_theta,
           k' = k_r - k_theta (paper's Eq. 5).
    """
    if kr is not None or ktheta is not None:
        kr = 0.0 if kr is None else kr
        ktheta = 0.0 if ktheta is None else ktheta
        k, kp = kr_ktheta_to_kkp(kr, ktheta)
        label = label or f"k_r={kr:g}, k_theta={ktheta:g}"
    f = parse_observable(obs)
    step = step or partial(dkt_step, k=k, kp=kp, p=p)
    rng = np.random.default_rng(seed)
    X0 = np.stack(sample_sphere(M, rng), axis=1)
    Ns = _checkpoints(N, n_checkpoints)
    dev = np.empty((M, len(Ns)))
    C_acc = np.zeros(N)
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    chunk = max(100, int(mem_budget / N))

    for lo in range(0, M, chunk):
        hi = min(M, lo + chunk)
        x, y, z = X0[lo:hi, 0].copy(), X0[lo:hi, 1].copy(), X0[lo:hi, 2].copy()
        F = np.empty((hi - lo, N))
        for n in range(N):
            F[:, n] = f.func(x, y, z)
            x, y, z = step(x, y, z)
        F -= f.mean                                   # f0 = f - <f>
        dev[lo:hi] = np.cumsum(F, axis=1)[:, Ns - 1] / Ns
        for s in range(0, hi - lo, 200):              # FFT autocorrelation
            Fk = np.fft.rfft(F[s:s + 200], nfft, axis=1)
            ac = np.fft.irfft(Fk * np.conj(Fk), nfft, axis=1)[:, :N]
            C_acc += ac.sum(axis=0)
    # unbiased lag normalisation, average over all trajectories
    C = C_acc / M / (N - np.arange(N))

    sq = dev ** 2
    D = sq.mean(axis=0)
    D_se = sq.std(axis=0, ddof=1) / np.sqrt(M)

    S0, S1 = np.cumsum(C), np.cumsum(np.arange(N) * C)
    Nm = Ns - 1
    D_corr = (2 * S0[Nm] - C[0] - 2 * S1[Nm] / Ns) / Ns

    alpha, aerr = _fit_tail(Ns, D, D_se, tail_decades)
    res = MeanErgodicResult(
        label=label or f"k={k:g}, k'={kp:g}", obs=f, N=N, M=M, Ns=Ns, D=D,
        D_se=D_se, D_corr=D_corr, C=C, dev=dev, x0=X0, alpha=alpha,
        alpha_err=aerr, verdict=_verdict(alpha, aerr, D[-1], D_se[-1]),
    )
    return res

# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render(res: MeanErgodicResult, path: Optional[str] = None, show: bool = False,
           return_fig: bool = False):
    """Four-panel figure: D_N decay, sphere map of A_N f, histograms, Cesaro means."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)
    Ns = res.Ns

    # (a) D_N ---------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(Ns, np.clip(res.D - 2 * res.D_se, 1e-30, None), res.D + 2 * res.D_se,
                    color="C0", alpha=0.25, lw=0)
    ax.loglog(Ns, res.D, "o-", ms=3, color="C0", label=r"direct $D_N$ (MC, $\pm2$ s.e.)")
    ax.loglog(Ns, np.clip(res.D_corr, 1e-30, None), "--", color="C3", lw=1.3,
              label=r"from $C(k)$ (cross-check)")
    i0 = min(2, len(Ns) - 1)
    ax.loglog(Ns, res.D[i0] * (Ns / Ns[i0]) ** -1.0, ":", color="k", lw=1, label=r"$\propto N^{-1}$")
    if np.isfinite(res.alpha):
        m = Ns >= Ns[-1] / 10
        ref = res.D[m][0] * (Ns[m] / Ns[m][0]) ** (-res.alpha)
        ax.loglog(Ns[m], ref, "-", color="C1", lw=2.2,
                  label=rf"tail fit $N^{{-{res.alpha:.2f}\pm{res.alpha_err:.2f}}}$")
    ax.set_xlabel("N"); ax.set_ylabel(r"$D_N=\|A_Nf-\langle f\rangle\|^2_{L^2}$")
    ax.set_title(f"(a) mean-ergodic distance, f={res.obs.name}")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, which="both")

    # (b) sphere map ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1], projection="mollweide")
    x0, y0, z0 = res.x0.T
    lon, lat = np.arctan2(y0, x0), np.arcsin(np.clip(z0, -1, 1))
    d = res.dev[:, -1]
    vmax = max(np.percentile(np.abs(d), 99), 1e-12)
    sc = ax.scatter(lon, lat, c=d, s=5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(rf"(b) $A_Nf(x_0)-\langle f\rangle$ at $N={Ns[-1]}$", fontsize=10)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
    cb.ax.tick_params(labelsize=7)

    # (c) histograms ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    idx = sorted(set(np.linspace(0, len(Ns) - 1, 4).astype(int)))
    allv = np.abs(res.dev[:, idx]).max()
    bins = np.linspace(-allv, allv, 70)
    for j, i in enumerate(idx):
        ax.hist(res.dev[:, i], bins=bins, histtype="step", lw=1.6, color=f"C{j}",
                density=True, label=f"N={Ns[i]}")
    ax.set_yscale("log"); ax.set_xlabel(r"$A_Nf(x_0)-\langle f\rangle$")
    ax.set_ylabel("density")
    ax.set_title("(c) distribution over initial conditions\n(collapse to 0 <=> ergodic limit)", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)

    # (d) Cesaro means of the correlation --------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    n = np.arange(1, res.N + 1)
    c0 = res.C[0] if res.C[0] != 0 else 1.0
    ces = np.cumsum(res.C) / n / c0
    ces_abs = np.cumsum(np.abs(res.C)) / n / c0
    ax.semilogx(n, ces, color="C0", lw=1.4, label=r"$\frac{1}{n}\sum_{k<n} C(k)/C(0)$  (mean-ergodic)")
    ax.semilogx(n, ces_abs, color="C2", lw=1.0, alpha=0.8,
                label=r"$\frac{1}{n}\sum_{k<n} |C(k)|/C(0)$  (weak-mixing, for contrast)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("n"); ax.set_title("(d) Cesàro means of the autocorrelation", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)

    fig.suptitle(f"Mean ergodicity | {res.label} | M={res.M}, N={res.N}\n{res.verdict}",
                 fontsize=10.5, y=0.995)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    if return_fig:
        return fig
    plt.close(fig)
    return path

def render_portraits(params, path=None, n_side=25, n_kicks=200, ncols=3,
                     title="DKT phase-space portraits"):
    """Phase-space portraits in the style of Figs. 1-3 of PRE 112, 014217:
    an n_side x n_side grid of initial conditions, n_kicks iterations.
    params: list of (k_r, k_theta).  Axes: phi = atan2(Y,X), cos(theta) = Z
    (the paper's projection is not specified in the text; change here to match)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(params) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows), squeeze=False)
    # grid uniform in (phi, cos theta)
    ph = np.linspace(-np.pi, np.pi, n_side, endpoint=False) + np.pi / n_side
    cz = np.linspace(-1, 1, n_side, endpoint=False) + 1.0 / n_side
    P, C = np.meshgrid(ph, cz)
    r = np.sqrt(1 - C ** 2)
    for ax, (kr, kt) in zip(axs.ravel(), params):
        k, kp = kr_ktheta_to_kkp(kr, kt)
        x, y, z = (r * np.cos(P)).ravel(), (r * np.sin(P)).ravel(), C.ravel()
        xs, ys, zs = [x], [y], [z]
        for _ in range(n_kicks):
            x, y, z = dkt_step(x, y, z, k, kp)
            xs.append(x); ys.append(y); zs.append(z)
        X, Y, Z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
        ax.plot(np.arctan2(Y, X), Z, ",", color="k", alpha=0.6)
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-1, 1)
        ax.set_title(rf"$k_r={kr:g},\ k_\theta={kt:g}$", fontsize=10)
        ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$Z=\cos\theta$")
    for ax in axs.ravel()[len(params):]:
        ax.axis("off")
    fig.suptitle(f"{title} ({n_side}x{n_side} ICs, {n_kicks} kicks)", y=1.0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Observable battery (guards against symmetry-blind observables)
# --------------------------------------------------------------------------

DEFAULT_BATTERY = ("x", "y", "z", "x2", "y2", "z2", "xz", "xy", "yz")


def mean_ergodic_battery(kr=None, ktheta=None, k=None, kp=None,
                         observables=DEFAULT_BATTERY, **kw):
    """Run the analysis for several observables and report the worst case.

    A single observable can be blind to non-ergodicity by symmetry (e.g. z
    averages to ~0 on tori around the y-axis even though phase space is
    regular).  'Mean ergodic' is only credible if D_N decays for ALL tested f;
    the worst-case D_N is the quantity to report.
    """
    results = {}
    for o in observables:
        if kr is not None or ktheta is not None:
            results[o] = mean_ergodic_analysis(kr=kr, ktheta=ktheta, obs=o, **kw)
        else:
            results[o] = mean_ergodic_analysis(k=k, kp=kp, obs=o, **kw)
    # normalise by Var(f) so observables are comparable
    rel = {o: r.D[-1] / r.obs.var for o, r in results.items()}
    worst = max(rel, key=rel.get)
    summary = dict(worst_observable=worst, worst_rel_D=rel[worst], rel_D=rel,
                   all_decay=all(r.verdict.startswith("DECAY") for r in results.values()))
    return results, summary


def render_battery(table: dict, path: Optional[str] = None, title=""):
    """table: {label: summary['rel_D']} -> heat map of log10(D_final/Var f)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(table)
    obs = list(next(iter(table.values())).keys())
    M = np.array([[table[l][o] for o in obs] for l in labels])
    fig, ax = plt.subplots(figsize=(1.0 * len(obs) + 3, 0.5 * len(labels) + 2))
    im = ax.imshow(np.log10(np.clip(M, 1e-12, None)), cmap="magma_r", aspect="auto",
                   vmin=-8, vmax=0)
    ax.set_xticks(range(len(obs))); ax.set_xticklabels(obs)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{np.log10(max(M[i, j], 1e-12)):.1f}", ha="center", va="center",
                    fontsize=7, color="w" if np.log10(max(M[i, j], 1e-12)) > -3.5 else "k")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\,D_N/\mathrm{Var}(f)$ at largest N")
    ax.set_title(title or "Mean-ergodic distance per observable (dark = non-ergodic part)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Parameter scan
# --------------------------------------------------------------------------

def scan_parameters(k_values: Sequence[float], kp_values: Sequence[float], obs="z",
                    N: int = 2 ** 12, M: int = 800, seed: int = 0, **kw):
    """Grid scan: returns arrays alpha, D_final indexed [i_kp, i_k]."""
    A = np.full((len(kp_values), len(k_values)), np.nan)
    Df = np.full_like(A, np.nan)
    for i, kp in enumerate(kp_values):
        for j, k in enumerate(k_values):
            r = mean_ergodic_analysis(k, kp, obs, N=N, M=M, seed=seed,
                                      n_checkpoints=25, **kw)
            A[i, j], Df[i, j] = r.alpha, r.D[-1]
    return dict(k=np.asarray(k_values), kp=np.asarray(kp_values), alpha=A, D_final=Df,
                obs=parse_observable(obs).name, N=N, M=M)


def render_scan(scan: dict, path: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ext = [scan["k"][0], scan["k"][-1], scan["kp"][0], scan["kp"][-1]]
    im = axs[0].imshow(scan["alpha"], origin="lower", extent=ext, aspect="auto",
                       cmap="viridis", vmin=0, vmax=1.2)
    axs[0].set_title(r"tail exponent $\alpha$ ($D_N\sim N^{-\alpha}$)")
    fig.colorbar(im, ax=axs[0])
    Df = np.clip(scan["D_final"], 1e-12, None)
    im = axs[1].imshow(np.log10(Df), origin="lower", extent=ext, aspect="auto", cmap="magma")
    axs[1].set_title(r"$\log_{10} D_N$ at largest N (high = non-ergodic part)")
    fig.colorbar(im, ax=axs[1])
    for a in axs:
        a.set_xlabel("k"); a.set_ylabel("k'")
    fig.suptitle(f"Mean-ergodicity scan, f={scan['obs']}, N={scan['N']}, M={scan['M']}")
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mean-ergodicity diagnostics for the DKT map")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--obs", default="z")
    ap.add_argument("--N", type=int, default=2 ** 14)
    ap.add_argument("--M", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mean_ergodicity.png")
    a = ap.parse_args()
    r = mean_ergodic_analysis(a.k, a.kp, a.obs, a.N, a.M, seed=a.seed)
    print(r.verdict)
    print("saved", render(r, a.out))
