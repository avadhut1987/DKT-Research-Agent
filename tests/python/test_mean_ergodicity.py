"""pytest tests for mean_ergodicity.py (analytic ground truth where possible)."""
import numpy as np
import pytest

from dkt_agent.mean_ergodicity import (dkt_step, sample_sphere, parse_observable,
                             mean_ergodic_analysis)


def test_map_preserves_sphere():
    rng = np.random.default_rng(1)
    x, y, z = sample_sphere(1000, rng)
    for _ in range(2000):
        x, y, z = dkt_step(x, y, z, 5.3, 2.1)
    assert np.max(np.abs(x * x + y * y + z * z - 1)) < 1e-10


def test_observable_moments():
    assert parse_observable("x2").mean == pytest.approx(1 / 3)
    assert parse_observable("x2y2").mean == pytest.approx(1 / 15)
    assert parse_observable("z").mean == 0.0
    assert parse_observable("z").var == pytest.approx(1 / 3)
    assert parse_observable("x2").var == pytest.approx(1 / 5 - 1 / 9)


def test_integrable_plateau_matches_theory():
    # k = k' = 0: T = rotation by pi/2 about y (period 4).  For f = x^2 the
    # time average tends to (x^2+z^2)/2, so D_inf = Var((x^2+z^2)/2) = 1/45.
    r = mean_ergodic_analysis(0.0, 0.0, "x2", N=2 ** 12, M=3000, seed=3)
    D, se = r.D_final
    assert abs(D - 1 / 45) < 4 * se + 1e-4
    assert "PLATEAU" in r.verdict


def test_integrable_mean_ergodic_for_odd_observable():
    # Same map, f = z: the 4-cycle average of (z, -x, -z, x) is 0 = <f>, so D_N -> 0
    # (like 1/N^2, from the O(1/N) boundary term).
    r = mean_ergodic_analysis(0.0, 0.0, "z", N=2 ** 12, M=2000, seed=3)
    assert r.D[-1] < 1e-5


def test_chaotic_decay():
    # standard kicked top at k=6: mostly chaotic, D_N should decay ~ 1/N
    r = mean_ergodic_analysis(6.0, 0.0, "z", N=2 ** 13, M=3000, seed=2)
    assert r.D[-1] < 0.05 * r.D[0]


def test_two_estimators_agree():
    r = mean_ergodic_analysis(3.0, 1.0, "z", N=2 ** 12, M=3000, seed=4)
    sel = r.Ns <= r.N // 8                      # estimators diverge only at lags ~ N
    rel = np.abs(r.D[sel] - r.D_corr[sel]) / (r.D_se[sel] + 1e-12)
    assert np.median(rel) < 3.0


# ---- added: checks against Purohit & Bhosale, PRE 112, 014217 -------------
def _paper_map(X, Y, Z, k, kp):
    """Eqs. (2)-(4) of the paper, transcribed independently."""
    A = Z * np.cos(k * X) + Y * np.sin(k * X)
    B = Y * np.cos(k * X) - Z * np.sin(k * X)
    return A, B * np.cos(kp * A) + X * np.sin(kp * A), -X * np.cos(kp * A) + B * np.sin(kp * A)


def test_map_matches_paper_eqs_2_to_4():
    rng = np.random.default_rng(0)
    x, y, z = sample_sphere(5000, rng)
    for k, kp in [(1.75, 0.25), (3, -1), (6, 0), (4.75, -2.75)]:
        a, b = dkt_step(x, y, z, k, kp), _paper_map(x, y, z, k, kp)
        assert max(np.abs(u - v).max() for u, v in zip(a, b)) < 1e-12


def test_eq5_roundtrip():
    from dkt_agent.mean_ergodicity import kr_ktheta_to_kkp, kkp_to_kr_ktheta
    assert kkp_to_kr_ktheta(*kr_ktheta_to_kkp(1.3, -0.4)) == (pytest.approx(1.3), pytest.approx(-0.4))
    assert kr_ktheta_to_kkp(1, 0.25) == (1.25, 0.75)


def test_poles_are_fixed_points():
    # paper Eq. (8): (0, +-1, 0) is fixed for any k, k'
    x, y, z = np.array([0.0]), np.array([1.0]), np.array([0.0])
    for k, kp in [(0.3, 5.1), (7.0, -2.0)]:
        a = dkt_step(x, y, z, k, kp)
        assert max(abs(float(u[0]) - float(v[0])) for u, v in zip(a, (x, y, z))) < 1e-14


def test_battery_catches_symmetry_blind_observable():
    from dkt_agent.mean_ergodicity import mean_ergodic_battery
    # k_r=0.75, k_theta=0 is regular (paper Fig. 1a): z looks 'ergodic' but y must not
    res, sm = mean_ergodic_battery(kr=0.75, ktheta=0, observables=("z", "y"),
                                   N=2 ** 11, M=1000, seed=1, n_checkpoints=25)
    assert sm["all_decay"] is False and sm["worst_observable"] == "y"


# ---- tool layer ---------------------------------------------------------------
def test_tool_layer_runs_and_saves(tmp_path):
    import json
    from dkt_agent.mean_ergodicity_tools import run_mean_ergodic_diagnostic, format_for_agent
    out = run_mean_ergodic_diagnostic(0.75, 0.0, ("z", "y"), n_kicks=2 ** 10,
                                      n_initial_conditions=500, seed=1,
                                      results_root=str(tmp_path))
    assert out["worst_observable"] == "y" and out["all_observables_decay"] is False
    json.dumps(out)                                   # must be JSON-serialisable
    files = out["artifacts"]["files"]
    assert "mean_ergodicity_summary.json" in files and "D_N_y.csv" in files
    assert "finite-data" in format_for_agent(out).lower()


def test_p_parameter_default_is_exact_pi_over_2():
    rng = np.random.default_rng(0)
    x, y, z = sample_sphere(100, rng)
    a = dkt_step(x, y, z, 2.0, 1.0)
    b = dkt_step(x, y, z, 2.0, 1.0, p=np.pi / 2)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    c = dkt_step(x, y, z, 2.0, 1.0, p=0.7)            # other p: still on the sphere
    assert np.max(np.abs(c[0] ** 2 + c[1] ** 2 + c[2] ** 2 - 1)) < 1e-12

def test_tool_layer_save_false_does_not_write_artifacts(tmp_path):
    from dkt_agent.mean_ergodicity_tools import run_mean_ergodic_diagnostic

    out = run_mean_ergodic_diagnostic(
        0.75, 0.0, ("z",), n_kicks=128,
        n_initial_conditions=100, seed=2, save=False,
        results_root=str(tmp_path),
    )
    assert out["artifacts"] == {}
    assert list(tmp_path.iterdir()) == []

