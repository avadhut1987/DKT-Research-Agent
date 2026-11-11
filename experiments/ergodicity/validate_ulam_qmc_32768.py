import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import qmc

# ============================================================
# Parameters
# ============================================================

KR = 1.0
KTHETA = 10000.0

K = KR + KTHETA
KP = KR - KTHETA
P_ANGLE = np.pi / 2

NZ, NPHI = 40, 80
NCELLS = NZ * NPHI

M = 15
NSAMPLES = 2**M

SEEDS = [20261111, 20261112]

OUT = Path("results/ergodicity_study/ulam_qmc_32768")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DKT map
# ============================================================

def dkt_map(x, y, z):
    cp, sp = np.cos(P_ANGLE), np.sin(P_ANGLE)

    x3 = x * cp + z * sp
    y3 = y
    z3 = z * cp - x * sp

    az = K * z3
    cz, sz = np.cos(az), np.sin(az)

    x2 = x3 * cz - y3 * sz
    y2 = y3 * cz + x3 * sz
    z2 = z3

    ax = KP * x2
    cx, sx = np.cos(ax), np.sin(ax)

    x1 = x2
    y1 = y2 * cx - z2 * sx
    z1 = z2 * cx + y2 * sx

    return x1, y1, z1


# ============================================================
# Observables
# ============================================================

def evaluate_observables(x, y, z):
    return {
        "X": x,
        "Y": y,
        "Z": z,
        "X2": x*x,
        "Y2": y*y,
        "Z2": z*z,
        "XY": x*y,
        "YZ": y*z,
        "ZX": z*x,
    }


# ============================================================
# Direct conditional averages in each cell
# ============================================================

def calculate(seed):
    sampler = qmc.Sobol(
        d=2,
        scramble=True,
        seed=seed,
    )

    # One randomized low-discrepancy pattern, reused in each cell.
    # Each cell receives a translated/rescaled copy of this pattern.
    uv = sampler.random_base2(M)

    result = {
        name: np.empty(NCELLS)
        for name in evaluate_observables(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
    }

    for iz in range(NZ):
        zlo = -1.0 + 2.0 * iz / NZ
        dz = 2.0 / NZ

        for iphi in range(NPHI):
            cell = iz * NPHI + iphi

            philo = -np.pi + 2*np.pi*iphi/NPHI
            dphi = 2*np.pi/NPHI

            z0 = zlo + dz * uv[:, 0]
            phi0 = philo + dphi * uv[:, 1]

            r = np.sqrt(np.maximum(0.0, 1.0-z0*z0))

            x0 = r * np.cos(phi0)
            y0 = r * np.sin(phi0)

            x1, y1, z1 = dkt_map(x0, y0, z0)

            obs = evaluate_observables(x1, y1, z1)

            for name, values in obs.items():
                result[name][cell] = np.mean(values)

        if (iz+1) % 5 == 0:
            print(f"Seed {seed}: completed {iz+1}/{NZ} z rows")

    return result


# ============================================================
# Independent randomized QMC estimates
# ============================================================

print("="*76)
print("DIRECT ULAM CONDITIONAL-AVERAGE VALIDATION")
print("="*76)
print(f"Grid: {NZ} x {NPHI}")
print(f"Sobol samples per cell: {NSAMPLES}")
print(f"Total map evaluations per scramble: {NCELLS*NSAMPLES:,}")

results = {}

for seed in SEEDS:
    print(f"\nStarting scramble seed {seed}")
    results[seed] = calculate(seed)


# ============================================================
# Compare independent scrambles
# ============================================================

rows = []

for name in results[SEEDS[0]]:
    a = results[SEEDS[0]][name]
    b = results[SEEDS[1]][name]

    difference = a-b

    rms_difference = np.sqrt(np.mean(difference**2))
    rms_a = np.sqrt(np.mean(a**2))
    rms_b = np.sqrt(np.mean(b**2))

    # Scale-independent error using the larger signal.
    relative = rms_difference / max(rms_a, rms_b, 1e-14)

    correlation = np.corrcoef(a, b)[0, 1]

    rows.append({
        "observable": name,
        "rms_difference": rms_difference,
        "rms_seed1": rms_a,
        "rms_seed2": rms_b,
        "relative_difference": relative,
        "correlation": correlation,
        "max_abs_difference": np.max(np.abs(difference)),
    })

    print(
        f"{name:5s} "
        f"RMS(diff)={rms_difference:.6e} "
        f"relative={relative:.6e} "
        f"corr={correlation:.8f}"
    )

df = pd.DataFrame(rows)
df.to_csv(OUT / "qmc_comparison.csv", index=False)

# Save cell-wise conditional averages for later use.
for seed in SEEDS:
    data = {"cell": np.arange(NCELLS)}
    data.update(results[seed])

    pd.DataFrame(data).to_csv(
        OUT / f"conditional_averages_seed_{seed}.csv",
        index=False,
    )

print("\n" + "="*76)
print("QMC COMPARISON SUMMARY")
print("="*76)
print(f"Mean relative difference: {df.relative_difference.mean():.8e}")
print(f"Median relative difference: {df.relative_difference.median():.8e}")
print(f"Mean correlation: {df.correlation.mean():.8f}")
print(f"\nSaved results in: {OUT}")
