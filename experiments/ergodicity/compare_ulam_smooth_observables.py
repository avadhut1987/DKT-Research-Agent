import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz

BASE = Path("results/ergodicity_study")

P1 = load_npz(
    BASE / "ulam_40x80_2048/ulam_transition_matrix.npz"
).tocsr()

P2 = load_npz(
    BASE / "ulam_40x80_2048_seed2/ulam_transition_matrix.npz"
).tocsr()

NZ, NPHI = 40, 80
NCELLS = NZ * NPHI

# Equal-area cell centres
z = -1 + 2 * (np.arange(NZ) + 0.5) / NZ
phi = -np.pi + 2 * np.pi * (np.arange(NPHI) + 0.5) / NPHI

Z, PHI = np.meshgrid(z, phi, indexing="ij")
R = np.sqrt(1 - Z**2)
X = R * np.cos(PHI)
Y = R * np.sin(PHI)

X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()

# Smooth polynomial observables, including higher-order probes
observables = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "X2": X**2,
    "Y2": Y**2,
    "Z2": Z**2,
    "XY": X*Y,
    "YZ": Y*Z,
    "ZX": Z*X,
    "X3": X**3,
    "Y3": Y**3,
    "Z3": Z**3,
    "XYZ": X*Y*Z,
    "X2Y": X**2*Y,
    "X2Z": X**2*Z,
    "Y2Z": Y**2*Z,
}

rows = []

print("=" * 78)
print("INDEPENDENT ULAM REALIZATIONS: SMOOTH OBSERVABLE COMPARISON")
print("=" * 78)

for name, f in observables.items():
    # Remove the uniform-sphere mean
    f = f - np.mean(f)

    # One-step conditional expectation in each source cell
    g1 = P1 @ f
    g2 = P2 @ f

    difference = g1 - g2

    rms_difference = np.sqrt(np.mean(difference**2))
    rms_signal = np.sqrt(np.mean(g1**2))
    relative_error = rms_difference / max(rms_signal, 1e-15)

    correlation = np.corrcoef(g1, g2)[0, 1]

    rows.append({
        "observable": name,
        "rms_difference": rms_difference,
        "rms_signal_seed1": rms_signal,
        "relative_error": relative_error,
        "correlation": correlation,
        "max_abs_difference": np.max(np.abs(difference)),
    })

    print(
        f"{name:5s} "
        f"RMS(diff)={rms_difference:.6e} "
        f"relative={relative_error:.6e} "
        f"corr={correlation:.8f} "
        f"max={np.max(np.abs(difference)):.6e}"
    )

df = pd.DataFrame(rows)
out = BASE / "ulam_smooth_observable_comparison"
out.mkdir(exist_ok=True)

df.to_csv(out / "smooth_observable_comparison.csv", index=False)

print("\nSUMMARY")
print("-" * 78)
print(f"Mean relative error:   {df.relative_error.mean():.8e}")
print(f"Median relative error: {df.relative_error.median():.8e}")
print(f"Maximum relative error:{df.relative_error.max():.8e}")
print(f"Mean correlation:     {df.correlation.mean():.8f}")

print(f"\nSaved: {out / 'smooth_observable_comparison.csv'}")
