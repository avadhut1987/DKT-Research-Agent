import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

# ============================================================
# DKT parameters
# ============================================================

KR = 1.0
KTHETA = 10000.0
P_ANGLE = np.pi / 2

K = KR + KTHETA
KPRIME = KR - KTHETA

NZ = 40
NPHI = 80
SAMPLES_PER_CELL = 2048
SEED = 20261112

NCELLS = NZ * NPHI

OUT = Path("results/ergodicity_study/ulam_40x80_2048_seed2")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================
# Vectorized implementation of the actual classical DKT map
# ============================================================

def dkt_map(x, y, z):
    cp = np.cos(P_ANGLE)
    sp = np.sin(P_ANGLE)

    # Rotation about y
    x3 = x * cp + z * sp
    y3 = y
    z3 = z * cp - x * sp

    # First nonlinear twist: about z
    angle_z = K * z3
    cz = np.cos(angle_z)
    sz = np.sin(angle_z)

    x2 = x3 * cz - y3 * sz
    y2 = y3 * cz + x3 * sz
    z2 = z3

    # Second nonlinear twist: about x
    angle_x = KPRIME * x2
    cx = np.cos(angle_x)
    sx = np.sin(angle_x)

    x1 = x2
    y1 = y2 * cx - z2 * sx
    z1 = z2 * cx + y2 * sx

    return x1, y1, z1


# ============================================================
# Equal-area coordinates:
# z = cos(theta), phi in [-pi, pi)
# ============================================================

def destination_cell(x, y, z):
    phi = np.arctan2(y, x)
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    iz = np.floor((z + 1.0) * 0.5 * NZ).astype(int)
    iphi = np.floor((phi + np.pi) / (2 * np.pi) * NPHI).astype(int)

    iz = np.clip(iz, 0, NZ - 1)
    iphi = np.clip(iphi, 0, NPHI - 1)

    return iz * NPHI + iphi


# ============================================================
# Estimate Ulam transition matrix
# P[i,j] = probability of transition from cell i to cell j
# ============================================================

rows = []
cols = []

print("=" * 76)
print("ULAM TRANSITION MATRIX: DOUBLE KICKED TOP")
print("=" * 76)
print(f"kr                 = {KR}")
print(f"ktheta             = {KTHETA}")
print(f"k                  = {K}")
print(f"kprime             = {KPRIME}")
print(f"p                  = {P_ANGLE}")
print(f"Partition          = {NZ} x {NPHI}")
print(f"Number of cells    = {NCELLS}")
print(f"Samples per cell   = {SAMPLES_PER_CELL}")
print(f"Total map calls    = {NCELLS * SAMPLES_PER_CELL}")
print()

for iz in range(NZ):
    zlo = -1.0 + 2.0 * iz / NZ
    zhi = -1.0 + 2.0 * (iz + 1) / NZ

    for iphi in range(NPHI):
        source = iz * NPHI + iphi

        phil = -np.pi + 2.0 * np.pi * iphi / NPHI
        phih = -np.pi + 2.0 * np.pi * (iphi + 1) / NPHI

        # Uniform sampling in equal-area coordinates
        z0 = rng.uniform(zlo, zhi, SAMPLES_PER_CELL)
        phi0 = rng.uniform(phil, phih, SAMPLES_PER_CELL)

        rxy = np.sqrt(np.maximum(0.0, 1.0 - z0**2))
        x0 = rxy * np.cos(phi0)
        y0 = rxy * np.sin(phi0)

        x1, y1, z1 = dkt_map(x0, y0, z0)
        destinations = destination_cell(x1, y1, z1)

        rows.extend([source] * SAMPLES_PER_CELL)
        cols.extend(destinations.tolist())

    if (iz + 1) % 5 == 0:
        print(f"Completed z rows: {iz + 1}/{NZ}")

# Each source cell has the same number of samples.
counts = csr_matrix(
    (
        np.ones(len(rows), dtype=np.float64),
        (np.asarray(rows), np.asarray(cols)),
    ),
    shape=(NCELLS, NCELLS),
)

counts.sum_duplicates()

Pmat = counts.copy()
Pmat.data /= SAMPLES_PER_CELL

save_npz(OUT / "ulam_transition_matrix.npz", Pmat)
save_npz(OUT / "ulam_transition_counts.npz", counts)

print("\nTransition matrix constructed.")
print(f"Nonzero transitions: {Pmat.nnz}")
print(f"Matrix density:      {Pmat.nnz / (NCELLS**2):.6e}")


# ============================================================
# Basic stochasticity and area-preservation diagnostics
# ============================================================

row_sums = np.asarray(Pmat.sum(axis=1)).ravel()
col_sums = np.asarray(Pmat.sum(axis=0)).ravel()

print("\nSTOCHASTICITY CHECK")
print("-" * 76)
print(f"Maximum row-sum error:       {np.max(np.abs(row_sums - 1)):.6e}")
print(f"Mean column sum:             {np.mean(col_sums):.8f}")
print(f"Column-sum standard dev.:    {np.std(col_sums):.8f}")
print(f"Maximum |column sum - 1|:    {np.max(np.abs(col_sums - 1)):.8f}")

pd.DataFrame({
    "cell": np.arange(NCELLS),
    "row_sum": row_sums,
    "column_sum": col_sums,
}).to_csv(OUT / "ulam_cell_balance.csv", index=False)


# ============================================================
# Directed graph diagnostic
# ============================================================

n_scc, labels = connected_components(
    Pmat,
    directed=True,
    connection="strong",
    return_labels=True,
)

sizes = np.bincount(labels)
largest = np.sort(sizes)[::-1]

print("\nDIRECTED GRAPH DIAGNOSTIC")
print("-" * 76)
print(f"Number of strongly connected components: {n_scc}")
print(f"Largest component sizes: {largest[:15].tolist()}")

pd.DataFrame({
    "cell": np.arange(NCELLS),
    "strong_component": labels,
}).to_csv(OUT / "ulam_scc_labels.csv", index=False)


# ============================================================
# Eigenvalues closest to the unit circle
# ============================================================

print("\nCOMPUTING EIGENVALUES...")
try:
    vals = eigs(
        Pmat.T,
        k=20,
        which="LM",
        return_eigenvectors=False,
        tol=1e-8,
        maxiter=20000,
    )

    vals = vals[np.argsort(-np.abs(vals))]

    eig_df = pd.DataFrame({
        "real": vals.real,
        "imag": vals.imag,
        "absolute_value": np.abs(vals),
    })
    eig_df.to_csv(OUT / "ulam_eigenvalues.csv", index=False)

    print("\nEIGENVALUES CLOSEST TO THE UNIT CIRCLE")
    print("-" * 76)
    for i, val in enumerate(vals[:20]):
        print(
            f"{i + 1:2d}: "
            f"{val.real:+.8f} {val.imag:+.8f}i "
            f"|lambda|={abs(val):.8f}"
        )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(vals.real, vals.imag)
    circle = plt.Circle((0, 0), 1, fill=False, linestyle="--")
    ax.add_patch(circle)
    ax.axhline(0, linewidth=0.6)
    ax.axvline(0, linewidth=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title("Ulam transition matrix eigenvalues")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "ulam_eigenvalues.png", dpi=220)
    plt.close(fig)

except Exception as exc:
    print(f"Eigenvalue calculation failed: {exc}")
    vals = np.array([])


# ============================================================
# Save concise run summary
# ============================================================

with open(OUT / "summary.txt", "w") as f:
    f.write("ULAM TRANSITION MATRIX SUMMARY\n")
    f.write(f"kr={KR}, ktheta={KTHETA}, p={P_ANGLE}\n")
    f.write(f"grid={NZ}x{NPHI}, samples_per_cell={SAMPLES_PER_CELL}\n")
    f.write(f"nonzero_transitions={Pmat.nnz}\n")
    f.write(f"strongly_connected_components={n_scc}\n")
    f.write(f"largest_scc_sizes={largest[:15].tolist()}\n")
    f.write(f"column_sum_mean={np.mean(col_sums):.12f}\n")
    f.write(f"column_sum_std={np.std(col_sums):.12f}\n")
    f.write(f"column_sum_max_error={np.max(np.abs(col_sums - 1)):.12f}\n")
    if len(vals):
        f.write("eigenvalue_magnitudes=")
        f.write(str(np.abs(vals).tolist()))
        f.write("\n")

print("\n" + "=" * 76)
print("ULAM ANALYSIS COMPLETED")
print(f"Output directory: {OUT}")
print("=" * 76)
