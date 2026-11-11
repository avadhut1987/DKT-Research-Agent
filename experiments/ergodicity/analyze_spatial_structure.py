import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("results/ergodicity_study/local_ic_grid")
INPUT = BASE / "local_grid_observable_averages.csv"
OUT = BASE / "spatial_structure"
OUT.mkdir(parents=True, exist_ok=True)

THETAS = np.array([0.990, 0.995, 1.000, 1.005, 1.010])
PHIS = np.array([-0.760, -0.755, -0.750, -0.745, -0.740])
TIMES = [2500, 5000, 10000, 20000, 50000, 100000]
MAIN_TIMES = [10000, 50000, 100000]
OBSERVABLES = ["X2", "Y2", "Z2"]

N_PERM = 9999
SEED = 20261111
rng = np.random.default_rng(SEED)

df = pd.read_csv(INPUT)

required = {"theta", "phi", "T", "observable", "average"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

expected = len(THETAS) * len(PHIS)
for T in TIMES:
    for obs in OBSERVABLES:
        sub = df[(df["T"] == T) & (df["observable"] == obs)]
        if len(sub) != expected:
            raise ValueError(
                f"{obs}, T={T}: expected {expected} values, found {len(sub)}"
            )

def make_grid(T, obs):
    sub = df[(df["T"] == T) & (df["observable"] == obs)]
    grid = sub.pivot(index="theta", columns="phi", values="average")
    grid = grid.reindex(index=THETAS, columns=PHIS)
    if grid.isna().any().any():
        raise ValueError(f"Missing grid entries for {obs}, T={T}")
    return grid.to_numpy(dtype=float)

# Every horizontal and vertical nearest-neighbour edge.
EDGES = []
for i in range(5):
    for j in range(5):
        if i < 4:
            EDGES.append((i, j, i + 1, j, "theta"))
        if j < 4:
            EDGES.append((i, j, i, j + 1, "phi"))

def spatial_statistics(grid):
    values = grid.ravel()
    mean = values.mean()
    variance = np.mean((values - mean) ** 2)
    std = np.sqrt(variance)

    dtheta = np.diff(grid, axis=0)
    dphi = np.diff(grid, axis=1)
    edge_diffs = np.array([
        grid[i2, j2] - grid[i1, j1]
        for i1, j1, i2, j2, _ in EDGES
    ])

    # Moran's I for the 5x5 four-neighbour grid.
    # Each undirected edge is counted once.
    products = np.array([
        (grid[i1, j1] - mean) * (grid[i2, j2] - mean)
        for i1, j1, i2, j2, _ in EDGES
    ])
    E = len(EDGES)
    moran = (25.0 / E) * products.sum() / np.sum((values - mean) ** 2)

    return {
        "mean": mean,
        "std": std,
        "minimum": values.min(),
        "maximum": values.max(),
        "range": values.max() - values.min(),
        "theta_neighbor_RMS": np.sqrt(np.mean(dtheta**2)),
        "phi_neighbor_RMS": np.sqrt(np.mean(dphi**2)),
        "all_neighbor_RMS": np.sqrt(np.mean(edge_diffs**2)),
        "Moran_I": moran,
    }

def moran_from_values(values):
    grid = values.reshape(5, 5)
    mean = values.mean()
    denom = np.sum((values - mean) ** 2)
    products = sum(
        (grid[i1, j1] - mean) * (grid[i2, j2] - mean)
        for i1, j1, i2, j2, _ in EDGES
    )
    return (25.0 / len(EDGES)) * products / denom

# ------------------------------------------------------------
# A. Spatial statistics and permutation test
# ------------------------------------------------------------

records = []

for T in TIMES:
    for obs in OBSERVABLES:
        grid = make_grid(T, obs)
        stats = spatial_statistics(grid)
        values = grid.ravel()

        null = np.empty(N_PERM)
        for b in range(N_PERM):
            null[b] = moran_from_values(rng.permutation(values))

        # Two-sided permutation test: either clustering or alternation.
        p_two = (
            1 + np.count_nonzero(np.abs(null) >= abs(stats["Moran_I"]))
        ) / (N_PERM + 1)

        p_positive = (
            1 + np.count_nonzero(null >= stats["Moran_I"])
        ) / (N_PERM + 1)

        records.append({
            "T": T,
            "observable": obs,
            **stats,
            "null_Moran_mean": null.mean(),
            "null_Moran_std": null.std(ddof=1),
            "permutation_p_two_sided": p_two,
            "permutation_p_positive": p_positive,
        })

        if T in MAIN_TIMES:
            print(
                f"T={T:6d} {obs:>2s} | "
                f"std={stats['std']:.6f} | "
                f"theta_RMS={stats['theta_neighbor_RMS']:.6f} | "
                f"phi_RMS={stats['phi_neighbor_RMS']:.6f} | "
                f"Moran I={stats['Moran_I']:+.5f} | "
                f"permutation p={p_two:.5f}"
            )

stats_df = pd.DataFrame(records)
stats_df.to_csv(OUT / "spatial_statistics.csv", index=False)

# ------------------------------------------------------------
# B. Heatmaps: 3 observables x 3 observation times
# ------------------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)

for r, obs in enumerate(OBSERVABLES):
    for c, T in enumerate(MAIN_TIMES):
        ax = axes[r, c]
        grid = make_grid(T, obs)

        im = ax.pcolormesh(
            PHIS, THETAS, grid,
            shading="nearest",
            cmap="viridis"
        )

        ax.set_title(f"{obs}, T={T:,}")
        ax.set_xlabel(r"$\phi_0$")
        ax.set_ylabel(r"$\theta_0$")
        ax.set_xticks(PHIS)
        ax.set_yticks(THETAS)
        ax.tick_params(axis="x", rotation=35)

        fig.colorbar(im, ax=ax, label=f"Mean {obs}")

fig.suptitle(
    r"Spatial structure of Birkhoff averages: "
    r"$k_r=1,\ k_\theta=10^4$",
    fontsize=15
)
fig.savefig(OUT / "birkhoff_average_heatmaps.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------
# C. Neighbour-difference maps at T=100000
# ------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)

for r, obs in enumerate(OBSERVABLES):
    grid = make_grid(100000, obs)

    dtheta = np.abs(np.diff(grid, axis=0))
    dphi = np.abs(np.diff(grid, axis=1))

    im1 = axes[r, 0].imshow(
        dtheta, origin="lower", aspect="auto", cmap="magma"
    )
    axes[r, 0].set_title(f"{obs}: |difference along theta|")
    axes[r, 0].set_xlabel("phi grid interval")
    axes[r, 0].set_ylabel("theta grid interval")
    fig.colorbar(im1, ax=axes[r, 0])

    im2 = axes[r, 1].imshow(
        dphi, origin="lower", aspect="auto", cmap="magma"
    )
    axes[r, 1].set_title(f"{obs}: |difference along phi|")
    axes[r, 1].set_xlabel("phi grid interval")
    axes[r, 1].set_ylabel("theta grid point")
    fig.colorbar(im2, ax=axes[r, 1])

fig.suptitle("Nearest-neighbour Birkhoff-average differences at T=100000")
fig.savefig(OUT / "nearest_neighbor_difference_maps.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------
# D. Evolution of spatial variation with observation time
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

for ax, obs in zip(axes, OBSERVABLES):
    sub = stats_df[stats_df["observable"] == obs]
    ax.plot(sub["T"], sub["std"], "o-", label="Spatial standard deviation")
    ax.set_xscale("log")
    ax.set_xlabel("Observation time T")
    ax.set_ylabel("Spatial standard deviation")
    ax.set_title(obs)
    ax.grid(True, alpha=0.3)

fig.suptitle("Persistence of initial-condition dependence")
fig.savefig(OUT / "spatial_variation_vs_time.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------
# E. Report
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("SPATIAL STRUCTURE SUMMARY")
print("=" * 90)

columns = [
    "T", "observable", "std",
    "theta_neighbor_RMS", "phi_neighbor_RMS",
    "Moran_I", "null_Moran_mean",
    "permutation_p_two_sided"
]
print(stats_df[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nSaved files:")
for name in [
    "spatial_statistics.csv",
    "birkhoff_average_heatmaps.png",
    "nearest_neighbor_difference_maps.png",
    "spatial_variation_vs_time.png",
]:
    print(OUT / name)

print("\nSpatial-structure analysis completed.")
