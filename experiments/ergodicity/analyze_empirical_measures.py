from pathlib import Path
from itertools import combinations
import csv
import numpy as np
from scipy.spatial.distance import jensenshannon

from dkt_agent.classical_analysis import load_trajectory, equal_area_measure


BASE = Path("results/ergodicity_study")
OUTPUT = BASE / "empirical_measure_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "IC01": BASE / "kt10000_ic01_100k.csv",
    "IC02": BASE / "multiple_ic/ic02.csv",
    "IC03": BASE / "multiple_ic/ic03.csv",
    "IC04": BASE / "multiple_ic/ic04.csv",
    "IC05": BASE / "multiple_ic/ic05.csv",
}

# All lengths refer to evolved points, excluding the initial point.
PREFIX_LENGTHS = [10_000, 30_000, 100_000]
BLOCK_LENGTH = 10_000

RESOLUTIONS = [
    (20, 40),
    (40, 80),
    (80, 160),
]


def js_divergence(p, q):
    """Jensen-Shannon divergence, logarithm base 2."""
    return float(jensenshannon(p, q, base=2.0) ** 2)


def save_matrix(path, labels, matrix):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["IC"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{x:.12g}" for x in row])


def pairwise_matrix(measures, labels):
    matrix = np.zeros((len(labels), len(labels)))
    for i, j in combinations(range(len(labels)), 2):
        matrix[i, j] = matrix[j, i] = js_divergence(
            measures[labels[i]], measures[labels[j]]
        )
    return matrix


print("=" * 85)
print("EMPIRICAL-MEASURE COMPARISON")
print("=" * 85)

trajectories = {}
for label, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)
    points = load_trajectory(path)
    if len(points) < 100_001:
        raise ValueError(f"{label}: expected at least 100001 points")
    trajectories[label] = points
    print(f"{label}: loaded {len(points)} points from {path}")

labels = list(trajectories)

# ---------------------------------------------------------------------
# A. Cumulative empirical measures: compare different initial conditions
# ---------------------------------------------------------------------

cross_rows = []

for n_z, n_phi in RESOLUTIONS:
    print(f"\nPartition: {n_z} x {n_phi} ({n_z*n_phi} cells)")

    for T in PREFIX_LENGTHS:
        measures = {
            label: equal_area_measure(
                trajectories[label][1:T+1],
                n_z=n_z,
                n_phi=n_phi,
            )
            for label in labels
        }

        matrix = pairwise_matrix(measures, labels)

        filename = OUTPUT / f"cross_ic_{T}_{n_z}x{n_phi}.csv"
        save_matrix(filename, labels, matrix)

        upper = matrix[np.triu_indices(len(labels), k=1)]

        cross_rows.append({
            "n_z": n_z,
            "n_phi": n_phi,
            "points": T,
            "mean_pairwise_JS": float(np.mean(upper)),
            "median_pairwise_JS": float(np.median(upper)),
            "max_pairwise_JS": float(np.max(upper)),
            "min_pairwise_JS": float(np.min(upper)),
        })

        print(
            f"  T={T:6d}: "
            f"mean JS={np.mean(upper):.6g}, "
            f"median JS={np.median(upper):.6g}, "
            f"max JS={np.max(upper):.6g}"
        )

with open(OUTPUT / "cross_ic_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(cross_rows[0]))
    writer.writeheader()
    writer.writerows(cross_rows)


# ---------------------------------------------------------------------
# B. Temporal-block measures: compare consecutive portions of each orbit
# ---------------------------------------------------------------------

temporal_rows = []

for n_z, n_phi in RESOLUTIONS:
    print(f"\nTemporal blocks: partition {n_z} x {n_phi}")

    for label, points in trajectories.items():
        block_measures = {}

        for block in range(10):
            start = 1 + block * BLOCK_LENGTH
            stop = start + BLOCK_LENGTH

            block_points = points[start:stop]

            if len(block_points) != BLOCK_LENGTH:
                raise ValueError(
                    f"{label}, block {block}: got {len(block_points)} points"
                )

            block_measures[f"B{block+1:02d}"] = equal_area_measure(
                block_points,
                n_z=n_z,
                n_phi=n_phi,
            )

        block_labels = list(block_measures)
        matrix = pairwise_matrix(block_measures, block_labels)

        filename = OUTPUT / f"temporal_{label}_{n_z}x{n_phi}.csv"
        save_matrix(filename, block_labels, matrix)

        upper = matrix[np.triu_indices(10, k=1)]

        # Compare the first five blocks with the last five blocks.
        early_late = []
        for i in range(5):
            for j in range(5, 10):
                early_late.append(matrix[i, j])

        # Compare adjacent blocks only.
        adjacent = [matrix[i, i+1] for i in range(9)]

        row = {
            "IC": label,
            "n_z": n_z,
            "n_phi": n_phi,
            "mean_all_block_pairs_JS": float(np.mean(upper)),
            "median_all_block_pairs_JS": float(np.median(upper)),
            "max_all_block_pairs_JS": float(np.max(upper)),
            "mean_early_late_JS": float(np.mean(early_late)),
            "mean_adjacent_block_JS": float(np.mean(adjacent)),
        }
        temporal_rows.append(row)

        print(
            f"  {label}: "
            f"mean all-pairs={row['mean_all_block_pairs_JS']:.6g}, "
            f"early-late={row['mean_early_late_JS']:.6g}, "
            f"adjacent={row['mean_adjacent_block_JS']:.6g}"
        )

with open(OUTPUT / "temporal_block_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(temporal_rows[0]))
    writer.writeheader()
    writer.writerows(temporal_rows)


# ---------------------------------------------------------------------
# C. Print full-resolution cross-IC matrices for inspection
# ---------------------------------------------------------------------

print("\n" + "=" * 85)
print("CROSS-INITIAL-CONDITION JS DIVERGENCES")
print("=" * 85)

for T in PREFIX_LENGTHS:
    path = OUTPUT / f"cross_ic_{T}_40x80.csv"
    print(f"\nT = {T}, partition = 40 x 80")
    with open(path) as f:
        for row in csv.reader(f):
            print("  ".join(row))


print("\n" + "=" * 85)
print("ANALYSIS COMPLETE")
print(f"Results saved in: {OUTPUT}")
print("=" * 85)
