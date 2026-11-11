import csv
import numpy as np
from pathlib import Path

BASE = Path("results/ergodicity_study")
OUT = BASE / "observable_convergence"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "IC01_512": BASE / "kt10000_ic01_100k.csv",
    "IC01_1024": BASE / "multiple_ic/ic01_1024.csv",

    "IC02_512": BASE / "multiple_ic/ic02.csv",
    "IC03_512": BASE / "multiple_ic/ic03.csv",
    "IC04_512": BASE / "multiple_ic/ic04.csv",
    "IC05_512": BASE / "multiple_ic/ic05.csv",

    "IC02_1024": BASE / "multiple_ic_1024/ic02.csv",
    "IC03_1024": BASE / "multiple_ic_1024/ic03.csv",
    "IC04_1024": BASE / "multiple_ic_1024/ic04.csv",
    "IC05_1024": BASE / "multiple_ic_1024/ic05.csv",
}

OBSERVABLES = {
    "X": lambda x, y, z: x,
    "Y": lambda x, y, z: y,
    "Z": lambda x, y, z: z,
    "X2": lambda x, y, z: x*x,
    "Y2": lambda x, y, z: y*y,
    "Z2": lambda x, y, z: z*z,
    "XY": lambda x, y, z: x*y,
    "YZ": lambda x, y, z: y*z,
    "ZX": lambda x, y, z: z*x,
}

CUMULATIVE_TIMES = [2500, 5000, 10000, 20000, 50000, 100000]
BLOCK_LENGTHS = [2500, 5000, 10000, 20000]


def load_observables(path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.genfromtxt(path, delimiter=",", names=True)

    # Exclude step=0 so that all trajectories use kicks 1...100000.
    xyz = np.column_stack((data["X"], data["Y"], data["Z"]))[1:100001]

    if len(xyz) != 100000:
        raise ValueError(
            f"{path}: expected 100000 post-initial points, got {len(xyz)}"
        )

    x, y, z = xyz.T

    values = {
        name: np.asarray(func(x, y, z), dtype=float)
        for name, func in OBSERVABLES.items()
    }

    return values


cumulative_rows = []
block_rows = []

for trajectory, path in FILES.items():
    print("\n" + "=" * 85)
    print(f"TRAJECTORY: {trajectory}")
    print("=" * 85)

    values = load_observables(path)

    # Cumulative time averages
    print("\nCUMULATIVE TIME AVERAGES")
    print("Observable | " + " | ".join(f"T={t}" for t in CUMULATIVE_TIMES))

    for name, a in values.items():
        cumulative_sum = np.cumsum(a)

        row_values = []
        for T in CUMULATIVE_TIMES:
            avg = cumulative_sum[T-1] / T
            row_values.append(avg)

            cumulative_rows.append({
                "trajectory": trajectory,
                "observable": name,
                "T": T,
                "average": avg,
            })

        print(
            f"{name:10s} | " +
            " | ".join(f"{v:+.8f}" for v in row_values)
        )

    # Non-overlapping block averages
    print("\nBLOCK-AVERAGE STATISTICS")
    print("Observable | L | blocks | mean(block avg) | std(block avg)")

    for L in BLOCK_LENGTHS:
        nblocks = len(next(iter(values.values()))) // L

        for name, a in values.items():
            trimmed = a[:nblocks * L]
            block_means = trimmed.reshape(nblocks, L).mean(axis=1)

            mean_blocks = block_means.mean()
            std_blocks = block_means.std(ddof=1) if nblocks > 1 else 0.0

            block_rows.append({
                "trajectory": trajectory,
                "observable": name,
                "block_length": L,
                "number_of_blocks": nblocks,
                "mean_block_average": mean_blocks,
                "std_block_average": std_blocks,
                "min_block_average": block_means.min(),
                "max_block_average": block_means.max(),
            })

            print(
                f"{name:10s} | {L:5d} | {nblocks:6d} | "
                f"{mean_blocks:+.8f} | {std_blocks:.8f}"
            )

# Save machine-readable results
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT / "cumulative_averages.csv", cumulative_rows)
write_csv(OUT / "block_average_statistics.csv", block_rows)

print("\n" + "=" * 85)
print("OBSERVABLE CONVERGENCE ANALYSIS COMPLETED")
print(f"Results saved in: {OUT}")
print("=" * 85)
