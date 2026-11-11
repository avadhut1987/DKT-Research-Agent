import numpy as np
from pathlib import Path
from itertools import combinations

OUT = Path("results/ergodicity_study/block_spatial_analysis")
data = np.load(OUT / "all_10_block_histograms.npz")

H512 = data["512bit"]
H1024 = data["1024bit"]

THRESHOLDS = [1, 3, 5, 10]

def jaccard(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return np.nan
    return np.logical_and(a, b).sum() / union

def analyse(name, histograms):
    print("\n" + "="*75)
    print(name)
    print("="*75)

    for q in THRESHOLDS:
        masks = histograms >= q
        occupied = masks.reshape(10, -1).sum(axis=1)

        overlaps = [
            jaccard(masks[i], masks[j])
            for i, j in combinations(range(10), 2)
        ]

        adjacent = [
            jaccard(masks[i], masks[i+1])
            for i in range(9)
        ]

        print(f"\nMinimum visits per cell q = {q}")
        print("Occupied cells per block:", occupied)
        print(
            f"All-pair Jaccard: mean={np.nanmean(overlaps):.5f}, "
            f"median={np.nanmedian(overlaps):.5f}"
        )
        print(
            f"Adjacent-block Jaccard: mean={np.nanmean(adjacent):.5f}, "
            f"median={np.nanmedian(adjacent):.5f}"
        )

analyse("512-bit", H512)
analyse("1024-bit", H1024)

print("\n" + "="*75)
print("CROSS-PRECISION: CORRESPONDING TEMPORAL BLOCKS")
print("="*75)

for q in THRESHOLDS:
    values = [
        jaccard(H512[i] >= q, H1024[i] >= q)
        for i in range(10)
    ]
    print(f"\nq={q}")
    print("Blockwise Jaccard:", np.round(values, 5))
    print(
        f"Mean={np.nanmean(values):.5f}, "
        f"median={np.nanmedian(values):.5f}"
    )

print("\nSupport-overlap analysis completed.")
