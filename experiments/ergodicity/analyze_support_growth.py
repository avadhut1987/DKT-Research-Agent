import numpy as np
from pathlib import Path

OUT = Path("results/ergodicity_study/block_spatial_analysis")
data = np.load(OUT / "all_10_block_histograms.npz")

NZ, NPHI = 40, 80
BLOCK_SIZE = 10_000

def analyse(name, histograms):
    print("\n" + "="*80)
    print(name)
    print("="*80)

    cumulative = np.zeros((NZ, NPHI), dtype=histograms.dtype)
    previous_mask = np.zeros((NZ, NPHI), dtype=bool)

    for i, hist in enumerate(histograms):
        current_mask = hist > 0

        new_cells = current_mask & (~previous_mask)
        new_count = new_cells.sum()

        total_visited = (previous_mask | current_mask).sum()

        new_fraction = (
            new_count / total_visited if total_visited else np.nan
        )

        cumulative_before = cumulative.copy()
        previous_count = cumulative_before.sum()

        block_prob = hist.ravel() / BLOCK_SIZE

        if previous_count > 0:
            previous_prob = (
                cumulative_before.ravel() / previous_count
            )
            tv = 0.5 * np.abs(
                block_prob - previous_prob
            ).sum()
        else:
            tv = np.nan

        cumulative += hist
        previous_mask |= current_mask

        print(
            f"Block {i+1:02d}: "
            f"new_cells={new_count:4d}, "
            f"cumulative_support={total_visited:4d}, "
            f"new_fraction={new_fraction:.6f}, "
            f"TV_vs_previous_cumulative={tv:.6f}"
        )

    print("Final cumulative support:", previous_mask.sum())

analyse("512-bit", data["512bit"])
analyse("1024-bit", data["1024bit"])

print("\nSupport-growth analysis completed.")
