import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("results/ergodicity_study")
OUT = BASE / "block_spatial_analysis"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "512-bit": BASE / "kt10000_ic01_100k.csv",
    "1024-bit": BASE / "multiple_ic/ic01_1024.csv",
}

NZ, NPHI = 40, 80
BLOCK_SIZE = 10_000
NBLOCKS = 10

def load_blocks(path):
    data = np.genfromtxt(path, delimiter=",", names=True)

    xyz = np.column_stack([data["X"], data["Y"], data["Z"]])

    # Exclude the initial condition (step 0).
    xyz = xyz[1:1 + NBLOCKS * BLOCK_SIZE]

    if len(xyz) != NBLOCKS * BLOCK_SIZE:
        raise ValueError(f"Unexpected trajectory length in {path}")

    histograms = []

    for b in range(NBLOCKS):
        points = xyz[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]

        z = np.clip(points[:, 2], -1.0, 1.0)
        phi = np.arctan2(points[:, 1], points[:, 0])

        hist, _, _ = np.histogram2d(
            z, phi,
            bins=(NZ, NPHI),
            range=((-1, 1), (-np.pi, np.pi))
        )

        histograms.append(hist)

    return np.asarray(histograms)

all_hist = {}

for name, path in FILES.items():
    print(f"\nLoading {name}: {path}")
    all_hist[name] = load_blocks(path)
    print("Loaded:", all_hist[name].shape)

# Use one common color scale for both precisions.
vmax = max(h.max() / BLOCK_SIZE for h in all_hist.values()
           for h in h)

for name, histograms in all_hist.items():
    fig, axes = plt.subplots(
        2, 5, figsize=(18, 7), constrained_layout=True
    )

    for i, ax in enumerate(axes.flat):
        prob = histograms[i] / BLOCK_SIZE

        im = ax.imshow(
            prob.T,
            origin="lower",
            extent=(-np.pi, np.pi, -1, 1),
            aspect="auto",
            cmap="inferno",
            vmin=0,
            vmax=vmax
        )

        ax.set_title(f"Block {i+1:02d}")
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$z=\cos\theta$")

    fig.colorbar(
        im, ax=axes.ravel().tolist(),
        label="Empirical probability per equal-area cell",
        shrink=0.85
    )

    fig.suptitle(
        f"{name}: all ten temporal blocks "
        r"($k_r=1,\ k_\theta=10^4$)"
    )

    output = OUT / f"{name.replace('-', '')}_all_10_blocks.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print("Saved:", output)

# Save raw counts for reproducibility.
np.savez_compressed(
    OUT / "all_10_block_histograms.npz",
    **{name.replace("-", ""): hist for name, hist in all_hist.items()}
)

print("\nAll ten block maps generated.")
