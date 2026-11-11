import numpy as np
from pathlib import Path

BASE = Path("results/ergodicity_study")
FILES = {
    "512-bit": BASE / "kt10000_ic01_100k.csv",
    "1024-bit": BASE / "multiple_ic/ic01_1024.csv",
}

BLOCK_SIZE = 10_000
NBLOCKS = 10
N_PERM = 500
SEED = 20261111
RESOLUTIONS = [(40, 80), (80, 160)]

def load_xyz(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.column_stack((d["X"], d["Y"], d["Z"]))[
        1:1 + BLOCK_SIZE * NBLOCKS
    ]

def cell_indices(xyz, nz, nphi):
    z = np.clip(xyz[:, 2], -1, 1)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    iz = np.minimum(
        ((z + 1) * nz / 2).astype(int), nz - 1
    )
    iphi = np.minimum(
        ((phi + np.pi) * nphi / (2*np.pi)).astype(int),
        nphi - 1
    )

    return iz * nphi + iphi

def block_histograms(indices, ncells):
    return np.array([
        np.bincount(
            indices[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE],
            minlength=ncells
        )
        for i in range(NBLOCKS)
    ])

def adjacent_metrics(H):
    masks = H > 0
    probs = H / H.sum(axis=1, keepdims=True)

    jaccard = []
    tv = []

    for i in range(NBLOCKS - 1):
        a, b = masks[i], masks[i+1]
        union = np.logical_or(a, b).sum()
        intersection = np.logical_and(a, b).sum()

        jaccard.append(intersection / union if union else np.nan)
        tv.append(0.5 * np.abs(probs[i] - probs[i+1]).sum())

    return np.mean(jaccard), np.mean(tv)

rng = np.random.default_rng(SEED)

for nz, nphi in RESOLUTIONS:
    ncells = nz * nphi

    print("\n" + "="*85)
    print(f"RESOLUTION: {nz} x {nphi}")
    print("="*85)

    for name, path in FILES.items():
        xyz = load_xyz(path)
        indices = cell_indices(xyz, nz, nphi)

        H = block_histograms(indices, ncells)
        observed_j, observed_tv = adjacent_metrics(H)

        null_j = []
        null_tv = []

        for _ in range(N_PERM):
            shuffled = rng.permutation(indices)
            Hp = block_histograms(shuffled, ncells)
            j, tv = adjacent_metrics(Hp)
            null_j.append(j)
            null_tv.append(tv)

        null_j = np.asarray(null_j)
        null_tv = np.asarray(null_tv)

        print(f"\n{name}")
        print(
            f"Chronological adjacent Jaccard: {observed_j:.6f}"
        )
        print(
            f"Shuffled Jaccard: "
            f"{null_j.mean():.6f} +/- {null_j.std():.6f}"
        )
        print(
            f"Chronological adjacent TV: {observed_tv:.6f}"
        )
        print(
            f"Shuffled TV: "
            f"{null_tv.mean():.6f} +/- {null_tv.std():.6f}"
        )

        print(
            "Jaccard shuffled 2.5%-97.5% interval:",
            np.quantile(null_j, [0.025, 0.975])
        )
        print(
            "TV shuffled 2.5%-97.5% interval:",
            np.quantile(null_tv, [0.025, 0.975])
        )

print("\nTemporal-order control completed.")
