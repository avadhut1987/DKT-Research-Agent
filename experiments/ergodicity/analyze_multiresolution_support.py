import numpy as np
from pathlib import Path

BASE = Path("results/ergodicity_study")
FILES = {
    "512-bit": BASE / "kt10000_ic01_100k.csv",
    "1024-bit": BASE / "multiple_ic/ic01_1024.csv",
}

RESOLUTIONS = [(20, 40), (40, 80), (80, 160)]
BLOCK_SIZE = 10_000
NBLOCKS = 10

def load_xyz(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    xyz = np.column_stack((d["X"], d["Y"], d["Z"]))
    return xyz[1:1 + BLOCK_SIZE * NBLOCKS]

def block_histograms(xyz, nz, nphi):
    result = []

    for b in range(NBLOCKS):
        pts = xyz[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]

        z = np.clip(pts[:, 2], -1, 1)
        phi = np.arctan2(pts[:, 1], pts[:, 0])

        h, _, _ = np.histogram2d(
            z, phi,
            bins=(nz, nphi),
            range=((-1, 1), (-np.pi, np.pi))
        )
        result.append(h)

    return np.asarray(result)

def jaccard(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else np.nan

trajectories = {
    name: load_xyz(path) for name, path in FILES.items()
}

for nz, nphi in RESOLUTIONS:
    print("\n" + "=" * 85)
    print(f"RESOLUTION: {nz} x {nphi}")
    print("=" * 85)

    H = {
        name: block_histograms(xyz, nz, nphi)
        for name, xyz in trajectories.items()
    }

    cumulative = {
        name: np.zeros((nz, nphi), dtype=float)
        for name in FILES
    }

    masks = {
        name: np.zeros((nz, nphi), dtype=bool)
        for name in FILES
    }

    print(
        "Block | 512 new | 1024 new | "
        "512 support | 1024 support | cross-precision Jaccard"
    )

    for i in range(NBLOCKS):
        new_counts = {}

        for name in FILES:
            current = H[name][i] > 0
            new = current & (~masks[name])
            new_counts[name] = int(new.sum())

            masks[name] |= current
            cumulative[name] += H[name][i]

        overlap = jaccard(
            masks["512-bit"], masks["1024-bit"]
        )

        print(
            f"{i+1:5d} | "
            f"{new_counts['512-bit']:8d} | "
            f"{new_counts['1024-bit']:9d} | "
            f"{masks['512-bit'].sum():11d} | "
            f"{masks['1024-bit'].sum():12d} | "
            f"{overlap:.6f}"
        )

    print(
        "Final cumulative support fractions:",
        f"512={masks['512-bit'].mean():.6f},",
        f"1024={masks['1024-bit'].mean():.6f}"
    )

print("\nMulti-resolution support analysis completed.")
