import numpy as np
from pathlib import Path

BASE = Path("results/ergodicity_study")

FILES = {
    "512-bit": BASE / "kt10000_ic01_100k.csv",
    "1024-bit": BASE / "multiple_ic/ic01_1024.csv",
}

BLOCK_LENGTHS = [2500, 5000, 10000, 20000]
RESOLUTIONS = [(40, 80), (80, 160)]
N_PERM = 500
SEED = 20261113


def load_xyz(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    xyz = np.column_stack((d["X"], d["Y"], d["Z"]))
    return xyz[1:100001]


def cell_indices(xyz, nz, nphi):
    z = np.clip(xyz[:, 2], -1.0, 1.0)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    iz = np.minimum(
        ((z + 1.0) * nz / 2.0).astype(int),
        nz - 1
    )

    iphi = np.minimum(
        ((phi + np.pi) * nphi / (2.0 * np.pi)).astype(int),
        nphi - 1
    )

    return iz * nphi + iphi


def block_histograms(indices, block_length, nblocks, ncells):
    return np.array([
        np.bincount(
            indices[i * block_length:(i + 1) * block_length],
            minlength=ncells
        )
        for i in range(nblocks)
    ])


def adjacent_metrics(H):
    masks = H > 0
    probs = H / H.sum(axis=1, keepdims=True)

    jaccard = []
    tv = []

    for i in range(len(H) - 1):
        a = masks[i]
        b = masks[i + 1]

        union = np.logical_or(a, b).sum()
        intersection = np.logical_and(a, b).sum()

        jaccard.append(
            intersection / union if union else np.nan
        )

        tv.append(
            0.5 * np.abs(probs[i] - probs[i + 1]).sum()
        )

    return np.nanmean(jaccard), np.nanmean(tv)


rng = np.random.default_rng(SEED)

for nz, nphi in RESOLUTIONS:
    ncells = nz * nphi

    print("\n" + "=" * 105)
    print(f"RESOLUTION: {nz} x {nphi}")
    print("=" * 105)

    for name, path in FILES.items():
        xyz = load_xyz(path)
        indices = cell_indices(xyz, nz, nphi)

        print(f"\nPRECISION: {name}")

        print(
            "L | offset | blocks | "
            "chronological J | shuffled J mean [95% interval] | "
            "chronological TV | shuffled TV mean [95% interval]"
        )

        for L in BLOCK_LENGTHS:
            total_blocks = len(indices) // L

            # Keep the same number of complete blocks
            # for all four partition origins.
            nblocks = total_blocks - 1

            offsets = [0, L // 4, L // 2, 3 * L // 4]

            for offset in offsets:
                start = offset
                stop = start + nblocks * L

                selected = indices[start:stop]

                if len(selected) != nblocks * L:
                    raise RuntimeError("Incorrect block selection.")

                H = block_histograms(
                    selected, L, nblocks, ncells
                )

                observed_j, observed_tv = adjacent_metrics(H)

                null_j = np.empty(N_PERM)
                null_tv = np.empty(N_PERM)

                for r in range(N_PERM):
                    shuffled = rng.permutation(selected)

                    Hp = block_histograms(
                        shuffled, L, nblocks, ncells
                    )

                    null_j[r], null_tv[r] = adjacent_metrics(Hp)

                j_mean = null_j.mean()
                tv_mean = null_tv.mean()

                j_lo, j_hi = np.quantile(
                    null_j, [0.025, 0.975]
                )

                tv_lo, tv_hi = np.quantile(
                    null_tv, [0.025, 0.975]
                )

                print(
                    f"{L:5d} | {offset:6d} | {nblocks:6d} | "
                    f"{observed_j:.6f} | "
                    f"{j_mean:.6f} [{j_lo:.6f}, {j_hi:.6f}] | "
                    f"{observed_tv:.6f} | "
                    f"{tv_mean:.6f} [{tv_lo:.6f}, {tv_hi:.6f}]"
                )

print("\nPartition-origin sensitivity analysis completed.")
