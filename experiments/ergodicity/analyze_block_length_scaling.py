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
SEED = 20261112

def load_xyz(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.column_stack((d["X"], d["Y"], d["Z"]))[1:100001]

def cell_indices(xyz, nz, nphi):
    z = np.clip(xyz[:, 2], -1, 1)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    iz = np.minimum(((z + 1)*nz/2).astype(int), nz-1)
    ip = np.minimum(
        ((phi + np.pi)*nphi/(2*np.pi)).astype(int),
        nphi-1
    )
    return iz*nphi + ip

def metrics(indices, block_length, ncells):
    nblocks = len(indices)//block_length
    indices = indices[:nblocks*block_length]

    H = np.array([
        np.bincount(
            indices[i*block_length:(i+1)*block_length],
            minlength=ncells
        )
        for i in range(nblocks)
    ])

    masks = H > 0
    probs = H / block_length

    js, tvs = [], []

    for i in range(nblocks-1):
        a, b = masks[i], masks[i+1]
        union = np.logical_or(a,b).sum()
        inter = np.logical_and(a,b).sum()

        js.append(inter/union if union else np.nan)
        tvs.append(0.5*np.abs(probs[i]-probs[i+1]).sum())

    return np.nanmean(js), np.nanmean(tvs), nblocks

rng = np.random.default_rng(SEED)

for nz, nphi in RESOLUTIONS:
    ncells = nz*nphi

    print("\n"+"="*90)
    print(f"RESOLUTION {nz} x {nphi}")
    print("="*90)

    for name, path in FILES.items():
        xyz = load_xyz(path)
        indices = cell_indices(xyz,nz,nphi)

        print(f"\n{name}")
        print(
            "L | blocks | chronological Jaccard | "
            "shuffled Jaccard (mean +/- sd) | "
            "chronological TV | shuffled TV (mean +/- sd)"
        )

        for L in BLOCK_LENGTHS:
            obs_j, obs_tv, nb = metrics(indices,L,ncells)

            null_j, null_tv = [], []

            for _ in range(N_PERM):
                shuffled = rng.permutation(indices)
                j,tv,_ = metrics(shuffled,L,ncells)
                null_j.append(j)
                null_tv.append(tv)

            null_j = np.asarray(null_j)
            null_tv = np.asarray(null_tv)

            print(
                f"{L:5d} | {nb:6d} | "
                f"{obs_j:.6f} | "
                f"{null_j.mean():.6f} +/- {null_j.std():.6f} | "
                f"{obs_tv:.6f} | "
                f"{null_tv.mean():.6f} +/- {null_tv.std():.6f}"
            )

print("\nBlock-length scaling analysis completed.")
