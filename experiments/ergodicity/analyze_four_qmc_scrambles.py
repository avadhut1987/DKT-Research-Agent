import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/ergodicity_study")

FILES = [
    BASE / "ulam_qmc_16384" /
        "conditional_averages_seed_20261111.csv",
    BASE / "ulam_qmc_16384" /
        "conditional_averages_seed_20261112.csv",
    BASE / "ulam_qmc_16384_extra" /
        "conditional_averages_seed_20261113.csv",
    BASE / "ulam_qmc_16384_extra" /
        "conditional_averages_seed_20261114.csv",
]

OBS = ["X", "Y", "Z", "X2", "Y2", "Z2", "XY", "YZ", "ZX"]

data = [pd.read_csv(f) for f in FILES]

for d in data[1:]:
    assert np.array_equal(data[0]["cell"], d["cell"])

OUT = BASE / "qmc_four_scrambles"
OUT.mkdir(exist_ok=True)

summary = []
cell_results = pd.DataFrame({"cell": data[0]["cell"]})

print("=" * 88)
print("FOUR-INDEPENDENT-SCRAMBLE QMC UNCERTAINTY ANALYSIS")
print("=" * 88)

for name in OBS:
    values = np.stack([d[name].to_numpy() for d in data])

    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    rms_mean = np.sqrt(np.mean(mean**2))
    rms_std = np.sqrt(np.mean(std**2))

    relative_uncertainty = rms_std / max(rms_mean, 1e-14)

    cell_results[f"{name}_mean"] = mean
    cell_results[f"{name}_std"] = std

    summary.append({
        "observable": name,
        "rms_mean": rms_mean,
        "rms_scramble_std": rms_std,
        "relative_uncertainty": relative_uncertainty,
        "median_cell_std": np.median(std),
        "max_cell_std": np.max(std),
    })

    print(
        f"{name:4s} "
        f"RMS(mean)={rms_mean:.7e} "
        f"RMS(scramble std)={rms_std:.7e} "
        f"relative uncertainty={relative_uncertainty:.7e} "
        f"max cell std={np.max(std):.7e}"
    )

summary = pd.DataFrame(summary)

summary.to_csv(OUT / "four_scramble_summary.csv", index=False)
cell_results.to_csv(OUT / "four_scramble_cell_estimates.csv", index=False)

print("\nSaved:")
print(OUT / "four_scramble_summary.csv")
print(OUT / "four_scramble_cell_estimates.csv")
