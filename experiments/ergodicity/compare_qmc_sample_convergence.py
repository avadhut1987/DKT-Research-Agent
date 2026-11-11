import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/ergodicity_study")

DIR12 = BASE / "ulam_qmc"
DIR14 = BASE / "ulam_qmc_16384"

SEEDS = [20261111, 20261112]

OBS = ["X", "Y", "Z", "X2", "Y2", "Z2", "XY", "YZ", "ZX"]

rows = []

print("=" * 88)
print("NESTED SOBOL SAMPLE-CONVERGENCE TEST")
print("=" * 88)

for seed in SEEDS:
    a = pd.read_csv(
        DIR12 / f"conditional_averages_seed_{seed}.csv"
    )
    b = pd.read_csv(
        DIR14 / f"conditional_averages_seed_{seed}.csv"
    )

    assert np.array_equal(a["cell"], b["cell"])

    print(f"\nSCRAMBLE SEED: {seed}")
    print("-" * 88)

    for name in OBS:
        v12 = a[name].to_numpy()
        v14 = b[name].to_numpy()

        diff = v14 - v12

        rms12 = np.sqrt(np.mean(v12**2))
        rms14 = np.sqrt(np.mean(v14**2))
        rms_diff = np.sqrt(np.mean(diff**2))

        relative = rms_diff / max(rms14, 1e-14)
        corr = np.corrcoef(v12, v14)[0, 1]

        rows.append({
            "seed": seed,
            "observable": name,
            "rms_M12": rms12,
            "rms_M14": rms14,
            "rms_change": rms_diff,
            "relative_change": relative,
            "correlation": corr,
            "max_abs_change": np.max(np.abs(diff)),
        })

        print(
            f"{name:4s} "
            f"RMS(change)={rms_diff:.7e} "
            f"relative={relative:.7e} "
            f"corr={corr:.8f} "
            f"max={np.max(np.abs(diff)):.7e}"
        )

df = pd.DataFrame(rows)

OUT = BASE / "qmc_sample_convergence"
OUT.mkdir(exist_ok=True)

df.to_csv(OUT / "nested_sample_convergence.csv", index=False)

summary = (
    df.groupby("observable")
      .agg(
          mean_rms_change=("rms_change", "mean"),
          max_rms_change=("rms_change", "max"),
          mean_relative_change=("relative_change", "mean"),
          mean_correlation=("correlation", "mean"),
      )
      .reset_index()
)

summary.to_csv(OUT / "convergence_summary.csv", index=False)

print("\n" + "=" * 88)
print("SUMMARY ACROSS SCRAMBLES")
print("=" * 88)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.7e}"))

print(f"\nSaved results in {OUT}")
