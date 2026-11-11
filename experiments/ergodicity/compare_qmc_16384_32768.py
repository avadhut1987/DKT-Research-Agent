import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/ergodicity_study")

DIR14 = BASE / "ulam_qmc_16384"
DIR15 = BASE / "ulam_qmc_32768"

SEEDS = [20261111, 20261112]
OBS = ["X", "Y", "Z", "X2", "Y2", "Z2", "XY", "YZ", "ZX"]

print("=" * 88)
print("NESTED SOBOL CONVERGENCE: 16384 -> 32768")
print("=" * 88)

rows = []

for seed in SEEDS:
    a = pd.read_csv(DIR14 / f"conditional_averages_seed_{seed}.csv")
    b = pd.read_csv(DIR15 / f"conditional_averages_seed_{seed}.csv")

    assert np.array_equal(a["cell"], b["cell"])

    print(f"\nSCRAMBLE SEED: {seed}")
    print("-" * 88)

    for name in OBS:
        v14 = a[name].to_numpy()
        v15 = b[name].to_numpy()

        diff = v15 - v14
        rms_change = np.sqrt(np.mean(diff**2))
        rms_signal = np.sqrt(np.mean(v15**2))

        relative = rms_change / max(rms_signal, 1e-14)
        corr = np.corrcoef(v14, v15)[0, 1]

        rows.append({
            "seed": seed,
            "observable": name,
            "rms_change": rms_change,
            "relative_change": relative,
            "correlation": corr,
            "max_abs_change": np.max(np.abs(diff)),
        })

        print(
            f"{name:4s} "
            f"RMS(change)={rms_change:.7e} "
            f"relative={relative:.7e} "
            f"corr={corr:.8f} "
            f"max={np.max(np.abs(diff)):.7e}"
        )

OUT = BASE / "qmc_16384_32768_comparison"
OUT.mkdir(exist_ok=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "nested_convergence.csv", index=False)

summary = df.groupby("observable").agg(
    mean_rms_change=("rms_change", "mean"),
    mean_relative_change=("relative_change", "mean"),
    mean_correlation=("correlation", "mean"),
).reset_index()

summary.to_csv(OUT / "summary.csv", index=False)

print("\nSUMMARY")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.7e}"))
print(f"\nSaved in: {OUT}")
