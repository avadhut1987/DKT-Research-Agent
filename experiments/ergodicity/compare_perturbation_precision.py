import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/ergodicity_study")
DIR1024 = BASE / "controlled_perturbations"
DIR2048 = BASE / "controlled_perturbations_2048"
OUT = DIR2048 / "precision_comparison"
OUT.mkdir(parents=True, exist_ok=True)

TIMES = [10_000, 50_000, 100_000]
OBS = ["X2", "Y2", "Z2"]

REFERENCES = {
    "A": {
        "1024": BASE / "multiple_ic/ic01_1024.csv",
        "2048": DIR2048 / "A_reference.csv",
    },
    "B": {
        "1024": BASE / "local_ic_grid/theta0995_phi0760.csv",
        "2048": DIR2048 / "B_reference.csv",
    },
    "C": {
        "1024": BASE / "local_ic_grid/theta1005_phi0745.csv",
        "2048": DIR2048 / "C_reference.csv",
    },
}

CASES = [
    ("A", "phi", 0.000625),
    ("A", "theta", 0.000625),
    ("B", "phi", 0.002500),
    ("C", "phi", 0.000625),
    ("C", "theta", 0.001250),
]

def filename(name, axis, h, side):
    return f"{name}_h{h:.7f}_{axis}_{side}.csv"

def load_observable_averages(path):
    df = pd.read_csv(path)
    if len(df) < max(TIMES) + 1:
        raise ValueError(f"Insufficient rows in {path}: {len(df)}")

    result = {}
    for obs in OBS:
        if obs == "X2":
            values = df["X"].to_numpy() ** 2
        elif obs == "Y2":
            values = df["Y"].to_numpy() ** 2
        else:
            values = df["Z"].to_numpy() ** 2

        result[obs] = {
            T: float(np.mean(values[1:T+1]))
            for T in TIMES
        }
    return result

averages = {"1024": {}, "2048": {}}

for precision in ("1024", "2048"):
    for name in REFERENCES:
        path = REFERENCES[name][precision]
        print(f"Loading {precision}-bit reference {name}: {path}")
        averages[precision][f"{name}_reference"] = (
            load_observable_averages(path)
        )

    folder = DIR1024 if precision == "1024" else DIR2048

    for name, axis, h in CASES:
        for side in ("plus", "minus"):
            tag = filename(name, axis, h, side)
            path = folder / tag
            print(f"Loading {precision}-bit perturbation: {path}")
            averages[precision][tag] = load_observable_averages(path)

rows = []

for name, axis, h in CASES:
    for side in ("plus", "minus"):
        tag = filename(name, axis, h, side)

        for T in TIMES:
            for obs in OBS:
                ref_tag = f"{name}_reference"

                ref1024 = averages["1024"][ref_tag][obs][T]
                ref2048 = averages["2048"][ref_tag][obs][T]

                pert1024 = averages["1024"][tag][obs][T]
                pert2048 = averages["2048"][tag][obs][T]

                delta1024 = pert1024 - ref1024
                delta2048 = pert2048 - ref2048

                rows.append({
                    "reference": name,
                    "axis": axis,
                    "h": h,
                    "side": side,
                    "T": T,
                    "observable": obs,
                    "reference_1024": ref1024,
                    "reference_2048": ref2048,
                    "reference_precision_error": abs(ref2048-ref1024),
                    "perturbed_1024": pert1024,
                    "perturbed_2048": pert2048,
                    "perturbed_precision_error": abs(pert2048-pert1024),
                    "response_1024": delta1024,
                    "response_2048": delta2048,
                    "response_precision_error": abs(delta2048-delta1024),
                    "response_sign_agrees": (
                        np.sign(delta1024) == np.sign(delta2048)
                    ),
                })

df = pd.DataFrame(rows)
detail_path = OUT / "precision_response_comparison.csv"
df.to_csv(detail_path, index=False)

summary = (
    df.groupby(["reference", "axis", "h", "T"])
      .agg(
          max_reference_error=("reference_precision_error", "max"),
          max_perturbed_error=("perturbed_precision_error", "max"),
          max_response_error=("response_precision_error", "max"),
          sign_agreement_fraction=("response_sign_agrees", "mean"),
      )
      .reset_index()
)

summary_path = OUT / "precision_response_summary.csv"
summary.to_csv(summary_path, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda x: f"{x:.8e}")

print("\nREFERENCE AND PERTURBATION PRECISION COMPARISON")
print("=" * 110)
print(summary.to_string(index=False))

print("\nDETAILED SIGNED RESPONSES")
print("=" * 110)
print(
    df[[
        "reference", "axis", "h", "T", "observable", "side",
        "response_1024", "response_2048",
        "response_precision_error", "response_sign_agrees"
    ]].to_string(index=False)
)

print(f"\nSaved detailed results: {detail_path}")
print(f"Saved summary:          {summary_path}")
