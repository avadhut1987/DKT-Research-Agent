import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

BASE = Path("results/ergodicity_study/local_ic_grid")
INPUT = BASE / "local_grid_observable_averages.csv"
OUT = BASE / "spatial_structure"
OUT.mkdir(parents=True, exist_ok=True)

TIMES = [10000, 50000, 100000]
OBSERVABLES = ["X2", "Y2", "Z2"]
MAX_DISTANCE = 0.012

data = pd.read_csv(INPUT)

required = {"theta", "phi", "T", "observable", "average"}
if not required.issubset(data.columns):
    raise ValueError(f"Missing columns: {required - set(data.columns)}")

records = []

for T in TIMES:
    subset = data[data["T"] == T]
    pivot = subset.pivot(
        index=["theta", "phi"],
        columns="observable",
        values="average"
    )

    if pivot.empty or not set(OBSERVABLES).issubset(pivot.columns):
        raise ValueError(f"Missing observable data at T={T}")

    points = []
    for (theta, phi), row in pivot.iterrows():
        points.append({
            "theta": float(theta),
            "phi": float(phi),
            **{obs: float(row[obs]) for obs in OBSERVABLES}
        })

    for a, b in combinations(points, 2):
        th1, ph1 = a["theta"], a["phi"]
        th2, ph2 = b["theta"], b["phi"]

        dtheta = th2 - th1
        dphi = ph2 - ph1

        # Great-circle distance on the unit sphere.
        cos_d = (
            np.cos(th1) * np.cos(th2)
            + np.sin(th1) * np.sin(th2) * np.cos(dphi)
        )
        distance = np.arccos(np.clip(cos_d, -1.0, 1.0))

        if distance > MAX_DISTANCE:
            continue

        theta_mid = 0.5 * (th1 + th2)
        dphi_physical = np.sin(theta_mid) * abs(dphi)

        if abs(dtheta) > 1.5 * dphi_physical:
            direction = "theta-dominated"
        elif dphi_physical > 1.5 * abs(dtheta):
            direction = "phi-dominated"
        else:
            direction = "mixed"

        record = {
            "T": T,
            "theta1": th1, "phi1": ph1,
            "theta2": th2, "phi2": ph2,
            "distance": distance,
            "dtheta": abs(dtheta),
            "dphi_physical": dphi_physical,
            "direction": direction,
        }

        for obs in OBSERVABLES:
            record[f"D_{obs}"] = abs(a[obs] - b[obs])

        records.append(record)

pairs = pd.DataFrame(records)
pairs.to_csv(OUT / "birkhoff_pairwise_differences.csv", index=False)

# Distance bins chosen to separate the scales represented by the 5x5 grid.
edges = [0.0, 0.0055, 0.0075, 0.0095, 0.0125]
labels = [
    "0–0.0055",
    "0.0055–0.0075",
    "0.0075–0.0095",
    "0.0095–0.0125",
]
pairs["distance_bin"] = pd.cut(
    pairs["distance"], bins=edges, labels=labels,
    include_lowest=True, right=True
)

summary_rows = []
for (T, distance_bin), group in pairs.groupby(
    ["T", "distance_bin"], observed=False
):
    for obs in OBSERVABLES:
        values = group[f"D_{obs}"].to_numpy()
        if len(values) == 0:
            continue
        summary_rows.append({
            "T": T,
            "distance_bin": str(distance_bin),
            "n_pairs": len(values),
            "observable": obs,
            "distance_median": group["distance"].median(),
            "D_median": np.median(values),
            "D_q25": np.quantile(values, 0.25),
            "D_q75": np.quantile(values, 0.75),
            "D_min": np.min(values),
            "D_max": np.max(values),
            "slope_median": np.median(values / group["distance"].to_numpy()),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "birkhoff_separation_summary.csv", index=False)

print("\nPAIRWISE SEPARATION ANALYSIS")
print("=" * 100)
print(f"Number of initial conditions: {pairs[['theta1','phi1']].drop_duplicates().shape[0]}")
print(f"Maximum spherical separation: {MAX_DISTANCE}")
print(f"Number of pairs retained per observation time: {len(pairs) // len(TIMES)}")

for T in TIMES:
    print(f"\n{'=' * 90}\nT = {T}\n{'=' * 90}")
    for obs in OBSERVABLES:
        print(f"\n{obs}")
        view = summary[
            (summary["T"] == T) & (summary["observable"] == obs)
        ]
        if view.empty:
            print("No pairs in selected distance bins.")
            continue
        print(view[
            ["distance_bin", "n_pairs", "distance_median",
             "D_median", "D_q25", "D_q75", "D_min", "D_max",
             "slope_median"]
        ].to_string(index=False, float_format=lambda x: f"{x:.7g}"))

# Directional comparison at each time.
print("\nDIRECTIONAL COMPARISON")
print("=" * 100)
for T in TIMES:
    print(f"\nT = {T}")
    view = pairs[pairs["T"] == T]
    for direction, group in view.groupby("direction"):
        print(f"\n{direction}: {len(group)} pairs")
        for obs in OBSERVABLES:
            print(
                f"  {obs}: median D={group[f'D_{obs}'].median():.7g}, "
                f"median D/d={np.median(group[f'D_{obs}']/group['distance']):.7g}"
            )

# Plot median differences against separation, with interquartile ranges.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
time_styles = {
    10000: "o-",
    50000: "s-",
    100000: "^-",
}

for ax, obs in zip(axes, OBSERVABLES):
    for T in TIMES:
        view = summary[
            (summary["T"] == T) & (summary["observable"] == obs)
        ].dropna(subset=["distance_median", "D_median"])

        if view.empty:
            continue

        x = view["distance_median"].to_numpy()
        y = view["D_median"].to_numpy()
        lower = y - view["D_q25"].to_numpy()
        upper = view["D_q75"].to_numpy() - y

        ax.errorbar(
            x, y, yerr=np.vstack([lower, upper]),
            fmt=time_styles[T], capsize=3, label=f"T={T}"
        )

    ax.set_title(obs)
    ax.set_xlabel("Initial-condition geodesic separation")
    ax.set_ylabel(r"$|\Delta \overline{A}|$")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("Finite-time Birkhoff-average differences vs initial separation")
fig.tight_layout()
fig.savefig(
    OUT / "birkhoff_difference_vs_separation.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

print("\nSaved:")
print(OUT / "birkhoff_pairwise_differences.csv")
print(OUT / "birkhoff_separation_summary.csv")
print(OUT / "birkhoff_difference_vs_separation.png")
print("\nBirkhoff separation analysis completed.")
