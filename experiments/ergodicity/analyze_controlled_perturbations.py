import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("results/ergodicity_study")
PERT = BASE / "controlled_perturbations"
OUT = PERT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

OBSERVABLES = ["X2", "Y2", "Z2"]
TIMES = [10_000, 50_000, 100_000]
HS = [0.0025, 0.00125, 0.000625]

REFERENCES = {
    "A": {
        "theta": 1.000,
        "phi": -0.750,
        "file": BASE / "multiple_ic/ic01_1024.csv",
    },
    "B": {
        "theta": 0.995,
        "phi": -0.760,
        "file": BASE / "local_ic_grid/theta099_phi076.csv",
    },
    "C": {
        "theta": 1.005,
        "phi": -0.745,
        "file": BASE / "local_ic_grid/theta1005_phi0745.csv",
    },
}

def load_observable_averages(path):
    data = pd.read_csv(path)

    required = {"step", "X", "Y", "Z"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    # Exclude the initial condition at step zero.
    data = data[data["step"] > 0].sort_values("step")

    if len(data) < max(TIMES):
        raise ValueError(
            f"{path}: only {len(data)} post-initial points; "
            f"need {max(TIMES)}"
        )

    x = data["X"].to_numpy(dtype=float)
    y = data["Y"].to_numpy(dtype=float)
    z = data["Z"].to_numpy(dtype=float)

    values = {
        "X2": x*x,
        "Y2": y*y,
        "Z2": z*z,
    }

    result = {}
    for T in TIMES:
        result[T] = {
            name: float(np.mean(arr[:T]))
            for name, arr in values.items()
        }

    return result


def spherical_distance(theta1, phi1, theta2, phi2):
    # Stable haversine form of the great-circle distance.
    dtheta = theta2 - theta1
    dphi = phi2 - phi1

    hav = (
        np.sin(dtheta / 2.0)**2
        + np.sin(theta1) * np.sin(theta2)
        * np.sin(dphi / 2.0)**2
    )
    hav = np.clip(hav, 0.0, 1.0)
    return float(2.0 * np.arcsin(np.sqrt(hav)))


def perturbation_coordinates(theta0, phi0, h, direction):
    if direction == "theta_plus":
        return theta0 + h, phi0
    if direction == "theta_minus":
        return theta0 - h, phi0
    if direction == "phi_plus":
        return theta0, phi0 + h / np.sin(theta0)
    if direction == "phi_minus":
        return theta0, phi0 - h / np.sin(theta0)
    raise ValueError(direction)


print("=" * 85)
print("CONTROLLED PERTURBATION ANALYSIS")
print("=" * 85)

reference_data = {}
for name, ref in REFERENCES.items():
    reference_data[name] = load_observable_averages(ref["file"])
    print(f"Loaded reference {name}: {ref['file']}")

rows = []

for ref_name, ref in REFERENCES.items():
    theta0, phi0 = ref["theta"], ref["phi"]
    ref_avg = reference_data[ref_name]

    for h in HS:
        for axis in ["theta", "phi"]:
            plus_direction = f"{axis}_plus"
            minus_direction = f"{axis}_minus"

            plus_theta, plus_phi = perturbation_coordinates(
                theta0, phi0, h, plus_direction
            )
            minus_theta, minus_phi = perturbation_coordinates(
                theta0, phi0, h, minus_direction
            )

            plus_file = PERT / (
                f"{ref_name}_h{h:.7f}_{plus_direction}.csv"
            )
            minus_file = PERT / (
                f"{ref_name}_h{h:.7f}_{minus_direction}.csv"
            )

            plus_avg = load_observable_averages(plus_file)
            minus_avg = load_observable_averages(minus_file)

            dplus = spherical_distance(
                theta0, phi0, plus_theta, plus_phi
            )
            dminus = spherical_distance(
                theta0, phi0, minus_theta, minus_phi
            )

            for T in TIMES:
                for obs in OBSERVABLES:
                    a0 = ref_avg[T][obs]
                    ap = plus_avg[T][obs]
                    am = minus_avg[T][obs]

                    # Signed one-sided responses.
                    delta_plus = ap - a0
                    delta_minus = am - a0

                    # Central derivative using the actual total
                    # geodesic distance between the two perturbed points.
                    central_derivative = (ap - am) / (dplus + dminus)

                    # Symmetric second response, normalized by h^2.
                    curvature = (ap + am - 2.0*a0) / (h*h)

                    rows.append({
                        "reference": ref_name,
                        "axis": axis,
                        "h_nominal": h,
                        "T": T,
                        "observable": obs,
                        "reference_average": a0,
                        "plus_average": ap,
                        "minus_average": am,
                        "delta_plus": delta_plus,
                        "delta_minus": delta_minus,
                        "abs_delta_plus": abs(delta_plus),
                        "abs_delta_minus": abs(delta_minus),
                        "distance_plus": dplus,
                        "distance_minus": dminus,
                        "central_derivative": central_derivative,
                        "abs_central_derivative": abs(central_derivative),
                        "symmetric_curvature": curvature,
                    })

df = pd.DataFrame(rows)
raw_path = OUT / "controlled_perturbation_responses.csv"
df.to_csv(raw_path, index=False)

# Summary across the three reference points.
summary = (
    df.groupby(["axis", "h_nominal", "T", "observable"])
      .agg(
          n=("reference", "count"),
          median_abs_delta_plus=("abs_delta_plus", "median"),
          median_abs_delta_minus=("abs_delta_minus", "median"),
          median_abs_derivative=("abs_central_derivative", "median"),
          median_signed_derivative=("central_derivative", "median"),
          median_abs_curvature=("symmetric_curvature",
                                lambda x: np.median(np.abs(x))),
      )
      .reset_index()
)

summary_path = OUT / "controlled_perturbation_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nREFERENCE AVERAGES AT T=100000")
for ref_name in REFERENCES:
    print(f"\nReference {ref_name}")
    for obs in OBSERVABLES:
        print(f"  {obs}: {reference_data[ref_name][100000][obs]:.10f}")

print("\nMEDIAN RESPONSE SUMMARY")
for T in TIMES:
    print(f"\nT = {T}")
    subset = summary[summary["T"] == T]
    for _, row in subset.iterrows():
        print(
            f"{row['axis']:5s} h={row['h_nominal']:.7f} "
            f"{row['observable']:3s} "
            f"|D|={row['median_abs_delta_plus']:.6e} "
            f"|Dcentral|={row['median_abs_derivative']:.6e} "
            f"|curvature|={row['median_abs_curvature']:.6e}"
        )

# Plot median absolute central derivative against displacement.
for T in TIMES:
    for obs in OBSERVABLES:
        fig, ax = plt.subplots(figsize=(7, 5))

        for axis, marker in [("theta", "o"), ("phi", "s")]:
            sub = summary[
                (summary["T"] == T)
                & (summary["observable"] == obs)
                & (summary["axis"] == axis)
            ].sort_values("h_nominal")

            ax.plot(
                sub["h_nominal"],
                sub["median_abs_derivative"],
                marker=marker,
                linewidth=1.8,
                label=axis,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Nominal spherical displacement h")
        ax.set_ylabel("Median absolute central derivative")
        ax.set_title(f"{obs}: local response at T={T:,}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            OUT / f"derivative_{obs}_T{T}.png",
            dpi=180
        )
        plt.close(fig)

# Plot individual reference responses to expose reference dependence.
for T in TIMES:
    for obs in OBSERVABLES:
        fig, ax = plt.subplots(figsize=(7, 5))

        for ref_name in REFERENCES:
            for axis, linestyle in [
                ("theta", "-"),
                ("phi", "--"),
            ]:
                sub = df[
                    (df["T"] == T)
                    & (df["observable"] == obs)
                    & (df["reference"] == ref_name)
                    & (df["axis"] == axis)
                ].sort_values("h_nominal")

                ax.plot(
                    sub["h_nominal"],
                    sub["abs_central_derivative"],
                    marker="o",
                    linestyle=linestyle,
                    label=f"{ref_name}, {axis}",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Nominal spherical displacement h")
        ax.set_ylabel("Absolute central derivative")
        ax.set_title(f"{obs}: reference-resolved response, T={T:,}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()

        fig.savefig(
            OUT / f"reference_derivative_{obs}_T{T}.png",
            dpi=180
        )
        plt.close(fig)

print("\nSaved:")
print(raw_path)
print(summary_path)
print(f"Plots: {OUT}")
print("\nControlled perturbation analysis completed.")
