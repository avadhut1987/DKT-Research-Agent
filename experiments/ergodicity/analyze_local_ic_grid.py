
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("results/ergodicity_study")
OUT = BASE / "local_ic_grid"
OUT.mkdir(parents=True, exist_ok=True)

theta_values = [0.990, 0.995, 1.000, 1.005, 1.010]
phi_values = [-0.760, -0.755, -0.750, -0.745, -0.740]

# Explicit mapping avoids ambiguity between filename conventions.
FILES = {
    # theta = 0.990
    (0.990, -0.760): OUT / "theta099_phi076.csv",
    (0.990, -0.755): OUT / "theta0990_phi0755.csv",
    (0.990, -0.750): BASE / "multiple_ic_1024/ic03.csv",
    (0.990, -0.745): OUT / "theta0990_phi0745.csv",
    (0.990, -0.740): OUT / "theta099_phi074.csv",

    # theta = 0.995
    (0.995, -0.760): OUT / "theta0995_phi0760.csv",
    (0.995, -0.755): OUT / "theta0995_phi0755.csv",
    (0.995, -0.750): OUT / "theta0995_phi0750.csv",
    (0.995, -0.745): OUT / "theta0995_phi0745.csv",
    (0.995, -0.740): OUT / "theta0995_phi0740.csv",

    # theta = 1.000
    (1.000, -0.760): BASE / "multiple_ic_1024/ic05.csv",
    (1.000, -0.755): OUT / "theta1000_phi0755.csv",
    (1.000, -0.750): BASE / "multiple_ic/ic01_1024.csv",
    (1.000, -0.745): OUT / "theta1000_phi0745.csv",
    (1.000, -0.740): BASE / "multiple_ic_1024/ic04.csv",

    # theta = 1.005
    (1.005, -0.760): OUT / "theta1005_phi0760.csv",
    (1.005, -0.755): OUT / "theta1005_phi0755.csv",
    (1.005, -0.750): OUT / "theta1005_phi0750.csv",
    (1.005, -0.745): OUT / "theta1005_phi0745.csv",
    (1.005, -0.740): OUT / "theta1005_phi0740.csv",

    # theta = 1.010
    (1.010, -0.760): OUT / "theta101_phi076.csv",
    (1.010, -0.755): OUT / "theta1010_phi0755.csv",
    (1.010, -0.750): BASE / "multiple_ic_1024/ic02.csv",
    (1.010, -0.745): OUT / "theta1010_phi0745.csv",
    (1.010, -0.740): OUT / "theta101_phi074.csv",
}

times = [2500, 5000, 10000, 20000, 50000, 100000]
observables = ["X2", "Y2", "Z2"]

assert len(FILES) == 25
assert len(set(FILES.values())) == 25

results = []

for (theta, phi), path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != 100001:
        raise ValueError(f"Unexpected trajectory length: {path}")

    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=float)[1:]

    values = {
        "X2": xyz[:, 0] ** 2,
        "Y2": xyz[:, 1] ** 2,
        "Z2": xyz[:, 2] ** 2,
    }

    for T in times:
        for obs in observables:
            results.append({
                "theta": theta,
                "phi": phi,
                "T": T,
                "observable": obs,
                "average": np.mean(values[obs][:T]),
            })

data = pd.DataFrame(results)
data.to_csv(OUT / "local_grid_observable_averages.csv", index=False)

# Print the 5x5 maps at every observation time.
for T in times:
    print("\n" + "=" * 85)
    print(f"OBSERVATION TIME T = {T}")
    print("=" * 85)

    for obs in observables:
        matrix = (
            data[(data["T"] == T) & (data["observable"] == obs)]
            .pivot(index="theta", columns="phi", values="average")
            .reindex(index=theta_values, columns=phi_values)
        )

        print(f"\n{obs}")
        print(matrix.to_string(float_format=lambda x: f"{x:.8f}"))

# Spatial variation across initial conditions.
print("\n" + "=" * 85)
print("SPATIAL VARIATION ACROSS THE 25 INITIAL CONDITIONS")
print("=" * 85)

for T in times:
    print(f"\nT = {T}")
    for obs in observables:
        subset = data[
            (data["T"] == T) & (data["observable"] == obs)
        ]["average"]

        print(
            f"{obs}: min={subset.min():.8f}, "
            f"max={subset.max():.8f}, "
            f"range={subset.max()-subset.min():.8f}, "
            f"std={subset.std():.8f}"
        )

# Change in cumulative averages from 50k to 100k.
print("\n" + "=" * 85)
print("CHANGE IN CUMULATIVE AVERAGES: T=50000 TO T=100000")
print("=" * 85)

early = data[data["T"] == 50000].set_index(
    ["theta", "phi", "observable"]
)["average"]

late = data[data["T"] == 100000].set_index(
    ["theta", "phi", "observable"]
)["average"]

delta = (late - early).rename("delta").reset_index()

for obs in observables:
    d = delta[delta["observable"] == obs]["delta"]
    print(
        f"{obs}: max |delta|={d.abs().max():.8f}, "
        f"RMS delta={np.sqrt(np.mean(d**2)):.8f}"
    )

delta.to_csv(OUT / "local_grid_average_changes.csv", index=False)

# Final-time maps with common colour scale for each observable.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, obs in zip(axes, observables):
    matrix = (
        data[(data["T"] == 100000) & (data["observable"] == obs)]
        .pivot(index="theta", columns="phi", values="average")
        .reindex(index=theta_values, columns=phi_values)
    )

    image = ax.imshow(
        matrix.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[
            phi_values[0] - 0.0025,
            phi_values[-1] + 0.0025,
            theta_values[0] - 0.0025,
            theta_values[-1] + 0.0025,
        ],
    )

    ax.set_title(rf"$\overline{{{obs[0]}^2}}$")
    ax.set_xlabel(r"$\phi_0$")
    ax.set_ylabel(r"$\theta_0$")
    ax.set_xticks(phi_values)
    ax.set_yticks(theta_values)
    fig.colorbar(image, ax=ax)

fig.suptitle(
    r"Local initial-condition dependence: "
    r"$k_r=1,\ k_\theta=10^4,\ T=10^5$"
)
fig.tight_layout()

fig.savefig(OUT / "local_grid_second_moments_5x5.png", dpi=200)
plt.close(fig)

# Evolution of spatial standard deviation with observation time.
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, obs in zip(axes, observables):
    stds = []
    for T in times:
        subset = data[
            (data["T"] == T) & (data["observable"] == obs)
        ]["average"]
        stds.append(subset.std())

    ax.plot(times, stds, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Observation time T")
    ax.set_ylabel("Spatial standard deviation")
    ax.set_title(obs)

fig.suptitle("Initial-condition dependence versus observation time")
fig.tight_layout()

fig.savefig(OUT / "local_grid_spatial_std_vs_time.png", dpi=200)
plt.close(fig)

print("\nSaved:")
print(OUT / "local_grid_observable_averages.csv")
print(OUT / "local_grid_average_changes.csv")
print(OUT / "local_grid_second_moments_5x5.png")
print(OUT / "local_grid_spatial_std_vs_time.png")
