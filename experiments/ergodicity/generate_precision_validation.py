import math
import subprocess
from pathlib import Path

GENERATOR = "experiments/ergodicity/generate_bigfloat_trajectory.jl"
OUT = Path("results/ergodicity_study/controlled_perturbations_2048")
OUT.mkdir(parents=True, exist_ok=True)

BITS = "2048"
KR = "1.0"
KT = "10000.0"
NSTEPS = "100000"

REFERENCES = {
    "A": (1.000, -0.750),
    "B": (0.995, -0.760),
    "C": (1.005, -0.745),
}

CASES = [
    ("A", "phi", 0.000625),
    ("A", "theta", 0.000625),
    ("B", "phi", 0.002500),
    ("C", "phi", 0.000625),
    ("C", "theta", 0.001250),
]

jobs = {}

# Reference trajectories.
for name, (theta, phi) in REFERENCES.items():
    jobs[f"{name}_reference"] = (theta, phi)

# Symmetric perturbation pairs.
for name, axis, h in CASES:
    theta0, phi0 = REFERENCES[name]

    if axis == "theta":
        jobs[f"{name}_h{h:.7f}_theta_plus"] = (theta0+h, phi0)
        jobs[f"{name}_h{h:.7f}_theta_minus"] = (theta0-h, phi0)
    else:
        dp = h / math.sin(theta0)
        jobs[f"{name}_h{h:.7f}_phi_plus"] = (theta0, phi0+dp)
        jobs[f"{name}_h{h:.7f}_phi_minus"] = (theta0, phi0-dp)

for tag, (theta, phi) in jobs.items():
    output = OUT / f"{tag}.csv"

    if output.exists():
        print(f"SKIP existing: {output}")
        continue

    command = [
        "julia",
        "--project=QuantumKickedTop.jl",
        GENERATOR,
        BITS, KR, KT,
        f"{theta:.15f}",
        f"{phi:.15f}",
        NSTEPS,
        str(output),
    ]

    print("\n" + "="*75)
    print(f"{tag}: theta={theta:.15f}, phi={phi:.15f}")
    print("="*75)

    subprocess.run(command, check=True)

print("\n2048-bit precision-validation trajectories completed.")
