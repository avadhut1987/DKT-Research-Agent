import subprocess
from pathlib import Path
import math

ROOT = Path(".")
GENERATOR = "experiments/ergodicity/generate_bigfloat_trajectory.jl"
OUT = Path("results/ergodicity_study/controlled_perturbations")
OUT.mkdir(parents=True, exist_ok=True)

BITS = "1024"
KR = "1.0"
KTHETA = "10000.0"
NSTEPS = "100000"

REFERENCES = {
    "A": (1.000, -0.750),
    "B": (0.995, -0.760),
    "C": (1.005, -0.745),
}

DISPLACEMENTS = [0.0025, 0.00125, 0.000625]

for name, (theta0, phi0) in REFERENCES.items():
    for h in DISPLACEMENTS:
        # Equal geodesic displacement to first order.
        perturbations = [
            ("theta_plus", theta0 + h, phi0),
            ("theta_minus", theta0 - h, phi0),
            ("phi_plus", theta0, phi0 + h / math.sin(theta0)),
            ("phi_minus", theta0, phi0 - h / math.sin(theta0)),
        ]

        for direction, theta, phi in perturbations:
            tag = f"{name}_h{h:.7f}_{direction}"
            output = OUT / f"{tag}.csv"

            if output.exists():
                print(f"SKIP existing: {output}")
                continue

            command = [
                "julia",
                "--project=QuantumKickedTop.jl",
                GENERATOR,
                BITS, KR, KTHETA,
                f"{theta:.12f}",
                f"{phi:.12f}",
                NSTEPS,
                str(output),
            ]

            print("\n" + "=" * 75)
            print(f"Generating {tag}")
            print(f"theta={theta:.12f}, phi={phi:.12f}, h={h}")
            print("=" * 75)

            subprocess.run(command, check=True)

print("\nAll controlled perturbation trajectories generated.")
