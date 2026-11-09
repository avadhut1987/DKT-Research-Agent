
#!/usr/bin/env julia

# Classical Double Kicked Top trajectory simulator.
# Uses the QuantumKickedTop.jl package as the physics engine.

using QuantumKickedTop
using LinearAlgebra

const ClassicalMap = QuantumKickedTop.ClassicalMap
const ClassicalUtils = QuantumKickedTop.ClassicalUtils

function usage()
    println("""
    Usage:
      julia --project=QuantumKickedTop.jl scripts/julia/simulate_dkt.jl \\
          kr ktheta theta0 phi0 nsteps output_csv [p]

    Arguments:
      kr          Mean kicking strength
      ktheta      Asymmetry / time-reversal-breaking parameter
      theta0      Initial polar angle (radians)
      phi0        Initial azimuthal angle (radians)
      nsteps      Number of kicks (positive integer)
      output_csv  Output trajectory CSV filename
      p           Free rotation angle (optional, default: pi/2)

    Parameter convention:
      k      = kr + ktheta
      kprime = kr - ktheta
    """)
end

function main(args)
    if length(args) < 6 || length(args) > 7
        usage()
        return 1
    end

    kr       = parse(Float64, args[1])
    ktheta   = parse(Float64, args[2])
    theta0   = parse(Float64, args[3])
    phi0     = parse(Float64, args[4])
    nsteps   = parse(Int, args[5])
    output   = args[6]
    p        = length(args) == 7 ? parse(Float64, args[7]) : pi / 2

    values = (kr, ktheta, theta0, phi0, p)

    all(isfinite, values) ||
        error("All floating-point parameters must be finite.")

    nsteps > 0 ||
        error("nsteps must be a positive integer.")

    # Convert research parameters to the package convention.
    k = kr + ktheta
    kprime = kr - ktheta

    params = (
        p = p,
        k = k,
        kprime = kprime
    )

    # Initial condition on the unit sphere.
    R = ClassicalUtils.spherical_to_cartesian(theta0, phi0)

    initial_norm = norm(R)
    max_norm_error = 0.0

    # Create output directory if necessary.
    output_dir = dirname(abspath(output))
    mkpath(output_dir)

    # Save the initial state and all subsequent kicks.
    open(output, "w") do io
        println(io, "step,X,Y,Z,norm")

        println(io,
            "0,", R[1], ",", R[2], ",", R[3], ",", norm(R)
        )

        for step in 1:nsteps
            R = ClassicalMap.map(R, params)

            current_norm = norm(R)
            max_norm_error = max(
                max_norm_error,
                abs(current_norm - initial_norm)
            )

            println(io,
                step, ",",
                R[1], ",",
                R[2], ",",
                R[3], ",",
                current_norm
            )
        end
    end

    println("status=success")
    println("kr=", kr)
    println("ktheta=", ktheta)
    println("k=", k)
    println("kprime=", kprime)
    println("p=", p)
    println("theta0=", theta0)
    println("phi0=", phi0)
    println("nsteps=", nsteps)
    println("trajectory_rows=", nsteps + 1)
    println("initial_norm=", initial_norm)
    println("final_norm=", norm(R))
    println("max_norm_error=", max_norm_error)
    println("output_csv=", abspath(output))

    return 0
end

exit(main(ARGS))
