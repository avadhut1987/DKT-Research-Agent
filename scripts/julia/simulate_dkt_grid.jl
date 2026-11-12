#!/usr/bin/env julia

# Batch classical Double Kicked Top phase-space evolution.
# Saves every point of every orbit, from kick 0 through nsteps.

using QuantumKickedTop
using LinearAlgebra

const ClassicalMap = QuantumKickedTop.ClassicalMap
const ClassicalUtils = QuantumKickedTop.ClassicalUtils

function usage()
    println("""
    Usage:
      julia --project=QuantumKickedTop.jl scripts/julia/simulate_dkt_grid.jl \\
          kr ktheta p grid_size nsteps output_csv

    grid_size must be between 5 and 50.
    Every orbit point, including the initial state, is saved.
    """)
end

function spherical_coordinates(R)
    r = norm(R)
    r > 0.0 || error("Encountered a zero-norm state.")

    theta = acos(clamp(R[3] / r, -1.0, 1.0))
    phi = atan(R[2], R[1])

    return theta, phi, r
end

function main(args)
    if length(args) != 6
        usage()
        return 1
    end

    kr        = parse(Float64, args[1])
    ktheta    = parse(Float64, args[2])
    p         = parse(Float64, args[3])
    grid_size = parse(Int, args[4])
    nsteps    = parse(Int, args[5])
    output    = args[6]

    all(isfinite, (kr, ktheta, p)) ||
        error("All floating-point parameters must be finite.")

    5 <= grid_size <= 50 ||
        error("grid_size must be between 5 and 50.")

    nsteps > 0 ||
        error("nsteps must be a positive integer.")

    k = kr + ktheta
    kprime = kr - ktheta

    params = (p=p, k=k, kprime=kprime)

    mkpath(dirname(abspath(output)))

    max_norm_error = 0.0
    total_initial_conditions = grid_size^2
    total_samples = total_initial_conditions * (nsteps + 1)

    open(output, "w") do io
        println(
            io,
            "grid_i,grid_j,kick,theta0,phi0,theta,phi,X,Y,Z,norm"
        )

        for i in 0:(grid_size - 1)
            theta0 = (i + 0.5) * pi / grid_size

            for j in 0:(grid_size - 1)
                phi0 = -pi + 2.0 * pi * j / grid_size

                R = ClassicalUtils.spherical_to_cartesian(
                    theta0, phi0
                )

                initial_norm = norm(R)

                # Save the initial point (kick 0).
                theta, phi, current_norm = spherical_coordinates(R)

                println(
                    io,
                    i + 1, ",", j + 1, ",0,",
                    theta0, ",", phi0, ",",
                    theta, ",", phi, ",",
                    R[1], ",", R[2], ",", R[3], ",",
                    current_norm
                )

                max_norm_error = max(
                    max_norm_error,
                    abs(current_norm - initial_norm)
                )

                # Save the state after every kick.
                for step in 1:nsteps
                    R = ClassicalMap.map(R, params)

                    all(isfinite, R) ||
                        error(
                            "Non-finite state at grid point ($i, $j), " *
                            "kick $step."
                        )

                    theta, phi, current_norm = spherical_coordinates(R)

                    max_norm_error = max(
                        max_norm_error,
                        abs(current_norm - initial_norm)
                    )

                    println(
                        io,
                        i + 1, ",", j + 1, ",", step, ",",
                        theta0, ",", phi0, ",",
                        theta, ",", phi, ",",
                        R[1], ",", R[2], ",", R[3], ",",
                        current_norm
                    )
                end
            end
        end
    end

    println("status=success")
    println("kr=", kr)
    println("ktheta=", ktheta)
    println("k=", k)
    println("kprime=", kprime)
    println("p=", p)
    println("grid_size=", grid_size)
    println("initial_conditions=", total_initial_conditions)
    println("nsteps=", nsteps)
    println("saved_samples=", total_samples)
    println("max_norm_error=", max_norm_error)
    println("output_csv=", abspath(output))

    return 0
end

exit(main(ARGS))
