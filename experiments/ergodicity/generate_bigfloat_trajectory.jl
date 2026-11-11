#!/usr/bin/env julia

using LinearAlgebra
using StaticArrays

function big_map(R, p, k, kp)
    X, Y, Z = R

    cp, sp = cos(p), sin(p)

    X3 = X*cp + Z*sp
    Y3 = Y
    Z3 = Z*cp - X*sp

    ckz, skz = cos(k*Z3), sin(k*Z3)

    X2 = X3*ckz - Y3*skz
    Y2 = Y3*ckz + X3*skz
    Z2 = Z3

    ckp, skp = cos(kp*X2), sin(kp*X2)

    return SVector{3,BigFloat}(
        X2,
        Y2*ckp - Z2*skp,
        Z2*ckp + Y2*skp
    )
end

function usage()
    println("""
    Usage:
      julia --project=QuantumKickedTop.jl \\
        experiments/ergodicity/generate_bigfloat_trajectory.jl \\
        bits kr ktheta theta0 phi0 nsteps output_csv [p]

    Example:
      julia --project=QuantumKickedTop.jl \\
        experiments/ergodicity/generate_bigfloat_trajectory.jl \\
        512 1.0 10000 1.0 -0.75 10000 \\
        results/ergodicity_study/kt10000_ic01.csv
    """)
end

function main(args)
    if length(args) < 7 || length(args) > 8
        usage()
        return 1
    end

    bits = parse(Int, args[1])
    bits >= 128 || error("Use at least 128 bits.")

    output = args[7]
    nsteps = parse(Int, args[6])

    nsteps > 0 || error("nsteps must be positive.")

    setprecision(BigFloat, bits) do
        kr = parse(BigFloat, args[2])
        kt = parse(BigFloat, args[3])
        theta = parse(BigFloat, args[4])
        phi = parse(BigFloat, args[5])

        p = length(args) == 8 ?
            parse(BigFloat, args[8]) :
            acos(BigFloat(-1))/2

        k = kr + kt
        kp = kr - kt

        R = SVector{3,BigFloat}(
            sin(theta)*cos(phi),
            sin(theta)*sin(phi),
            cos(theta)
        )

        initial_norm = norm(R)
        max_norm_error = zero(BigFloat)

        output_path = abspath(output)
        mkpath(dirname(output_path))

        println("BIGFLOAT DKT TRAJECTORY")
        println("="^65)
        println("Precision (bits): ", bits)
        println("kr: ", kr)
        println("ktheta: ", kt)
        println("k: ", k)
        println("kprime: ", kp)
        println("p: ", p)
        println("theta0: ", theta)
        println("phi0: ", phi)
        println("Trajectory length: ", nsteps)
        println("Output: ", output_path)
        println()

        open(output_path, "w") do io
            println(io, "step,X,Y,Z,norm")

            # Save initial condition.
            println(io,
                "0,",
                Float64(R[1]), ",",
                Float64(R[2]), ",",
                Float64(R[3]), ",",
                Float64(norm(R))
            )

            for step in 1:nsteps
                R = big_map(R, p, k, kp)

                current_norm = norm(R)
                max_norm_error = max(
                    max_norm_error,
                    abs(current_norm - initial_norm)
                )

                # Convert only when writing the output.
                println(io,
                    step, ",",
                    Float64(R[1]), ",",
                    Float64(R[2]), ",",
                    Float64(R[3]), ",",
                    Float64(current_norm)
                )

                if step % 10000 == 0
                    println("Completed ", step, "/", nsteps)
                end
            end
        end

        println()
        println("status=success")
        println("precision_bits=", bits)
        println("trajectory_rows=", nsteps + 1)
        println("initial_norm=", Float64(initial_norm))
        println("final_norm=", Float64(norm(R)))
        println("max_norm_error=", Float64(max_norm_error))
        println("output_csv=", output_path)
    end

    return 0
end

exit(main(ARGS))
