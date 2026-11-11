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

function run_trajectory(bits, kt_string, nsteps, checkpoints)
    return setprecision(BigFloat, bits) do
        kr = parse(BigFloat, "1.0")
        kt = parse(BigFloat, kt_string)

        k = kr + kt
        kp = kr - kt
        p = acos(big(-1))/2

        theta = parse(BigFloat, "1.0")
        phi = parse(BigFloat, "-0.75")

        R = SVector{3,BigFloat}(
            sin(theta)*cos(phi),
            sin(theta)*sin(phi),
            cos(theta)
        )

        saved = Dict{Int,Vector{BigFloat}}()

        for step in 1:nsteps
            R = big_map(R, p, k, kp)

            if step in checkpoints
                saved[step] = collect(R)
            end
        end

        return saved
    end
end

function main()
    nsteps = 100000
    checkpoints = [1000, 10000, 50000, 100000]
    parameter_values = ["1000", "10000", "100000"]

    println("256-BIT vs 512-BIT TRAJECTORY CONVERGENCE")
    println("="^75)
    println("Trajectory length: ", nsteps)

    for kt in parameter_values
        println()
        println("-"^75)
        println("ktheta = ", kt)

        traj256 = run_trajectory(256, kt, nsteps, checkpoints)
        traj512 = run_trajectory(512, kt, nsteps, checkpoints)

        setprecision(BigFloat, 512) do
            println("step | 256-512 separation")

            for step in checkpoints
                separation = norm(
                    traj256[step] - traj512[step]
                )

                println(step, " | ", Float64(separation))
            end
        end
    end

    println()
    println("100,000-kick convergence test completed.")
end

main()
