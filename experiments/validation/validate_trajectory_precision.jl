using QuantumKickedTop
using StaticArrays
using LinearAlgebra

const CM = QuantumKickedTop.ClassicalMap

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

function main()
    setprecision(BigFloat, 256) do

        kr = 1.0
        theta_str = "1.0"
        phi_str = "-0.75"

        nsteps = 10000
        checkpoints = [100, 1000, 5000, 10000]

        parameter_values = [0.0, 1000.0, 10000.0, 100000.0]

        println("FLOAT64 vs 256-BIT TRAJECTORY COMPARISON")
        println("="^75)
        println("Initial condition: theta = ", theta_str,
                ", phi = ", phi_str)
        println("Number of kicks: ", nsteps)

        for kt in parameter_values
            k = kr + kt
            kp = kr - kt

            # Float64 trajectory using the package implementation.
            theta64 = parse(Float64, theta_str)
            phi64 = parse(Float64, phi_str)

            R64 = SVector{3,Float64}(
                sin(theta64)*cos(phi64),
                sin(theta64)*sin(phi64),
                cos(theta64)
            )

            # Independent 256-bit reference trajectory.
            theta_big = parse(BigFloat, theta_str)
            phi_big = parse(BigFloat, phi_str)

            Rbig = SVector{3,BigFloat}(
                sin(theta_big)*cos(phi_big),
                sin(theta_big)*sin(phi_big),
                cos(theta_big)
            )

            kt_big = BigFloat(string(kt))
            kr_big = BigFloat(string(kr))
            k_big = kr_big + kt_big
            kp_big = kr_big - kt_big
            p_big = acos(big(-1))/2

            p64 = pi/2

            max_norm_error64 = 0.0
            max_norm_error_big = zero(BigFloat)

            println()
            println("-"^75)
            println("ktheta = ", kt)
            println("step | trajectory separation | Float64 norm error")

            for step in 1:nsteps
                R64 = CM.map(R64, p64, k, kp)
                Rbig = big_map(Rbig, p_big, k_big, kp_big)

                max_norm_error64 = max(
                    max_norm_error64,
                    abs(norm(R64) - 1.0)
                )

                max_norm_error_big = max(
                    max_norm_error_big,
                    abs(norm(Rbig) - one(BigFloat))
                )

                if step in checkpoints
                    separation = norm(
                        BigFloat.(R64) - Rbig
                    )

                    println(
                        step, " | ",
                        Float64(separation), " | ",
                        max_norm_error64
                    )
                end
            end

            println("Maximum 256-bit norm error = ",
                    Float64(max_norm_error_big))
        end
    end
end

main()
