using QuantumKickedTop
using StaticArrays
using LinearAlgebra

const CM = QuantumKickedTop.ClassicalMap
const CU = QuantumKickedTop.ClassicalUtils

function finite_difference_jacobian(R, p, k, kp; h=1e-7)
    J = zeros(3, 3)

    for j in 1:3
        delta = zeros(3)
        delta[j] = h

        Rp = R + SVector{3,Float64}(delta)
        Rm = R - SVector{3,Float64}(delta)

        fp = CM.map(Rp, p, k, kp)
        fm = CM.map(Rm, p, k, kp)

        J[:, j] = (fp - fm) / (2h)
    end

    return J
end

function main()
    println("=================================================================")
    println("DKT PACKAGE JACOBIAN VALIDATION")
    println("=================================================================")

    p = pi / 2

    parameter_sets = [
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 10.0),
        (1.0, 100.0),
        (3.0, 0.0)
    ]

    initial_conditions = [
        (0.75, -1.0),
        (1.0, -0.75),
        (1.1, 0.3),
        (pi/4, pi/3)
    ]

    total = 0
    passed = 0
    failed = 0
    max_abs_error = 0.0
    max_rel_error = 0.0

    for (kr, kt) in parameter_sets
        k = kr + kt
        kp = kr - kt

        for (theta, phi) in initial_conditions
            total += 1

            R = CU.spherical_to_cartesian(theta, phi)

            J_analytic = Matrix(CM.tangent_map(R, p, k, kp))
            J_numeric = finite_difference_jacobian(R, p, k, kp)

            abs_error = maximum(abs.(J_analytic - J_numeric))
            rel_error = norm(J_analytic - J_numeric) /
                        max(norm(J_numeric), eps(Float64))

            max_abs_error = max(max_abs_error, abs_error)
            max_rel_error = max(max_rel_error, rel_error)

            if abs_error < 1e-4 || rel_error < 1e-6
                passed += 1
            else
                failed += 1
                println("FAILED: kr=$kr, ktheta=$kt, IC=($theta,$phi)")
                println("  Absolute error = $abs_error")
                println("  Relative error = $rel_error")
            end
        end
    end

    println()
    println("Total tests: ", total)
    println("Passed:      ", passed)
    println("Failed:      ", failed)
    println()
    println("Maximum absolute error: ", max_abs_error)
    println("Maximum relative error: ", max_rel_error)

    if failed == 0
        println()
        println("RESULT: PACKAGE JACOBIAN VALIDATED")
    else
        error("PACKAGE JACOBIAN VALIDATION FAILED")
    end
end

main()
