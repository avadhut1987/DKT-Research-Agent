using QuantumKickedTop
using StaticArrays
using LinearAlgebra

const CM = QuantumKickedTop.ClassicalMap
const CU = QuantumKickedTop.ClassicalUtils

function tangent_basis(R)
    reference = abs(R[3]) < 0.9 ?
        SVector(0.0, 0.0, 1.0) :
        SVector(1.0, 0.0, 0.0)

    e1 = normalize(cross(R, reference))
    e2 = normalize(cross(R, e1))

    return hcat(e1, e2)
end

function main()
    p = pi / 2

    parameter_sets = [
        (1.0, 0.0),
        (1.0, 100.0),
        (1.0, 1.0e3),
        (1.0, 1.0e4),
        (1.0, 1.0e5),
    ]

    initial_conditions = [
        (0.75, -1.0),
        (1.0, -0.75),
        (1.1, 0.3),
        (pi/4, pi/3),
        (1.4, -2.0),
    ]

    println("DKT TANGENT-PLANE AREA-PRESERVATION VALIDATION")
    println("="^65)

    for (kr, kt) in parameter_sets
        k = kr + kt
        kp = kr - kt

        max_det_error = 0.0
        max_tangent_error = 0.0

        for (theta, phi) in initial_conditions
            R = CU.spherical_to_cartesian(theta, phi)

            J = Matrix(CM.tangent_map(R, p, k, kp))
            Rnext = CM.map(R, p, k, kp)

            B = tangent_basis(R)
            Bnext = tangent_basis(Rnext)

            A = transpose(Bnext) * J * B

            det_error = abs(abs(det(A)) - 1.0)

            # Check that the propagated tangent vectors
            # are orthogonal to the next sphere normal.
            tangent_error = norm(
                transpose(Rnext) * J * B
            )

            max_det_error = max(max_det_error, det_error)
            max_tangent_error = max(
                max_tangent_error, tangent_error
            )
        end

        println()
        println("kr = ", kr, ", ktheta = ", kt)
        println("Maximum | |det(A)| - 1 | = ", max_det_error)
        println("Maximum tangent leakage    = ", max_tangent_error)
    end

    println()
    println("Validation completed.")
end

main()
