using QuantumKickedTop
using LinearAlgebra

const CM = QuantumKickedTop.ClassicalMap
const CU = QuantumKickedTop.ClassicalUtils

println("="^60)
println("DKT CLASSICAL MAP VALIDATION")
println("="^60)

p = pi / 2

# Different parameter combinations
parameter_sets = [
    (kr=1.0, ktheta=0.0),
    (kr=1.0, ktheta=0.5),
    (kr=1.0, ktheta=10.0),
    (kr=1.0, ktheta=100.0),
    (kr=3.0, ktheta=0.0)
]

# Different initial conditions
initial_conditions = [
    (0.75, -1.0),
    (1.0, -0.75),
    (1.1, 0.3),
    (pi/4, pi/3)
]

max_norm_error = 0.0
tests_passed = 0
total_tests = 0

for params in parameter_sets

    kr = params.kr
    ktheta = params.ktheta

    k = kr + ktheta
    kp = kr - ktheta

    for (theta, phi) in initial_conditions

        R = CU.spherical_to_cartesian(theta, phi)

        Rnext = CM.map(R, p, k, kp)

        norm_error = abs(norm(Rnext) - 1.0)

        global max_norm_error = max(max_norm_error, norm_error)
        global total_tests += 1

        if norm_error < 1e-12
            global tests_passed += 1
        else
            println("FAILED: kr=$kr, ktheta=$ktheta")
        end
    end
end

println("\nTotal tests: ", total_tests)
println("Passed:      ", tests_passed)
println("Failed:      ", total_tests - tests_passed)
println("Maximum norm error: ", max_norm_error)

if tests_passed == total_tests
    println("\nRESULT: CLASSICAL MAP VALIDATED")
else
    error("Classical map validation failed.")
end
