using QuantumKickedTop
using LinearAlgebra

const CU = QuantumKickedTop.ClassicalUtils

function main()
    println("=================================================================")
    println("DKT PACKAGE QR-BASED LYAPUNOV VALIDATION")
    println("=================================================================")

    p = pi / 2
    kr = 1.0
    ktheta = 0.0

    k = kr + ktheta
    kp = kr - ktheta

    params = (p=p, k=k, kprime=kp)
    space = (theta=0.75, phi=-1.0)

    burnin = 5000
    step_counts = [20000, 100000, 500000]

    println()
    println("Parameters:")
    println("kr      = ", kr)
    println("ktheta  = ", ktheta)
    println("p       = ", p)
    println("Initial condition = ", space)
    println("Burn-in = ", burnin)

    println()
    println("Convergence results:")

    results = []

    for n in step_counts
        spectrum = CU.get_lyapunov_spectrum(
            space, n, params; burnin=burnin
        )

        lambda1, lambda2 = spectrum
        exponent_sum = sum(spectrum)

        push!(results, (n, lambda1, lambda2, exponent_sum))

        println(
            "Steps = ", n,
            " | LLE = ", lambda1,
            " | lambda_2 = ", lambda2,
            " | Sum = ", exponent_sum
        )
    end

    println()
    println("Consistency checks:")

    max_sum = maximum(abs(result[4]) for result in results)
    println("Maximum absolute exponent sum = ", max_sum)

    # For this parameter point and initial condition, the
    # previously observed finite-time LLE decreases with time.
    lle_decreases = all(
        results[i+1][2] <= results[i][2] + 1e-10
        for i in 1:length(results)-1
    )

    println("LLE decreases with trajectory length: ", lle_decreases)

    # The area-preserving map should have a near-zero sum
    # of its two tangent-space Lyapunov exponents.
    sum_valid = max_sum < 1e-8
    println("Near-zero exponent sum: ", sum_valid)

    if !sum_valid
        error("Lyapunov exponent sum validation failed")
    end

    println()
    println("RESULT: PACKAGE LYAPUNOV VALIDATION COMPLETED")
end

main()
