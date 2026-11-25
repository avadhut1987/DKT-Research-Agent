#!/usr/bin/env julia

using QuantumKickedTop
using Printf

const ClassicalUtils = QuantumKickedTop.ClassicalUtils

function run_analysis(args)
    length(args) == 8 || error("Usage: dkt_lyapunov_ks.jl kr ktheta theta0 phi0 p nsteps transient output_csv")
    kr, ktheta, theta0, phi0, p = parse.(Float64, args[1:5])
    nsteps, transient = parse.(Int, args[6:7])
    output = args[8]
    all(isfinite, (kr, ktheta, theta0, phi0, p)) || error("Parameters must be finite")
    100 <= nsteps <= 500000 || error("nsteps must be between 100 and 500000")
    0 <= transient <= nsteps - 100 || error("transient must leave at least 100 analyzed kicks")

    space = (theta=theta0, phi=phi0)
    params = (p=p, k=kr + ktheta, kprime=kr - ktheta)
    n_analyzed = nsteps - transient
    fractions = (0.25, 0.50, 0.75, 1.0)
    checkpoints = sort!(unique!(max.(1, round.(Int, n_analyzed .* collect(fractions)))))

    mkpath(dirname(abspath(output)))
    open(output, "w") do io
        println(io, "analyzed_steps,lambda1,lambda2,sum_exponents,ks_pesin_estimate")
        for count in checkpoints
            total_steps = transient + count
            exponents = ClassicalUtils.get_lyapunov_spectrum(
                space, total_steps, params; burnin=transient
            )
            lambda1, lambda2 = exponents
            ks = sum(max.(exponents, 0.0))
            @printf(io, "%d,%.16e,%.16e,%.16e,%.16e\n",
                    count, lambda1, lambda2, lambda1 + lambda2, ks)
        end
    end
    println("status=success")
    println("kr=", kr)
    println("ktheta=", ktheta)
    println("nsteps=", nsteps)
    println("transient=", transient)
    println("checkpoints=", join(checkpoints, ","))
    println("output_csv=", abspath(output))
    return 0
end

exit(run_analysis(ARGS))
