#!/usr/bin/env julia
using QuantumKickedTop

function main(args)
    length(args) in (8, 9) || error("Usage: run_entropy_heatmap.jl s kr ktheta nsteps transient rdm_dim grid_size output_csv [p]")
    s, kr, kt = parse.(Float64, args[1:3])
    nsteps, transient, Q, grid_size = parse.(Int, args[4:7])
    output = args[8]
    p = length(args) == 9 ? parse(Float64, args[9]) : π / 2
    all(isfinite, (s, kr, kt, p)) || error("Parameters must be finite")
    grid_size >= 2 || error("grid_size must be at least 2")
    θs = collect(range(0.0, π; length=grid_size))
    ϕs = collect(range(-π, π; length=grid_size + 1))[1:end-1]
    U = QuantumKickedTop.FloquetSystem.floquet(s, p, kr + kt, kr - kt)
    space = (θs=θs, ϕs=ϕs)
    result = QuantumKickedTop.HeatmapTimeAveragedEntropy.time_averaged_vn_entropy_grid(
        s, U; rdm_dim=Q, nsteps=nsteps, transient=transient, space=space
    )
    mkpath(dirname(abspath(output)))
    open(output, "w") do io
        println(io, "theta,phi,normalized_average_entropy,early_late_abs_difference,status")
        for i in eachindex(θs), j in eachindex(ϕs)
            value = result.entropy[i,j]
            diff = result.diagnostics.early_late_difference[i,j]
            status = result.diagnostics.failed_mask[i,j] ? "failed" : "success"
            println(io, θs[i], ",", ϕs[j], ",", value, ",", diff, ",", status)
        end
    end
    println("status=success")
    println("grid_points=$(grid_size^2)")
    println("successful_points=$(result.diagnostics.successful_points)")
    println("failed_points=$(result.diagnostics.failed_points)")
    println("post_transient_samples=$(result.diagnostics.post_transient_samples)")
    println("output_csv=$(abspath(output))")
end
main(ARGS)
