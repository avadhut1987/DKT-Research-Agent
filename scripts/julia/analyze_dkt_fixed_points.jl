#!/usr/bin/env julia

# Fixed-point search and local stability analysis for the classical DKT map.
# Uses the authoritative map and analytical tangent Jacobian from QuantumKickedTop.jl.

using QuantumKickedTop
using LinearAlgebra
using StaticArrays

const ClassicalMap = QuantumKickedTop.ClassicalMap

function usage()
    println("""
    Usage:
      julia --project=QuantumKickedTop.jl scripts/julia/analyze_dkt_fixed_points.jl \\
          kr ktheta p nseeds tolerance max_iterations output_csv

    Example:
      julia --project=QuantumKickedTop.jl scripts/julia/analyze_dkt_fixed_points.jl \\
          1.0 0.0 1.5707963267948966 200 1e-10 100 results/fixed_points.csv
    """)
end

function tangent_basis(R::SVector{3,Float64})
    reference = abs(R[3]) < 0.9 ?
        SVector(0.0, 0.0, 1.0) :
        SVector(1.0, 0.0, 0.0)

    e1 = normalize(cross(R, reference))
    e2 = normalize(cross(R, e1))

    return hcat(e1, e2)
end

function sphere_update(R, B, delta)
    angle = norm(delta)

    if angle < 1e-14
        return R
    end

    tangent_displacement = B * delta

    updated = cos(angle) * R +
        (sin(angle) / angle) * tangent_displacement

    return normalize(SVector{3,Float64}(updated))
end

function fibonacci_seeds(n::Int)
    golden_angle = pi * (3.0 - sqrt(5.0))
    seeds = SVector{3,Float64}[]

    for i in 1:n
        z = 1.0 - 2.0 * (i - 0.5) / n
        radius = sqrt(max(0.0, 1.0 - z^2))
        phi = golden_angle * i

        push!(
            seeds,
            SVector(
                radius * cos(phi),
                radius * sin(phi),
                z
            )
        )
    end

    return seeds
end

function find_fixed_point(seed, params, tolerance, max_iterations)
    R = seed
    identity3 = Matrix{Float64}(I, 3, 3)

    for iteration in 0:max_iterations
        Rnext = ClassicalMap.map(R, params)
        residual_vector = Rnext - R
        residual = norm(residual_vector)

        if residual <= tolerance
            # A small residual alone is insufficient near a marginal
            # fixed point. Also require a small tangent Newton correction.
            Bcheck = tangent_basis(R)
            Jcheck = Matrix(ClassicalMap.tangent_map(R, params))
            gcheck = transpose(Bcheck) * residual_vector
            Mcheck = transpose(Bcheck) * (Jcheck - identity3) * Bcheck

            correction = try
                -(Mcheck \ gcheck)
            catch
                fill(Inf, 2)
            end

            correction_tolerance = max(100 * eps(Float64), 10 * tolerance)

            if residual == 0.0 ||
               (all(isfinite, correction) &&
                norm(correction) <= correction_tolerance)
                return (
                    point = R,
                    residual = residual,
                    iterations = iteration
                )
            end
        end

        B = tangent_basis(R)
        J = Matrix(ClassicalMap.tangent_map(R, params))

        # Newton equation projected onto the tangent plane.
        g = transpose(B) * residual_vector
        M = transpose(B) * (J - identity3) * B

        delta = try
            -(M \ g)
        catch
            return nothing
        end

        if !all(isfinite, delta)
            return nothing
        end

        # Limit the angular correction for numerical robustness.
        delta_norm = norm(delta)

        if delta_norm > 0.5
            delta .*= 0.5 / delta_norm
        end

        # Backtracking line search on the projected residual.
        current_norm = norm(g)
        accepted = false
        scale = 1.0

        for _ in 1:14
            trial = sphere_update(R, B, scale .* delta)
            trial_residual = ClassicalMap.map(trial, params) - trial
            trial_basis = tangent_basis(trial)
            trial_g = transpose(trial_basis) * trial_residual

            if norm(trial_g) < current_norm
                R = trial
                accepted = true
                break
            end

            scale *= 0.5
        end

        if !accepted
            return nothing
        end
    end

    # Apply the same full-residual and tangent-correction criterion
    # at the iteration limit as in the main convergence check.
    final_residual_vector = ClassicalMap.map(R, params) - R
    final_residual = norm(final_residual_vector)

    if final_residual <= tolerance
        Bfinal = tangent_basis(R)
        Jfinal = Matrix(ClassicalMap.tangent_map(R, params))
        gfinal = transpose(Bfinal) * final_residual_vector
        Mfinal = transpose(Bfinal) * (Jfinal - identity3) * Bfinal

        correction = try
            -(Mfinal \ gfinal)
        catch
            fill(Inf, 2)
        end

        correction_tolerance = max(100 * eps(Float64), 10 * tolerance)

        if final_residual == 0.0 ||
           (all(isfinite, correction) &&
            norm(correction) <= correction_tolerance)
            return (
                point = R,
                residual = final_residual,
                iterations = max_iterations
            )
        end
    end

    return nothing
end

function classify_stability(
    eigenvalues,
    tolerance;
    kr=nothing,
    axial=false
)
    # Known axial fixed-point transition for p = pi/2 and ktheta = 0.
    if axial && !isnothing(kr)
        if kr < 1.0 - tolerance
            return "stable"
        elseif kr > 1.0 + tolerance
            return "unstable"
        else
            return "parabolic_partially_stable"
        end
    end

    moduli = abs.(eigenvalues)

    # Eigenvalues near +1 or -1 indicate a parabolic/marginal case,
    # not a nontrivial elliptic rotation.
    if any(abs.(eigenvalues .- 1.0) .<= tolerance) ||
       any(abs.(eigenvalues .+ 1.0) .<= tolerance)
        return "parabolic_or_marginal"
    elseif maximum(moduli) > 1.0 + tolerance &&
           minimum(moduli) < 1.0 - tolerance
        return "hyperbolic"
    elseif all(abs.(moduli .- 1.0) .<= tolerance) &&
           any(abs.(imag.(eigenvalues)) .> tolerance)
        return "elliptic"
    else
        return "other"
    end
end

function main(args)
    if length(args) != 7
        usage()
        return 1
    end

    kr = parse(Float64, args[1])
    ktheta = parse(Float64, args[2])
    p = parse(Float64, args[3])
    nseeds = parse(Int, args[4])
    tolerance = parse(Float64, args[5])
    max_iterations = parse(Int, args[6])
    output = args[7]

    all(isfinite, (kr, ktheta, p, tolerance)) ||
        error("Floating-point parameters must be finite.")

    nseeds >= 4 || error("nseeds must be at least 4.")
    tolerance > 0 || error("tolerance must be positive.")
    max_iterations > 0 || error("max_iterations must be positive.")

    params = (
        p = p,
        k = kr + ktheta,
        kprime = kr - ktheta
    )

    candidates = NamedTuple[]
    converged_seeds = 0

    # Include the two analytically known axial fixed points explicitly.
    seeds = SVector{3,Float64}[
        SVector(0.0, 1.0, 0.0),
        SVector(0.0, -1.0, 0.0)
    ]
    append!(seeds, fibonacci_seeds(nseeds))

    for seed in seeds
        result = find_fixed_point(
            seed,
            params,
            tolerance,
            max_iterations
        )

        isnothing(result) && continue
        converged_seeds += 1

        # Remove duplicate solutions on the sphere.
        duplicate = any(
            norm(result.point - candidate.point) < 1e-7
            for candidate in candidates
        )

        if !duplicate
            push!(candidates, result)
        end
    end

    output_dir = dirname(abspath(output))
    mkpath(output_dir)

    open(output, "w") do io
        println(
            io,
            "fixed_point,X,Y,Z,residual,iterations," *
            "stability,lambda1_real,lambda1_imag," *
            "lambda2_real,lambda2_imag,determinant"
        )

        identity3 = Matrix{Float64}(I, 3, 3)

        for (index, candidate) in enumerate(candidates)
            R = candidate.point
            B = tangent_basis(R)
            J = Matrix(ClassicalMap.tangent_map(R, params))

            # Local Jacobian acting within the tangent plane.
            A = transpose(B) * J * B
            eigenvalues = sort(eigvals(A); by=abs, rev=true)

            axial = (
                min(
                    norm(R - SVector(0.0, 1.0, 0.0)),
                    norm(R - SVector(0.0, -1.0, 0.0))
                ) <= 1e-8
            ) && abs(p - pi / 2) <= 1e-10 &&
                 abs(ktheta) <= 1e-10

            stability = classify_stability(
                eigenvalues,
                1e-6;
                kr=kr,
                axial=axial
            )

            println(
                io,
                index, ",",
                R[1], ",", R[2], ",", R[3], ",",
                candidate.residual, ",",
                candidate.iterations, ",",
                stability, ",",
                real(eigenvalues[1]), ",",
                imag(eigenvalues[1]), ",",
                real(eigenvalues[2]), ",",
                imag(eigenvalues[2]), ",",
                det(A)
            )
        end
    end

    println("status=success")
    println("kr=", kr)
    println("ktheta=", ktheta)
    println("p=", p)
    println("nseeds=", nseeds)
    println("converged_seeds=", converged_seeds)
    println("unique_fixed_points=", length(candidates))
    println("tolerance=", tolerance)
    println("output_csv=", abspath(output))

    return 0
end

exit(main(ARGS))
