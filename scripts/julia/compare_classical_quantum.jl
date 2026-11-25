#!/usr/bin/env julia

# Kick-resolved comparison of a classical DKT orbit and quantum spin expectations.
# Uses the package's actual dense Floquet matrix and coherent-state convention.

using QuantumKickedTop
using QuantumOptics
using LinearAlgebra

const ClassicalMap = QuantumKickedTop.ClassicalMap
const ClassicalUtils = QuantumKickedTop.ClassicalUtils
const FloquetSystem = QuantumKickedTop.FloquetSystem
const QuantumUtils = QuantumKickedTop.QuantumUtils

function main(args)
    length(args) in (7, 8) || error(
        "Usage: compare_classical_quantum.jl s kr ktheta theta0 phi0 nsteps output_csv [p]"
    )

    s, kr, ktheta, theta0, phi0 = parse.(Float64, args[1:5])
    nsteps = parse(Int, args[6])
    output = args[7]
    p = length(args) == 8 ? parse(Float64, args[8]) : π / 2

    all(isfinite, (s, kr, ktheta, theta0, phi0, p)) ||
        error("All floating-point parameters must be finite.")
    s > 0 && isapprox(2s, round(2s); atol=1e-12, rtol=0) ||
        error("s must be a positive integer or half-integer.")
    nsteps >= 0 || error("nsteps must be non-negative.")
    0 <= theta0 <= π || error("theta0 must lie in [0, π].")
    -π <= phi0 <= π || error("phi0 must lie in [-π, π].")

    k = kr + ktheta
    kprime = kr - ktheta
    params = (p=p, k=k, kprime=kprime)

    # Match the package's classical spherical convention and its coherent state.
    classical = ClassicalUtils.spherical_to_cartesian(theta0, phi0)
    basis = SpinBasis(s)
    state = copy(QuantumUtils.coherentspinstate(basis, theta0, phi0).data)
    U = FloquetSystem.floquet(s, p, k, kprime) # package returns a dense matrix

    # Spin operators use the same SpinBasis ordering as coherentspinstate.
    Jx = dense(0.5 * sigmax(basis))
    Jy = dense(0.5 * sigmay(basis))
    Jz = dense(0.5 * sigmaz(basis))

    function normalized_expectation(operator, vector)
        real(dot(vector, operator.data * vector)) / s
    end

    function quantum_xyz(vector)
        return (
            normalized_expectation(Jx, vector),
            normalized_expectation(Jy, vector),
            normalized_expectation(Jz, vector),
        )
    end

    q0 = quantum_xyz(state)
    initial_alignment_error = norm(collect(q0) - collect(classical))
    initial_alignment_error <= 1e-8 || error(
        "Initial coherent-state convention does not match the classical coordinates. " *
        "Initial discrepancy = $(initial_alignment_error)."
    )

    mkpath(dirname(abspath(output)))
    max_state_norm_error = 0.0
    max_classical_norm_error = 0.0

    open(output, "w") do io
        println(io, "kick,classical_X,classical_Y,classical_Z,quantum_X,quantum_Y,quantum_Z,delta_X,delta_Y,delta_Z,classical_norm,quantum_spin_norm,state_norm")

        for kick in 0:nsteps
            qx, qy, qz = quantum_xyz(state)
            dx = classical[1] - qx
            dy = classical[2] - qy
            dz = classical[3] - qz
            classical_norm = norm(classical)
            quantum_spin_norm = sqrt(qx^2 + qy^2 + qz^2)
            state_norm = norm(state)

            all(isfinite, (qx, qy, qz, dx, dy, dz, classical_norm,
                           quantum_spin_norm, state_norm)) ||
                error("Non-finite result at kick $kick.")

            println(io, join((kick, classical[1], classical[2], classical[3],
                              qx, qy, qz, dx, dy, dz, classical_norm,
                              quantum_spin_norm, state_norm), ","))

            max_classical_norm_error = max(
                max_classical_norm_error, abs(classical_norm - 1.0)
            )
            max_state_norm_error = max(max_state_norm_error, abs(state_norm - 1.0))

            if kick < nsteps
                classical = ClassicalMap.map(classical, params)
                state = U.data * state
            end
        end
    end

    println("status=success")
    println("s=", s)
    println("hilbert_dimension=", Int(round(2s)) + 1)
    println("kr=", kr)
    println("ktheta=", ktheta)
    println("k=", k)
    println("kprime=", kprime)
    println("p=", p)
    println("theta0=", theta0)
    println("phi0=", phi0)
    println("nsteps=", nsteps)
    println("rows=", nsteps + 1)
    println("initial_alignment_error=", initial_alignment_error)
    println("max_classical_norm_error=", max_classical_norm_error)
    println("max_state_norm_error=", max_state_norm_error)
    println("output_csv=", abspath(output))
end

main(ARGS)
