#!/usr/bin/env julia
using QuantumKickedTop
function main(args)
 length(args) in (8,9) || error("Usage: run_entanglement_dynamics.jl s kr ktheta theta0 phi0 nsteps rdm_dim output_csv [p]")
 s,kr,kt,θ,ϕ=parse.(Float64,args[1:5]); n,Q=parse.(Int,args[6:7]); out=args[8]; p=length(args)==9 ? parse(Float64,args[9]) : π/2
 all(isfinite,(s,kr,kt,θ,ϕ,p)) || error("Parameters must be finite")
 U=QuantumKickedTop.FloquetSystem.floquet(s,p,kr+kt,kr-kt)
 r=QuantumKickedTop.EntanglementDynamics.vn_entropy_dynamics(s,U,θ,ϕ;rdm_dim=Q,nsteps=n)
 mkpath(dirname(abspath(out)))
 open(out,"w") do io
  println(io,"kick,normalized_entropy,state_norm")
  for i in eachindex(r.kick); println(io,r.kick[i],",",r.normalized_entropy[i],",",r.state_norm[i]); end
 end
 println("status=success"); println("rows=$(length(r.kick))"); println("Smax_bits=$(r.Smax_bits)"); println("max_norm_error=$(r.max_norm_error)"); println("output_csv=$(abspath(out))")
end
main(ARGS)
