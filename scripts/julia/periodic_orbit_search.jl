#!/usr/bin/env julia
using LinearAlgebra, StaticArrays, Random

# Include the exact ClassicalMap source shipped by the QuantumKickedTop.jl
# submodule. This keeps the solver authoritative while avoiding loading the
# unrelated quantum/plotting stack for a classical periodic-orbit calculation.
const CLASSICAL_MAP_SOURCE = normpath(joinpath(@__DIR__, "..", "..",
    "QuantumKickedTop.jl", "src", "classical_dynamics", "ClassicalMap.jl"))
isfile(CLASSICAL_MAP_SOURCE) || error("Missing authoritative ClassicalMap source: " * CLASSICAL_MAP_SOURCE)
include(CLASSICAL_MAP_SOURCE)
const CM = ClassicalMap

function basis(R::SVector{3,Float64})
    ref = abs(R[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(1.0,0.0,0.0)
    e1 = normalize(cross(R,ref)); e2 = normalize(cross(R,e1))
    hcat(e1,e2)
end
function sphere_update(R,B,d)
    a=norm(d); a<1e-15 && return R
    normalize(SVector{3,Float64}(cos(a)*R+(sin(a)/a)*(B*d)))
end
function fibonacci_seeds(n)
    ga=pi*(3-sqrt(5.0)); out=SVector{3,Float64}[]
    for i in 1:n
        z=1-2*(i-0.5)/n; r=sqrt(max(0.0,1-z^2)); ph=ga*i
        push!(out,SVector(r*cos(ph),r*sin(ph),z))
    end
    out
end
function orbit_data(R,n,params)
    pts=SVector{3,Float64}[]; JT=Matrix{Float64}(I,3,3); x=R
    for _ in 1:n
        push!(pts,x); JT=Matrix(CM.tangent_map(x,params))*JT
        x=CM.map(x,params)
    end
    pts,x,JT
end
function iterate_endpoint(R,n,params)
    x=R
    for _ in 1:n
        x=CM.map(x,params)
    end
    x
end
function solve_seed(seed,n,params,tol,maxit)
    R=seed; I3=Matrix{Float64}(I,3,3)
    for it in 0:maxit
        pts,endpt,JT=orbit_data(R,n,params); res=endpt-R
        norm(res)<=tol && return (point=R,points=pts,residual=norm(res),iterations=it)
        B=basis(R); g=B'*res; M=B'*(JT-I3)*B
        d = try
            -(M \ g)
        catch
            return nothing
        end
        all(isfinite,d) || return nothing
        nd=norm(d); nd>0.35 && (d .*= 0.35/nd)
        accepted=false; scale=1.0
        for _ in 1:18
            trial=sphere_update(R,B,scale*d)
            trial_end=iterate_endpoint(trial,n,params)
            if norm(basis(trial)'*(trial_end-trial))<norm(g)
                R=trial; accepted=true; break
            end
            scale*=0.5
        end
        accepted || return nothing
    end
    nothing
end
divisors(n)=[d for d in 1:n-1 if n%d==0]
function primitive(R,n,params,tol)
    for d in divisors(n)
        x=iterate_endpoint(R,d,params)
        norm(x-R)<=max(10tol,1e-12) && return false
    end
    true
end
function same_cycle(a,b,tol)
    length(a)==length(b) || return false
    n=length(a)
    for s in 0:n-1
        maximum(norm(a[i]-b[mod1(i+s,n)]) for i in 1:n)<=tol && return true
    end
    false
end
function stability(item,n,params)
    _,endpoint,JT=orbit_data(item.point,n,params)
    A=basis(item.point)'*JT*basis(item.point); ev=eigvals(A); sort!(ev;by=abs,rev=true); m=abs.(ev)
    detA=det(A); t=1e-6
    label=if !isfinite(detA) || abs(detA-1.0)>1e-5
        "unresolved_area_preservation_error"
    elseif any(abs(abs(v)-1)<=t && abs(imag(v))<=t for v in ev)
        "parabolic_or_marginal"
    elseif maximum(m)>1+t && minimum(m)<1-t
        "hyperbolic"
    elseif all(abs.(m.-1).<=t) && any(abs.(imag.(ev)).>t)
        "elliptic"
    elseif all(abs.(m.-1).<=t)
        "marginal_or_unresolved"
    else
        "unresolved"
    end
    (ev=ev,det=detA,closure=norm(endpoint-item.point),label=label)
end
csvval(x)=occursin(r"[,\"\n\r]",string(x)) ? "\""*replace(string(x),"\""=>"\"\"")*"\"" : string(x)

function main(args)
    length(args)==9 || error("Usage: julia --project=QuantumKickedTop.jl scripts/julia/periodic_orbit_search.jl kr ktheta p max_period nseeds tolerance max_iterations seed output.csv")
    kr,kt,p=parse.(Float64,args[1:3]); nmax,nseeds=parse.(Int,args[4:5])
    tol=parse(Float64,args[6]); maxit,seed=parse.(Int,args[7:8]); output=args[9]
    all(isfinite,(kr,kt,p,tol)) || error("Inputs must be finite.")
    1<=nmax<=20 || error("max_period must be 1..20")
    nseeds>=16 || error("nseeds must be >=16")
    tol>0 && maxit>0 || error("tolerance and max_iterations must be positive")
    params=(p=p,k=kr+kt,kprime=kr-kt)
    rng=MersenneTwister(seed)
    seeds=fibonacci_seeds(nseeds)
    append!(seeds,[SVector(0.0,1.0,0.0),SVector(0.0,-1.0,0.0)])
    append!(seeds,[SVector{3,Float64}(normalize(randn(rng,3))) for _ in 1:nseeds])
    found=NamedTuple[]
    for n in 1:nmax
        conv=0; nonprim=0
        for s in seeds
            q=solve_seed(s,n,params,tol,maxit); isnothing(q) && continue; conv+=1
            if !primitive(q.point,n,params,tol); nonprim+=1; continue; end
            any(x->x.period==n && same_cycle(x.result.points,q.points,max(10tol,1e-12)),found) ||
                push!(found,(period=n,result=q))
        end
        println("period=$n converged_seeds=$conv rejected_nonprimitive=$nonprim unique=$(count(x->x.period==n,found))")
    end
    mkpath(dirname(abspath(output))); pointfile=splitext(output)[1]*"_points.csv"
    open(output,"w") do io
        println(io,"orbit,primitive_period,X0,Y0,Z0,residual,iterations,closure_error,lambda1_real,lambda1_imag,lambda2_real,lambda2_imag,determinant,stability")
        for (i,x) in enumerate(found)
            st=stability(x.result,x.period,params); R=x.result.point; e=st.ev
            row=(i,x.period,R...,x.result.residual,x.result.iterations,st.closure,
                 real(e[1]),imag(e[1]),real(e[2]),imag(e[2]),st.det,st.label)
            println(io,join(csvval.(row),","))
        end
    end
    open(pointfile,"w") do io
        println(io,"orbit,primitive_period,point_index,X,Y,Z")
        for (i,x) in enumerate(found), (j,R) in enumerate(x.result.points)
            println(io,join((i,x.period,j,R[1],R[2],R[3]),","))
        end
    end
    println("status=success"); println("unique_orbits=$(length(found))")
    println("summary_csv=$(abspath(output))"); println("points_csv=$(abspath(pointfile))")
    println("search_is_exhaustive=false")
end
if abspath(PROGRAM_FILE)==abspath(String(@__FILE__)); main(ARGS); end
