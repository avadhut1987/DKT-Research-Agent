"""Python orchestration for the Julia arbitrary-period DKT search."""
from __future__ import annotations
import csv, json, math, os, shutil, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"julia"/"periodic_orbit_search.jl"
JULIA_ENV=ROOT/"scripts"/"julia"/"periodic_orbit_env"
RESULTS=ROOT/"results"/"diagnostics"/"periodic_orbits"

def _number(name,value):
    try: x=float(value)
    except (TypeError,ValueError) as exc: raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(x): raise ValueError(f"{name} must be finite.")
    return x

def _read(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def run_periodic_orbit_search(k_r,k_theta=0.0,p=math.pi/2,max_period=6,n_seeds=256,
                              tolerance=1e-10,max_iterations=100,seed=20261126,
                              julia_executable="julia",results_root=None):
    """Find numerically detected primitive periods 1..max_period (not exhaustive)."""
    kr=_number("k_r",k_r); kt=_number("k_theta",k_theta); rot=_number("p",p)
    tol=_number("tolerance",tolerance)
    for name,val,lo,hi in (("max_period",max_period,1,20),("n_seeds",n_seeds,16,100000),
                           ("max_iterations",max_iterations,1,10000),("seed",seed,0,2**31-1)):
        if isinstance(val,bool) or not isinstance(val,int) or not lo<=val<=hi:
            raise ValueError(f"{name} must be an integer in [{lo},{hi}].")
    if tol<=0: raise ValueError("tolerance must be positive.")
    if not shutil.which(julia_executable): raise FileNotFoundError(f"Julia not found: {julia_executable}")
    if not SCRIPT.is_file(): raise FileNotFoundError(f"Missing Julia script: {SCRIPT}")
    root=Path(results_root) if results_root else RESULTS
    root.mkdir(parents=True,exist_ok=True)
    outdir=Path(tempfile.mkdtemp(prefix="periodic_orbits_",dir=root))
    summary=outdir/"periodic_orbits.csv"
    julia_env=os.environ.copy()
    julia_env["JULIA_PKG_PRECOMPILE_AUTO"]="0"
    setup=subprocess.run(
        [julia_executable,f"--project={JULIA_ENV}","-e","using Pkg; Pkg.instantiate()"],
        cwd=ROOT,text=True,capture_output=True,check=False,timeout=1800,env=julia_env)
    (outdir/"environment_setup.txt").write_text(setup.stdout+"\n"+setup.stderr,encoding="utf-8")
    if setup.returncode:
        raise RuntimeError(f"Could not instantiate lightweight Julia environment; see {outdir/'environment_setup.txt'}")
    cmd=[julia_executable,f"--project={JULIA_ENV}",str(SCRIPT),
         str(kr),str(kt),str(rot),str(max_period),str(n_seeds),str(tol),
         str(max_iterations),str(seed),str(summary)]
    proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False,timeout=3600,env=julia_env)
    (outdir/"stdout.txt").write_text(proc.stdout,encoding="utf-8")
    (outdir/"stderr.txt").write_text(proc.stderr,encoding="utf-8")
    if proc.returncode: raise RuntimeError(f"Julia search failed ({proc.returncode}); see {outdir/'stderr.txt'}\n{proc.stderr[-3000:]}")
    points=outdir/"periodic_orbits_points.csv"
    if not summary.is_file() or not points.is_file(): raise RuntimeError("Julia did not create both CSV outputs.")
    orbits=_read(summary); point_rows=_read(points); counts={}
    for row in orbits:
        n=int(row["primitive_period"]); counts[str(n)]=counts.get(str(n),0)+1
    result={"schema_version":1,"diagnostic":"numerical primitive periodic-orbit search",
      "created_utc":datetime.now(timezone.utc).isoformat(),
      "parameters":{"k_r":kr,"k_theta":kt,"k":kr+kt,"k_prime":kr-kt,"p":rot,
       "max_period":max_period,"n_seeds_per_family":n_seeds,"seed":seed,
       "tolerance":tol,"max_iterations":max_iterations},
      "search":{"unique_primitive_orbits":len(orbits),"orbits_by_primitive_period":counts,
       "orbit_points_written":len(point_rows),"exhaustive":False,
       "caveat":"Finite multi-start Newton search; failure to detect an orbit does not show it does not exist."},
      "artifacts":{"directory":str(outdir),"summary_csv":str(summary),"points_csv":str(points),
       "stdout":str(outdir/"stdout.txt"),"stderr":str(outdir/"stderr.txt")},"orbits":orbits}
    meta=outdir/"periodic_orbit_search.json"; result["artifacts"]["metadata_json"]=str(meta)
    meta.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def compare_periodic_orbit_searches(k_r,k_theta_values=(0.0,10000.0),**kwargs):
    """Compare detected cycles across ktheta values at fixed kr and p."""
    if len(k_theta_values)<2: raise ValueError("At least two k_theta values are required.")
    return {"diagnostic":"fixed-k_r periodic-orbit comparison","k_r":float(k_r),
      "runs":[run_periodic_orbit_search(k_r,k_theta=v,**kwargs) for v in k_theta_values],
      "interpretation_note":"Compare orbit geometry at fixed k_r; k_theta is not a control of classical chaoticity, LLE, or KSE."}

def format_periodic_orbit_result(result):
    p=result["parameters"]; s=result["search"]
    lines=["Numerical DKT periodic-orbit search",
      f"k_r={p['k_r']:g}, k_theta={p['k_theta']:g}, p={p['p']:.8g}",
      f"searched primitive periods 1..{p['max_period']}",
      f"seed points: {p['n_seeds_per_family']} in each of two families",
      f"unique detected orbits: {s['unique_primitive_orbits']}"]
    for n,count in sorted(s["orbits_by_primitive_period"].items(),key=lambda x:int(x[0])):
        lines.append(f"  primitive period {n}: {count}")
    lines.extend(["Search is finite and non-exhaustive.",f"Artifacts: {result['artifacts']['directory']}"])
    return "\n".join(lines)

try:
    from langchain_core.tools import tool
    @tool
    def search_dkt_periodic_orbits(k_r:float,k_theta:float=0.0,p:float=math.pi/2,
       max_period:int=6,n_seeds:int=256,tolerance:float=1e-10,
       max_iterations:int=100,seed:int=20261126)->str:
        """Search DKT primitive periodic orbits using Julia's map and tangent map.
        Reports closure residuals and Floquet multipliers. Finite search is not exhaustive.
        Compare k_theta at fixed k_r for orbit geometry, not changed chaoticity."""
        return format_periodic_orbit_result(run_periodic_orbit_search(
          k_r,k_theta,p,max_period,n_seeds,tolerance,max_iterations,seed))
    periodic_orbit_langchain_tool=search_dkt_periodic_orbits

    @tool
    def compare_dkt_periodic_orbits(k_r:float,p:float=math.pi/2,
       max_period:int=6,n_seeds:int=256,tolerance:float=1e-10,
       max_iterations:int=100,seed:int=20261126)->str:
        """Compare detected DKT cycles at fixed kr for ktheta=0 and 10000.
        This compares orbit geometry, not a change in classical chaoticity."""
        result=compare_periodic_orbit_searches(
          k_r,(0.0,10000.0),p=p,max_period=max_period,n_seeds=n_seeds,
          tolerance=tolerance,max_iterations=max_iterations,seed=seed)
        return "\n\n".join(format_periodic_orbit_result(run) for run in result["runs"]) + "\n" + result["interpretation_note"]

    periodic_orbit_comparison_langchain_tool=compare_dkt_periodic_orbits
except ImportError:
    periodic_orbit_langchain_tool=None
    periodic_orbit_comparison_langchain_tool=None
