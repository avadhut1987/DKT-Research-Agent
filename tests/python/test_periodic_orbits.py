import math
from pathlib import Path
import pytest
from dkt_agent.periodic_orbit_tools import _number,format_periodic_orbit_result

def test_number_validation():
    assert _number("kr",1)==1.0
    for v in (float("nan"),float("inf"),"bad"):
        with pytest.raises(ValueError): _number("kr",v)

def test_format_lists_periods_and_nonexhaustive():
    r={"parameters":{"k_r":1.0,"k_theta":0.0,"p":math.pi/2,"max_period":3,"n_seeds_per_family":16},
       "search":{"unique_primitive_orbits":2,"orbits_by_primitive_period":{"1":1,"2":1}},
       "artifacts":{"directory":"results/test"}}
    s=format_periodic_orbit_result(r)
    assert "primitive period 2: 1" in s
    assert "non-exhaustive" in s.lower()

def test_script_has_primitive_and_cycle_checks():
    p=Path(__file__).parents[2]/"scripts"/"julia"/"periodic_orbit_search.jl"
    s=p.read_text()
    assert "function primitive" in s and "function same_cycle" in s
    assert "JT=" in s and "search_is_exhaustive=false" in s
