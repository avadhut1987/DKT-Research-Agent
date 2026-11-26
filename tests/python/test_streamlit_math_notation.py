"""Regression checks for mathematical notation in the Streamlit interface."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_FILES = (
    "app.py",
    "dkt_agent/research_studio.py",
    "dkt_agent/mean_ergodic_dashboard.py",
    "dkt_agent/periodic_orbit_dashboard.py",
    "dkt_agent/streamlit_mixing_dashboard.py",
)


def test_map_parameters_are_rendered_with_latex_notation():
    source = "\n".join(
        (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in UI_FILES
    )
    assert r"$k_r$" in source
    assert r"$k_\theta$" in source


def test_streamlit_ui_avoids_unformatted_parameter_and_diagnostic_glyphs():
    source = "\n".join(
        (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in UI_FILES
    )
    for unformatted in ("kᵣ", "kθ", "λ₁", "λ₂", "θ–φ", "n_φ", "²"):
        assert unformatted not in source
