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


def test_research_studio_renders_plain_text_parameters_as_latex():
    from dkt_agent.research_studio import _render_scientific_text

    source = (
        "Investigate k_theta from 0 to 10000 when k_r = 1 and p = pi/2. "
        "Use theta_0 and phi_0; compare lambda_1, lambda_2 and L2."
    )
    rendered = _render_scientific_text(source)
    assert "$k_\\theta$" in rendered
    assert "$k_r$" in rendered
    assert "$\\pi/2$" in rendered
    assert "$\\theta_0$" in rendered
    assert "$\\phi_0$" in rendered
    assert "$\\lambda_1$" in rendered
    assert "$\\lambda_2$" in rendered
    assert "$L^2$" in rendered


def test_research_studio_preserves_existing_latex_and_original_question():
    from dkt_agent.research_studio import _render_scientific_text

    source = r"Compare $k_\\theta$ at fixed $k_r=1$."
    assert _render_scientific_text(source) == source
