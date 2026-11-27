from __future__ import annotations

import math

from dkt_agent.presentation import json_safe, normalize_math_notation


def test_normalize_dkt_parameters_in_plain_text():
    text = "At k_r=1 and k_theta=0 with p=pi/2."
    rendered = normalize_math_notation(text)
    assert "$k_r$" in rendered
    assert "$k_\theta$" in rendered
    assert "$p$" in rendered
    assert "$\pi/2$" in rendered


def test_normalize_line_wrapped_and_latex_like_subscripts():
    text = "At k\nr=1 and k_\\theta=0."
    rendered = normalize_math_notation(text)
    assert "$k_r$" in rendered
    assert "$k_\theta$" in rendered


def test_preserve_existing_math_and_normalize_parenthesis_delimiters():
    text = r"Already $k_r=1$ and \(p=\pi/2\), plus \[\Delta\theta\]."
    rendered = normalize_math_notation(text)
    assert "$k_r=1$" in rendered
    assert "$p=\pi/2$" in rendered
    assert "$$\Delta\theta$$" in rendered


def test_json_safe_converts_nonfinite_values_and_nested_structures():
    value = {"k_r": 1.0, "bad": math.nan, "nested": (1, 2)}
    assert json_safe(value) == {"k_r": 1.0, "bad": None, "nested": [1, 2]}
