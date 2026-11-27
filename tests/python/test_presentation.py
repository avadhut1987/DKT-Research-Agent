from __future__ import annotations

import math

from dkt_agent.presentation import format_json, json_safe, normalize_math_notation


def test_normalize_dkt_parameters_in_plain_text():
    text = "At k_r=1 and k_theta=0 with p=pi/2."
    rendered = normalize_math_notation(text)
    assert "$k_r$" in rendered
    assert r"$k_\theta$" in rendered
    assert "$p$" in rendered
    assert r"$\pi/2$" in rendered


def test_normalize_line_wrapped_and_latex_like_subscripts():
    text = "At k\nr=1 and k_\\theta=0."
    rendered = normalize_math_notation(text)
    assert "$k_r$" in rendered
    assert r"$k_\theta$" in rendered


def test_preserve_existing_math_and_normalize_parenthesis_delimiters():
    text = r"Already $k_r=1$ and \(p=\pi/2\), plus \[\Delta\theta\]."
    rendered = normalize_math_notation(text)
    assert "$k_r=1$" in rendered
    assert r"$p=\pi/2$" in rendered
    assert r"$$\Delta\theta$$" in rendered


def test_json_safe_converts_nonfinite_values_and_nested_structures():
    value = {"k_r": 1.0, "bad": math.nan, "nested": (1, 2)}
    assert json_safe(value) == {"k_r": 1.0, "bad": None, "nested": [1, 2]}



def test_format_json_is_complete_valid_and_strict():
    value = {"arguments": {"k_r": 1, "k_theta": 0, "observables": ["x", "y"]}}
    rendered = format_json(value)
    assert '"k_r": 1,' in rendered
    assert '"k_theta": 0,' in rendered
    assert '"observables": [' in rendered
    import json
    assert json.loads(rendered) == value


def test_format_json_serializes_nonfinite_values_as_null():
    rendered = format_json({"value": math.nan})
    assert rendered == '{\n  "value": null\n}'


def test_normalize_math_notation_removes_invisible_wrapping_characters():
    text = "At k\u200b_\u200btheta=0 and p\u00a0=\u00a0pi/2."
    rendered = normalize_math_notation(text)
    assert r"$k_\theta$" in rendered
    assert "$p$" in rendered
    assert r"$\pi/2$" in rendered
