"""Presentation helpers shared by the DKT Research Agent UI."""
from __future__ import annotations

import math
import re
from typing import Any


def normalize_math_notation(value: Any) -> str:
    """Convert common DKT plain-text symbols to Streamlit-compatible LaTeX.

    Existing $...$, $$...$$, \( ... \), and \[ ... \] math is protected from
    substitutions. This is a display-only transformation; stored research text
    and machine-readable JSON are not modified.
    """
    if not isinstance(value, str):
        return str(value)

    protected: list[str] = []

    def hold(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00MATH{len(protected) - 1}\x00"

    text = re.sub(
        r"\$\$.*?\$\$|\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\)",
        hold,
        value,
        flags=re.S,
    )

    # Normalize common DKT parameters, including line-wrapped subscripts,
    # Unicode subscripts, and LaTeX-like text emitted without math delimiters.
    patterns = (
        (r"(?<![\w])k\s*(?:_\s*\{?\s*|[\s-]+)(?:\\)?theta\s*\}?(?![\w])", r"$k_\theta$"),
        (r"(?<![\w])k\s*(?:_\s*\{?\s*|[\s-]+)(?:\\)?r\s*\}?(?![\w])", r"$k_r$"),
        (r"(?<![\w])k[ᵣ](?![\w])", r"$k_r$"),
        (r"(?<![\w])k[θ](?![\w])", r"$k_\theta$"),
        (r"(?<![\w])θ(?![\w])", r"$\theta$"),
        (r"(?<![\w])φ(?![\w])", r"$\phi$"),
        (r"(?<![\w])π(?![\w])", r"$\pi$"),
        (r"(?<![\w])theta[_\s-]*0(?![\w])", r"$\theta_0$"),
        (r"(?<![\w])phi[_\s-]*0(?![\w])", r"$\phi_0$"),
        (r"(?<![\w])Delta[_\s-]*theta(?![\w])", r"$\Delta\theta$"),
        (r"(?<![\w])Delta[_\s-]*phi(?![\w])", r"$\Delta\phi$"),
        (r"(?<![\w])pi\s*/\s*2(?![\w])", r"$\pi/2$"),
        (r"(?<![\w])pi(?![\w])", r"$\pi$"),
        (r"(?<![\w])theta(?![\w])", r"$\theta$"),
        (r"(?<![\w])phi(?![\w])", r"$\phi$"),
        (r"(?<![\w])lambda(?![\w])", r"$\lambda$"),
        (r"(?<![\w])tau(?![\w])", r"$\tau$"),
        (r"(?<![\w])epsilon(?![\w])", r"$\epsilon$"),
        (r"(?<![\w])alpha(?![\w])", r"$\alpha$"),
        (r"(?<![\w])beta(?![\w])", r"$\beta$"),
        (r"(?<![\w])k(?=\s*(?:=|′|'|,|;|\)|$))", r"$k$"),
        (r"(?<![\w])p(?=\s*(?:=|:|∈|≤|≥|,|;|\)|$))", r"$p$"),
        (r"(?<![\w])p(?=\s+(?:is|denotes|remains|at|defaults|takes))", r"$p$"),
    )
    generated: list[str] = []

    def hold_generated(replacement: str) -> str:
        generated.append(replacement)
        return f"\x00GENERATED{len(generated) - 1}\x00"

    for pattern, replacement in patterns:
        text = re.sub(
            pattern,
            lambda _m, rep=replacement: hold_generated(rep),
            text,
            flags=re.I,
        )

    for index, original in enumerate(protected):
        if original.startswith(r"\(") and original.endswith(r"\)"):
            original = "$" + original[2:-2] + "$"
        elif original.startswith(r"\[") and original.endswith(r"\]"):
            original = "$$" + original[2:-2] + "$$"
        text = text.replace(f"\x00MATH{index}\x00", original)

    for index, replacement in enumerate(generated):
        text = text.replace(f"\x00GENERATED{index}\x00", replacement)

    return text


def json_safe(value: Any) -> Any:
    """Return values accepted by strict JSON serializers and Streamlit st.json."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if hasattr(value, "dict") and callable(value.dict):
        return json_safe(value.dict())
    return str(value)
