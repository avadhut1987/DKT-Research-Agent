"""Reusable, human-readable rendering for arbitrary structured research results.

Raw JSON remains available only through an explicit advanced-view expander or download.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st


def _label(value: Any) -> str:
    label = str(value).replace("_", " ").replace(".", " · ").strip()
    label = re.sub(r"\bktheta\b|\bk theta\b|kθ", "kθ", label, flags=re.IGNORECASE)
    label = re.sub(r"\bkr\b|\bk r\b", "kᵣ", label, flags=re.IGNORECASE)
    label = re.sub(r"\btheta 0\b|\btheta0\b|θ 0", "θ₀", label, flags=re.IGNORECASE)
    label = re.sub(r"\bphi 0\b|\bphi0\b|φ 0", "φ₀", label, flags=re.IGNORECASE)
    return label[:1].upper() + label[1:] if label else label


def format_display_value(value: Any) -> str:
    """Format a scalar or short scalar sequence without exposing JSON syntax."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, (list, tuple)) and all(
        item is None or isinstance(item, (str, int, float, bool))
        for item in value
    ):
        return ", ".join(format_display_value(item) for item in value) if value else "None"
    return str(value)


def scalar_rows(mapping: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return display-ready rows for scalar fields and short scalar sequences."""
    rows: list[dict[str, str]] = []
    for key, value in mapping.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            rows.append({"Field": _label(key), "Value": format_display_value(value)})
        elif isinstance(value, (list, tuple)) and len(value) <= 12 and all(
            item is None or isinstance(item, (str, int, float, bool))
            for item in value
        ):
            rows.append({"Field": _label(key), "Value": format_display_value(value)})
    return rows


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten a list of record dictionaries into a readable table."""
    frame = pd.json_normalize(records, sep=" · ")
    frame.columns = [_label(column) for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: format_display_value(value)
            if value is None or isinstance(value, (dict, list, tuple, bool))
            else value
        )
    return frame


def render_structured_data(
    value: Any,
    *,
    title: str | None = None,
    max_items: int = 100,
    _depth: int = 0,
) -> None:
    """Render nested dictionaries/lists as tables and labeled sections, not raw JSON."""
    if title:
        st.markdown(f"**{title}**")

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError):
                st.markdown(value)
            else:
                render_structured_data(parsed, max_items=max_items, _depth=_depth)
        else:
            st.markdown(value)
        return

    if isinstance(value, Mapping):
        rows = scalar_rows(value)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        nested = [
            (key, child) for key, child in value.items()
            if not (
                child is None or isinstance(child, (str, int, float, bool))
                or (
                    isinstance(child, (list, tuple)) and len(child) <= 12
                    and all(item is None or isinstance(item, (str, int, float, bool))
                            for item in child)
                )
            )
        ]
        if not nested:
            return
        if _depth >= 5:
            st.caption(f"{len(nested)} additional nested field(s) omitted from this view.")
            return
        for key, child in nested:
            with st.container(border=True):
                st.markdown(f"**{_label(key)}**")
                render_structured_data(child, max_items=max_items, _depth=_depth + 1)
        return

    if isinstance(value, (list, tuple)):
        if not value:
            st.caption("No entries.")
            return
        # Numeric vector/matrix data (e.g. local initial clouds and principal
        # axes) should be shown as a compact table, never as dozens of
        # "Item 1", "Item 2" nested cards.
        if all(
            isinstance(row, (list, tuple)) and len(row) > 0
            and all(item is None or isinstance(item, (int, float, bool)) for item in row)
            for row in value
        ):
            width = max(len(row) for row in value)
            frame = pd.DataFrame(
                [list(row) + [None] * (width - len(row)) for row in value],
                columns=[f"Component {index + 1}" for index in range(width)],
            )
            st.dataframe(frame.head(max_items), use_container_width=True, hide_index=True)
            if len(frame) > max_items:
                st.caption(f"Showing the first {max_items:,} of {len(frame):,} vectors.")
            return
        if all(isinstance(item, Mapping) for item in value):
            frame = records_to_frame(list(value))
            st.dataframe(frame.head(max_items), use_container_width=True, hide_index=True)
            if len(frame) > max_items:
                st.caption(f"Showing the first {max_items:,} of {len(frame):,} records.")
            return
        if all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
            frame = pd.DataFrame({
                "Item": range(1, len(value) + 1),
                "Value": [format_display_value(item) for item in value],
            })
            st.dataframe(frame.head(max_items), use_container_width=True, hide_index=True)
            if len(frame) > max_items:
                st.caption(f"Showing the first {max_items:,} of {len(frame):,} entries.")
            return
        if _depth >= 5:
            st.caption(f"{len(value):,} nested entries.")
            return
        for index, item in enumerate(value[:max_items], start=1):
            with st.container(border=True):
                st.markdown(f"**Item {index}**")
                render_structured_data(item, max_items=max_items, _depth=_depth + 1)
        if len(value) > max_items:
            st.caption(f"Showing the first {max_items:,} of {len(value):,} entries.")
        return

    st.write(format_display_value(value))


def render_raw_json_expander(value: Any, *, label: str = "Raw JSON") -> None:
    """Keep the exact machine-readable representation available on demand."""
    with st.expander(label, expanded=False):
        st.code(
            json.dumps(value, indent=2, ensure_ascii=False, default=str),
            language="json",
        )
