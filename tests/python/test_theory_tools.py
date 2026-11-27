from __future__ import annotations

import json

import pytest

from dkt_agent import literature
from dkt_agent.symbolic_tools import symbolic_math_tool


def test_symbolic_simplification_and_derivative():
    simplified = symbolic_math_tool(
        operation="simplify",
        expressions=["(x**2 - 1)/(x - 1)"],
        variables=["x"],
    )
    assert simplified["result"]["expressions"] == ["x + 1"]

    derivative = symbolic_math_tool(
        operation="differentiate",
        expressions=["sin(x)**2"],
        variables=["x"],
    )
    assert derivative["result"]["derivative"] in {"2*sin(x)*cos(x)", "sin(2*x)"}
    assert "not a proof" in " ".join(simplified["limitations"])


def test_symbolic_jacobian():
    result = symbolic_math_tool(
        operation="jacobian",
        expressions=["x**2 + y", "x*y"],
        variables=["x", "y"],
    )
    assert result["result"]["jacobian"] == [["2*x", "1"], ["y", "x"]]


def test_symbolic_parser_rejects_arbitrary_code():
    with pytest.raises(ValueError, match="Unsupported|Unknown"):
        symbolic_math_tool(
            operation="simplify",
            expressions=["__import__('os').system('echo unsafe')"],
            variables=[],
        )


def test_literature_search_combines_and_deduplicates_sources(monkeypatch):
    monkeypatch.setattr(
        literature,
        "_search_arxiv",
        lambda query, limit: [{
            "source": "arXiv",
            "title": "A theorem about dynamics",
            "doi": "10.1234/example",
            "arxiv_id": "2601.01234",
        }],
    )
    monkeypatch.setattr(
        literature,
        "_search_crossref",
        lambda query, limit: [{
            "source": "Crossref",
            "title": "A theorem about dynamics",
            "doi": "10.1234/example",
        }, {
            "source": "Crossref",
            "title": "A different result",
            "doi": "10.1234/other",
        }],
    )
    result = literature.search_theoretical_literature("ergodic theory", 5)
    assert len(result["results"]) == 2
    assert result["sources"]["arxiv"]["status"] == "success"
    assert result["sources"]["crossref"]["status"] == "success"


def test_arxiv_paper_reader_returns_bounded_text(monkeypatch):
    monkeypatch.setattr(
        literature,
        "_get",
        lambda url: b"<html><body><h1>Paper title</h1><p>First paragraph.</p><p>Second paragraph.</p></body></html>",
    )
    result = literature.read_arxiv_paper("2601.01234", max_chars=1000)
    assert result["status"] == "success"
    assert "Paper title" in result["content"]
    assert "First paragraph." in result["content"]


def test_arxiv_reader_rejects_arbitrary_urls():
    with pytest.raises(ValueError, match="valid arXiv identifier"):
        literature.read_arxiv_paper("https://example.com/paper.pdf")
