"""Read-only scholarly discovery and open-paper inspection.

Searches Crossref and arXiv metadata. It does not claim that search ranking
establishes authority, correctness, peer review, or relevance.
"""
from __future__ import annotations

from html.parser import HTMLParser
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


USER_AGENT = "DKT-Research-Agent/0.1 (scholarly metadata retrieval)"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _get(url: str, *, timeout: int = 18) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml, application/json, text/html"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _plain_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _search_arxiv(query: str, limit: int) -> list[dict]:
    params = urlencode({
        "search_query": "all:" + query,
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    root = ET.fromstring(_get("https://export.arxiv.org/api/query?" + params))
    results = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        identifier = (entry.findtext("atom:id", default="", namespaces=ARXIV_NS)).strip()
        arxiv_id = identifier.rsplit("/", 1)[-1]
        title = _plain_text(entry.findtext("atom:title", default="", namespaces=ARXIV_NS))
        abstract = _plain_text(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS))
        authors = [
            _plain_text(node.findtext("atom:name", default="", namespaces=ARXIV_NS))
            for node in entry.findall("atom:author", ARXIV_NS)
        ]
        published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
        doi = entry.findtext("arxiv:doi", default="", namespaces={
            **ARXIV_NS, "arxiv": "http://arxiv.org/schemas/atom"
        })
        results.append({
            "source": "arXiv",
            "title": title,
            "authors": authors,
            "year": published[:4] if published else None,
            "published": published,
            "abstract": abstract,
            "arxiv_id": arxiv_id,
            "doi": doi or None,
            "url": identifier,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "peer_review_status": "Preprint record; peer-review status must be checked separately.",
        })
    return results


def _search_crossref(query: str, limit: int) -> list[dict]:
    params = urlencode({
        "query.bibliographic": query,
        "rows": limit,
        "select": "DOI,title,author,published,created,container-title,type,URL,abstract,score",
    })
    root = __import__("json").loads(_get("https://api.crossref.org/works?" + params).decode("utf-8"))
    results = []
    for item in root.get("message", {}).get("items", []):
        titles = item.get("title") or []
        title = _plain_text(titles[0] if titles else "")
        if not title:
            continue
        authors = []
        for author in item.get("author", []):
            name = " ".join(part for part in (author.get("given"), author.get("family")) if part)
            if name:
                authors.append(name)
        date_parts = (
            item.get("published", {}).get("date-parts")
            or item.get("created", {}).get("date-parts")
            or [[]]
        )
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        doi = item.get("DOI")
        results.append({
            "source": "Crossref",
            "title": title,
            "authors": authors,
            "year": year,
            "journal_or_book": (item.get("container-title") or [None])[0],
            "type": item.get("type"),
            "abstract": _plain_text(item.get("abstract")),
            "doi": doi,
            "url": item.get("URL") or (f"https://doi.org/{doi}" if doi else None),
            "peer_review_status": "Crossref metadata does not itself verify peer review.",
        })
    return results


def search_theoretical_literature(query: str, max_results: int = 8) -> dict:
    """Search arXiv and Crossref for theoretical-physics/mathematics literature."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty literature query is required.")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 12:
        raise ValueError("max_results must be an integer between 1 and 12.")

    sources = {}
    results = []
    for name, searcher in (("arxiv", _search_arxiv), ("crossref", _search_crossref)):
        try:
            found = searcher(query.strip(), max_results)
            sources[name] = {"status": "success", "results": len(found)}
            results.extend(found)
        except (HTTPError, URLError, TimeoutError, ValueError, ET.ParseError, KeyError) as exc:
            sources[name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:
            sources[name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    seen = set()
    unique = []
    for item in results:
        key = (item.get("doi") or item.get("arxiv_id") or item["title"]).lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return {
        "query": query,
        "sources": sources,
        "results": unique,
        "limitations": [
            "Search metadata and ranking are discovery aids, not evidence of correctness or authority.",
            "Crossref records may lack abstracts and do not establish peer-review status.",
            "arXiv entries are preprints unless publication status is independently verified.",
            "Read and verify primary sources before relying on a theorem, formula, or claim.",
        ],
    }


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "svg", "nav"}:
            self.skip_depth += 1
        if tag in {"p", "h1", "h2", "h3", "h4", "li"} and not self.skip_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "svg", "nav"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "h1", "h2", "h3", "h4", "li"} and not self.skip_depth:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            cleaned = re.sub(r"\s+", " ", data).strip()
            if cleaned:
                self.parts.append(cleaned)


def read_arxiv_paper(arxiv_id: str, max_chars: int = 24000, start_char: int = 0) -> dict:
    """Read available ar5iv HTML text for one arXiv paper in bounded chunks."""
    if not isinstance(arxiv_id, str) or not re.fullmatch(r"(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", arxiv_id.strip()):
        raise ValueError("Provide a valid arXiv identifier, not an arbitrary URL.")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1000 <= max_chars <= 40000:
        raise ValueError("max_chars must be between 1000 and 40000.")
    if isinstance(start_char, bool) or not isinstance(start_char, int) or start_char < 0:
        raise ValueError("start_char must be a non-negative integer.")

    url = "https://ar5iv.labs.arxiv.org/html/" + arxiv_id.strip()
    try:
        html = _get(url).decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        return {
            "status": "unavailable",
            "arxiv_id": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "message": f"HTML full text is unavailable from ar5iv: {type(exc).__name__}. Use the linked abstract/PDF page for manual inspection.",
        }

    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"\n{3,}", "\n\n", " ".join(parser.parts)).strip()
    end_char = min(start_char + max_chars, len(text))
    return {
        "status": "success",
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "full_text_url": url,
        "characters_total": len(text),
        "start_char": start_char,
        "end_char": end_char,
        "truncated": end_char < len(text),
        "next_start_char": end_char if end_char < len(text) else None,
        "content": text[start_char:end_char],
        "limitations": [
            "HTML conversion may omit or simplify equations, figures, tables, and references.",
            "Check the original paper/PDF before quoting equations or relying on technical details.",
        ],
    }
