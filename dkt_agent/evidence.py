"""Read-only retrieval of existing DKT research evidence."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()

TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".md", ".log", ".py"}
STOPWORDS = {
    "a", "an", "and", "are", "about", "does", "for", "from",
    "how", "is", "of", "on", "or", "the", "to", "what",
    "which", "with", "show", "find", "results", "data",
}

SEARCH_ALIASES = {
    "quadratic": {"x2", "y2", "z2"},
    "quadratics": {"x2", "y2", "z2"},
    "observable": {"observables"},
    "observables": {"observable"},
    "average": {"averages"},
    "averages": {"average"},
    "scrambles": {"scramble"},
    "scramble": {"scrambles"},
    "jensen": {"js"},
    "shannon": {"js"},
    "js": {"jensen", "shannon"},
    "birkhoff": {"time_average", "time_averages"},
    "ergodicity": {"ergodic"},
    "ergodic": {"ergodicity"},
}

MAX_FILES = 30
MAX_FILE_BYTES = 2_000_000
MAX_LINES_PER_FILE = 8
MAX_LINE_LENGTH = 700


def _tokens(query: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z0-9_.+-]+", query.lower())
    return list(dict.fromkeys(
        token for token in raw
        if token not in STOPWORDS and len(token) > 1
    ))


def search_research_evidence(query: str, max_files: int = 15) -> dict:
    """Search text files under results/. This function is read-only."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if isinstance(max_files, bool) or not isinstance(max_files, int):
        raise ValueError("max_files must be an integer.")

    if not 1 <= max_files <= MAX_FILES:
        raise ValueError(f"max_files must be between 1 and {MAX_FILES}.")

    terms = _tokens(query)

    expanded_terms = set(terms)
    for term in terms:
        expanded_terms.update(SEARCH_ALIASES.get(term, set()))

    if not terms:
        raise ValueError("Query contains no searchable terms.")

    if not RESULTS_ROOT.is_dir():
        return {
            "status": "no_results_directory",
            "results_root": str(RESULTS_ROOT),
            "matches": [],
        }

    candidates = []
    for path in RESULTS_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(RESULTS_ROOT)
            size = resolved.stat().st_size
        except (OSError, ValueError):
            continue
        if size <= MAX_FILE_BYTES:
            candidates.append(resolved)

    matches = []
    for path in candidates:
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        path_text = relative_path.lower()
        path_matches = [term for term in terms if term in path_text]

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        matching_lines = []
        content_terms = set()

        for line_number, line in enumerate(content.splitlines(), start=1):
            lower_line = line.lower()
            line_matches = [
                term
                for term in terms
                if any(
                    re.search(
                        rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])",
                        lower_line,
                    )
                    for candidate in {term} | SEARCH_ALIASES.get(term, set())
                )
            ]

            if not line_matches:
                continue

            content_terms.update(line_matches)

            if len(matching_lines) < MAX_LINES_PER_FILE:
                matching_lines.append({
                    "line": line_number,
                    "text": line[:MAX_LINE_LENGTH],
                    "matched_terms": line_matches,
                })


        if not path_matches and not content_terms:
            continue

        score = 3 * len(set(path_matches)) + len(content_terms)
        if path_matches and content_terms:
            score += 2

        matches.append({
            "path": relative_path,
            "score": score,
            "match_type": (
                "filename_and_content" if path_matches and content_terms
                else "filename_only" if path_matches
                else "content_only"
            ),
            "filename_matched_terms": sorted(set(path_matches)),
            "content_matched_terms": sorted(content_terms),
            "bytes": path.stat().st_size,
            "matching_lines": matching_lines,
        })

    matches.sort(key=lambda item: (-item["score"], item["path"]))
    selected = matches[:max_files]

    return {
        "status": "success",
        "query": query,
        "terms": terms,
	"expanded_terms": sorted(expanded_terms),
        "results_root": str(RESULTS_ROOT),
        "files_scanned": len(candidates),
        "files_matched": len(matches),
        "files_returned": len(selected),
        "matches": selected,
        "limitations": [
            "Only text-based files under results/ are searched.",
            "Files larger than 2 MB are skipped.",
            "Only the first eight matching lines per file are returned.",
            "Filename-only matches have no content excerpts.",
            "Keyword retrieval is not semantic retrieval.",
            "Retrieved records do not by themselves establish scientific conclusions.",
            "No experiment scripts or simulations are executed.",
        ],
    }


def read_research_csv(
    relative_path: str,
    max_rows: int = 100,
) -> dict:
    """
    Read an existing CSV under results/ and return rows as
    dictionaries keyed by the original CSV column names.

    Read-only. Does not execute scripts or modify research data.
    """
    import csv

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("A CSV path is required.")

    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise ValueError("max_rows must be an integer.")

    if not 1 <= max_rows <= 200:
        raise ValueError("max_rows must be between 1 and 200.")

    requested = Path(relative_path)

    if requested.is_absolute():
        raise ValueError("Use a path relative to the project results/ directory.")

    # Search results return paths prefixed with "results/".
    # Accept those paths as well as paths relative to results/.
    if requested.parts and requested.parts[0] == RESULTS_ROOT.name:
        requested = Path(*requested.parts[1:])

    if not requested.parts:
        raise ValueError("A CSV path inside results/ is required.")

    try:
        csv_path = (RESULTS_ROOT / requested).resolve()
        csv_path.relative_to(RESULTS_ROOT)
    except (OSError, ValueError):
        raise ValueError("Path must remain inside the results/ directory.")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files can be read by this function.")

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {relative_path}")

    if csv_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("CSV file exceeds the 2 MB size limit.")

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)

            if not reader.fieldnames:
                raise ValueError("CSV has no header row.")

            columns = reader.fieldnames
            rows = []
            reader_iter = iter(reader)

            for line_number in range(2, max_rows + 2):
                try:
                    row = next(reader_iter)
                except StopIteration:
                    break

                rows.append({
                    "line": line_number,
                    "values": dict(row),
                })

            # Check for one additional CSV record without rereading the file.
            try:
                next(reader_iter)
                truncated = True
            except StopIteration:
                truncated = False

    except (OSError, csv.Error) as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    return {
        "status": "success",
        "path": csv_path.relative_to(PROJECT_ROOT).as_posix(),
        "columns": columns,
        "rows_returned": len(rows),
        "rows": rows,
        "possibly_truncated": truncated,
        "interpretation_note": (
            "Values are returned under their original column names. "
            "Interpret each numerical field using the CSV header; "
            "do not infer column meanings from position alone."
        ),
    }
