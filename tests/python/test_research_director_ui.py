from pathlib import Path


def test_research_director_hides_raw_json_and_question_download():
    page = Path(__file__).resolve().parents[2] / "pages" / "1_Research_Director.py"
    source = page.read_text(encoding="utf-8")
    assert "Save question" in source
    assert "Download question (.txt)" not in source
    assert "st.json(" not in source
    assert "_render_json" not in source
    assert "Optional: export research record" in source
    assert "Download technical archive" in source
