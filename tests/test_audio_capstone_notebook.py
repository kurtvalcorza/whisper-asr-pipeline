"""Static learner-journey properties of the generated capstone notebook (no cells executed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_audio_capstone as builder  # noqa: E402


def _code_cells():
    return [
        (cell, "".join(cell["source"])) for cell in builder.build()["cells"] if cell["cell_type"] == "code"
    ]


def _index(fragment):
    matches = [i for i, (_, text) in enumerate(_code_cells()) if fragment in text]
    assert len(matches) == 1, fragment
    return matches[0]


def test_infrastructure_carrier_starts_collapsed():
    cell, text = _code_cells()[_index("FILES = ")]
    assert text.startswith("# @title Infrastructure")
    assert cell["metadata"] == {"cellView": "form", "jupyter": {"source_hidden": True}}
    collapsed = [c for c, _ in _code_cells() if c["metadata"]]
    assert collapsed == [cell], "only the embedded-source carrier is hidden"


def test_byod_is_searchable_without_rebuilding():
    search = _index("def show_search(")
    build = _index("RUN_BYOD = False")
    query = _index("BYOD_QUERY = ''")
    assert search < build < query
    build_text, query_text = _code_cells()[build][1], _code_cells()[query][1]
    assert "audio_byod.py" in build_text and "byod/latest.json" in build_text
    assert "audio_byod.py" not in query_text and "command(" not in query_text
    assert "show_search(BYOD_QUERY, BYOD['search_dir']" in query_text
    assert "'--index'" in _code_cells()[search][1] and "audio_sha256" in _code_cells()[search][1]


def test_failure_trace_shows_competing_evidence_for_both_categories():
    text = _code_cells()[_index("def candidate_table(")][1]
    assert "Candidate miss" in text and "Reranker demotion" in text
    assert "This category did not occur" in text and "automatic[row['doc_id']]" in text
    assert "not negatives" in text


def test_activity_latency_table_is_same_query_paired():
    text = _code_cells()[_index("summary['paired_latency']")][1]
    assert "paired['query_ids']" in text and "query cohorts differ" in text
