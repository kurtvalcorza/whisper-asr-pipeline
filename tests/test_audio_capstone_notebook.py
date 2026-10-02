"""Static learner-journey properties of the generated capstone notebook (no cells executed)."""

import ast
import base64
import json
import sys
import zlib
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


def test_notebook_fits_github_contents_api_for_colab():
    data = (json.dumps(builder.build(), indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    assert len(data) < builder.GITHUB_CONTENTS_LIMIT


def test_packed_files_expand_to_exact_carried_bytes():
    namespace = {}
    _, text = _code_cells()[_index("PACKED_FILES = ")]
    before_install = text.split("import base64, zlib")[0]
    exec(before_install, namespace)
    carried = builder.carried_files()
    assert set(namespace["PACKED_FILES"]) == set(builder.PACKED)
    assert not set(builder.PACKED) & set(namespace["FILES"])
    for name, packed in namespace["PACKED_FILES"].items():
        assert zlib.decompress(base64.b64decode(packed)).decode("utf-8") == carried[name]
    assert all(namespace["FILES"][name] == carried[name] for name in namespace["FILES"])


def test_carrier_lines_stay_short_and_round_trip():
    # One ~500,000-character carrier line can make the Colab editor unresponsive.
    notebook = json.loads(builder.NOTEBOOK.read_text(encoding="utf-8"))
    assert max(len(line) for cell in notebook["cells"] for line in cell["source"]) <= 2000
    long_line = "x" * 2500 + "\n"
    value = {"empty": "", "long": long_line, "multi": "a\nb\r\n'c'\n\n" + long_line + "tail"}
    for literal in (builder.carried_literal(value), builder.carried_literal(long_line + "end")):
        assert max(len(line) for line in literal.splitlines()) <= 2000
    assert ast.literal_eval(builder.carried_literal(value)) == value
    assert ast.literal_eval(builder.carried_literal(long_line + "end")) == long_line + "end"
    assert ast.literal_eval(builder.carried_literal("")) == ""
