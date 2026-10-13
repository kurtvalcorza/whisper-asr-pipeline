"""Print the lexical-overlap audit of a frozen query file against the frozen corpus references.

Usage: python tools/audit_audio_queries.py [queries.json] [--json]
Default: tools/audio_queries.json (current version); tools/audio_queries_v1.json is the
paraphrase version kept for comparison. Unit: normalised tokens longer than three characters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import audio_core as core  # noqa: E402


def audit(queries_path: Path) -> dict:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    documents = json.loads((TOOLS / "audio_sample.json").read_text(encoding="utf-8"))["records"]
    return core.query_overlap_audit(queries, documents)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--json"]
    path = Path(args[0]) if args else TOOLS / "audio_queries.json"
    result = audit(path)
    if "--json" in sys.argv:
        print(json.dumps(result, indent=2))
        return
    print(f"{path.name}: {result['queries']} queries x {result['documents']} references ({result['unit']})")
    mean, top = result["mean_anchor_overlap"], result["max_anchor_overlap"]
    print(f"  mean anchor overlap {mean:.4f}; max {top:.4f}")
    print(f"  mean overlap with other references {result['mean_other_overlap']:.4f}")
    print(
        f"  anchor strictly highest-overlap reference: {result['anchor_strictly_highest']}; "
        f"highest or tied (nonzero): {result['anchor_highest_or_tied']}"
    )


if __name__ == "__main__":
    main()
