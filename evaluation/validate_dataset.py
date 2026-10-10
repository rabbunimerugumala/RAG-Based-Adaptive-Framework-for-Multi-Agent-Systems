"""Validate benchmark datasets and report the number of usable records.

Usage:  python evaluation/validate_dataset.py            (all datasets in data/datasets)
        python evaluation/validate_dataset.py smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_rag.evaluation.datasets import validate_dataset  # noqa: E402


def main() -> int:
    base = ROOT / "data" / "datasets"
    names = sys.argv[1:] or sorted(p.name for p in base.iterdir() if p.is_dir())
    failed = 0
    for name in names:
        rep = validate_dataset(base / name)
        state = "OK " if rep.ok else "BAD"
        print(f"[{state}] {name}: {rep.num_documents} documents, {rep.num_usable}/{rep.num_questions} usable questions "
              f"({rep.num_answerable} answerable, {rep.num_unanswerable} unanswerable)")
        for issue in rep.issues[:10]:
            print(f"      - {issue}")
        failed += not rep.ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
