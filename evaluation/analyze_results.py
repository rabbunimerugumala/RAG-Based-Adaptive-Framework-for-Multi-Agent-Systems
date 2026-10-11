"""Re-analyse saved benchmark results from their per-question rows.

* recomputes each system's summary metrics from the rows (independent of the stored summary),
* paired comparison of two systems on the SAME questions with the exact two-sided McNemar test
  (binomial test on the discordant pairs; questions both systems got right or both got wrong are ignored),
* error categories per system.

Usage (PowerShell, project root):
    python evaluation/analyze_results.py <run_id>                          # all systems inside one run
    python evaluation/analyze_results.py <run_id> --a standard --b adaptive
    python evaluation/analyze_results.py <run_A> --other <run_B> --a standard --b adaptive
        (compares system a from run_A with system b from run_B; both runs must contain the same questions)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_rag.evaluation.metrics import aggregate  # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b and c discordant pairs (binomial with p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def outcome(r: dict) -> str:
    if r.get("error"):
        return "error"
    if r["answerable"]:
        return "correct" if r["correct"] else ("wrong_abstention" if r["abstained"] else "wrong_answer")
    return "correct_abstention" if r["abstained"] else "answered_unanswerable"


def paired(rows_a: dict, rows_b: dict, name_a: str, name_b: str) -> dict:
    common = [q for q in rows_a if q in rows_b and rows_a[q]["answerable"]]
    if len(common) != len([q for q in rows_a if rows_a[q]["answerable"]]):
        print("  ! the two systems do not share the same answerable questions; using the intersection")
    b = sum(rows_a[q]["correct"] and not rows_b[q]["correct"] for q in common)
    c = sum(rows_b[q]["correct"] and not rows_a[q]["correct"] for q in common)
    unans = [q for q in rows_a if q in rows_b and not rows_a[q]["answerable"]]
    ab_b = sum(rows_a[q]["abstained"] and not rows_b[q]["abstained"] for q in unans)
    ab_c = sum(rows_b[q]["abstained"] and not rows_a[q]["abstained"] for q in unans)
    res = {"n_answerable": len(common), f"only_{name_a}_correct": b, f"only_{name_b}_correct": c,
           "mcnemar_p_accuracy": round(mcnemar_exact(b, c), 4), "n_unanswerable": len(unans),
           f"only_{name_a}_abstained_correctly": ab_b, f"only_{name_b}_abstained_correctly": ab_c,
           "mcnemar_p_abstention": round(mcnemar_exact(ab_b, ab_c), 4)}
    return res


def load(run_id: str) -> dict:
    path = ROOT / "evaluation" / "results" / run_id / "results.json"
    if not path.exists():
        sys.exit(f"No results at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def by_method(res: dict) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in res["rows"]:
        out.setdefault(r["method"], {})[r["question_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--other", help="second run id (for cross-run comparisons on the same questions)")
    ap.add_argument("--a")
    ap.add_argument("--b")
    args = ap.parse_args()

    res = load(args.run_id)
    systems = by_method(res)
    print(f"Run {args.run_id} | dataset {res['dataset']['name']} | mode {res['mode']} | "
          f"{res['sample']['num_questions']} questions | systems {list(systems)}")
    for m, rows in systems.items():
        s = aggregate(list(rows.values()))
        stored = res["summary"].get(m, {})
        mism = [k for k in ("accuracy", "wrong_abstention_rate", "correct_abstention_rate")
                if stored.get(k) is not None and abs(stored[k] - s[k]) > 1e-4]
        print(f"\n[{m}] accuracy={s['accuracy']} wrong_abst={s['wrong_abstention_rate']} "
              f"correct_abst={s['correct_abstention_rate']} llm_calls={s['mean_llm_calls']} "
              f"latency={s['mean_latency_ms']} ms  (matches stored summary: {not mism})")
        print("   outcomes:", dict(Counter(outcome(r) for r in rows.values())))

    if args.other:
        other = by_method(load(args.other))
        if not (args.a and args.b):
            sys.exit("--a and --b are required with --other")
        pairs = [(f"{args.a}@A", systems[args.a], f"{args.b}@B", other[args.b])]
    elif args.a and args.b:
        pairs = [(args.a, systems[args.a], args.b, systems[args.b])]
    else:
        pairs = [(a, systems[a], b, systems[b]) for a, b in combinations(systems, 2)]
    for name_a, ra, name_b, rb in pairs:
        print(f"\nPaired {name_a} vs {name_b} (exact two-sided McNemar):")
        for k, v in paired(ra, rb, name_a, name_b).items():
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
