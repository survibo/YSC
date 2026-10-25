from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.rules import sample_rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", type=int, default=100)
    ap.add_argument("--keep-trivial", action="store_true", help="상수 출력·유효 깊이 0 규칙도 남김")
    ap.add_argument("--keep-pointwise", action="store_true", help="원소별 표 찾기 규칙도 남김")
    ap.add_argument("--out", default="rules/layer2.json")
    a = ap.parse_args()
    rules = sample_rules(a.count, a.n, a.m, a.min_steps, a.max_steps, a.seed, a.probe, a.keep_trivial,
                         a.keep_pointwise)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
    cols = ["spec", "sym_kind", "inv_size", "eq_size", "permuting_size", "dep_mean", "dep_max",
            "effective_depth", "nominal_depth", "distinct_outputs_frac"]
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rules)
    print(f"규칙 {len(rules)}개 → {out}, {out.with_suffix('.csv')}")
    for r in rules:
        print(f"{r['sym_kind']:11s} inv{r['inv_size']:>4} eq{r['eq_size']:>4} dep{r['dep_mean']:.1f} "
              f"depth{r['effective_depth']}/{r['nominal_depth']}  {r['spec']}")


if __name__ == "__main__":
    main()
