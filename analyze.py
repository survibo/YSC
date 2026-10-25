from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def first_step(rows, key, thr, below=False):
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan"):
            continue
        v = float(v)
        if math.isnan(v):
            continue
        if (v <= thr) if below else (v >= thr):
            return int(r["step"])
    return None


def summarize_run(run_dir: Path, fit_thr, gen_thr, sym_thr):
    cfg = json.loads((run_dir / "config.json").read_text())
    info = json.loads((run_dir / "rule_info.json").read_text())
    summ = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    if not rows:
        return None
    t_fit = first_step(rows, "train_exact", fit_thr)
    t_gen = first_step(rows, "test_exact", gen_thr)
    t_sym = first_step(rows, "sym_viol_tok", sym_thr, below=True)
    last = rows[-1]
    censored = t_gen is None
    return dict(
        run=run_dir.name, rule_arg=cfg["rule"], rule=info["spec"], model=cfg["model"], seed=cfg["seed"], wd=cfg["wd"],
        train_sets=cfg["data_meta"]["train_sets"], data=Path(cfg["data"]).name,
        sym_kind=info["sym_kind"], inv_size=info["inv_size"], eq_size=info["eq_size"],
        permuting_size=info["permuting_size"], dep_mean=info["dep_mean"],
        effective_depth=info["effective_depth"], nominal_depth=info["nominal_depth"],
        op_coverage=summ.get("op_coverage"),
        t_fit=t_fit, t_gen=t_gen if not censored else int(last["step"]), censored=int(censored),
        t_sym=t_sym,
        delay=(t_gen - t_fit) if (t_fit is not None and not censored) else None,
        ratio=(t_gen / t_fit) if (t_fit and not censored) else None,
        sym_before_gen=(int(t_sym <= t_gen) if (t_sym is not None and not censored) else None),
        last_step=int(last["step"]), final_train_exact=float(last["train_exact"]),
        final_test_exact=float(last["test_exact"]),
    )


def group_table(rows, keys=("rule", "model", "train_sets", "wd")):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        done = [r for r in rs if not r["censored"]]
        delays = [r["delay"] for r in done if r["delay"] is not None]
        ratios = [r["ratio"] for r in done if r["ratio"] is not None]
        out.append(dict(zip(keys, k), runs=len(rs), grokked=len(done),
                        grok_rate=len(done) / len(rs),
                        delay_median=statistics.median(delays) if delays else None,
                        ratio_median=statistics.median(ratios) if ratios else None,
                        sym_kind=rs[0]["sym_kind"], inv_size=rs[0]["inv_size"], eq_size=rs[0]["eq_size"],
                        dep_mean=rs[0]["dep_mean"], effective_depth=rs[0]["effective_depth"],
                        op_coverage=rs[0]["op_coverage"]))
    return out


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run 디렉터리 glob")
    ap.add_argument("--fit-thr", type=float, default=0.99)
    ap.add_argument("--gen-thr", type=float, default=0.95)
    ap.add_argument("--sym-thr", type=float, default=0.05)
    ap.add_argument("--out", default="results.csv", help="run별 표")
    ap.add_argument("--group-out", default="results_grouped.csv", help="조건별 요약 표")
    a = ap.parse_args()
    dirs = sorted({Path(p) for pat in a.runs for p in glob.glob(pat) if (Path(p) / "metrics.csv").exists()})
    rows = [r for d in dirs if (r := summarize_run(d, a.fit_thr, a.gen_thr, a.sym_thr))]
    write_csv(rows, a.out)
    grouped = group_table(rows)
    write_csv(grouped, a.group_out)
    print(f"run {len(rows)}개 → {a.out}, 조건 {len(grouped)}개 → {a.group_out}")
    for g in grouped:
        print(f"{g['rule'][:40]:40s} {g['model']:5s} tr{g['train_sets']:<7} grok {g['grokked']}/{g['runs']}"
              f"  delay {g['delay_median']}  ratio {g['ratio_median']}")


if __name__ == "__main__":
    main()
