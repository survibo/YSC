from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter

import numpy as np

from src.data import FORMAT_VERSION, save_split, unrank_combination
from src.rules import Rule, ref_pairs


def _row(rank, n, m, seed):
    base = list(unrank_combination(rank, n, m))
    random.Random(f"curate-{seed}-order-{rank}").shuffle(base)
    return tuple(base)


def _pairs(rule, row):
    st = rule.steps[0]
    comm = st.p("op") == "add"
    return [frozenset(p) if comm else p for p in ref_pairs(st, list(row), rule.n)], comm


def make_curated_split(rule_spec, n=50, m=5, train_count=1024, target=None, test_count=20000,
                       pool_size=30000, seed=0, min_value_count=20, tol=0.002, max_iter=20000,
                       sample=3000):
    rule = Rule(rule_spec, n)
    if len(rule.steps) != 1 or rule.steps[0].code != "REF":
        raise ValueError("REF 한 단계 규칙만 지원")
    total = math.comb(n, m)
    rng = random.Random(f"curate-{seed}")
    ranks = rng.sample(range(total), test_count + pool_size)
    te_ranks, pool_ranks = ranks[:test_count], ranks[test_count:]
    pool_rows = [_row(r, n, m, seed) for r in pool_ranks]
    pool_pairs = []
    comm = True
    for row in pool_rows:
        ps, comm = _pairs(rule, row)
        pool_pairs.append(set(ps))
    denom = n * (n + 1) / 2 if comm else n * n

    sel = list(range(train_count))
    rest = list(range(train_count, pool_size))
    pc, vc = Counter(), Counter()
    for i in sel:
        pc.update(pool_pairs[i])
        vc.update(pool_rows[i])
    cov = lambda: len(pc) / denom
    start = cov()

    it = 0
    if target is not None:
        up = target > start
        while abs(cov() - target) > tol and it < max_iter:
            it += 1
            out_cand = rng.sample(range(len(sel)), min(sample // 3, len(sel)))
            in_cand = rng.sample(range(len(rest)), min(sample, len(rest)))
            uniq = lambda i: sum(1 for p in pool_pairs[sel[i]] if pc[p] == 1)
            newp = lambda j: sum(1 for p in pool_pairs[rest[j]] if pc[p] == 0)
            ok_out = [i for i in out_cand if all(vc[v] > min_value_count for v in pool_rows[sel[i]])]
            if not ok_out:
                continue
            if up:
                i = min(ok_out, key=uniq)
                j = max(in_cand, key=newp)
                gain = newp(j) - uniq(i)
                if gain <= 0:
                    continue
            else:
                i = max(ok_out, key=uniq)
                j = min(in_cand, key=newp)
                gain = newp(j) - uniq(i)
                if gain >= 0:
                    continue
            a, b = sel[i], rest[j]
            pc.subtract(pool_pairs[a])
            for p in pool_pairs[a]:
                if pc[p] <= 0:
                    del pc[p]
            vc.subtract(pool_rows[a])
            pc.update(pool_pairs[b])
            vc.update(pool_rows[b])
            sel[i], rest[j] = b, a

    train_x = np.array([pool_rows[i] for i in sel], dtype=np.int16)
    test_x = np.array([_row(r, n, m, seed) for r in te_ranks], dtype=np.int16)
    meta = dict(format_version=FORMAT_VERSION, n=n, m=m, total_sets=total, train_sets=train_count,
                test_sets=test_count, train_frac=train_count / total, orders_per_set=1, seed=seed,
                train_rows=len(train_x), test_rows=len(test_x),
                curated=dict(rule=rule.spec, target=target, start_coverage=round(start, 4),
                             coverage=round(cov(), 4), iterations=it, pool_size=pool_size,
                             min_value_count=min(vc[v] for v in range(1, n + 1)),
                             reached=target is None or abs(cov() - target) <= tol))
    return dict(meta=meta, train=train_x, test=test_x,
                train_set=np.array([pool_ranks[i] for i in sel], dtype=np.int64),
                test_set=np.array(te_ranks, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser(description="연산 표 coverage를 맞춘 train split")
    ap.add_argument("--rule", required=True, help="REF 한 단계 규칙 (프리셋 이름 가능)")
    ap.add_argument("--train-count", type=int, required=True)
    ap.add_argument("--target", type=float, help="목표 coverage (없으면 무작위 선택)")
    ap.add_argument("--test-count", type=int, default=20000)
    ap.add_argument("--pool-size", type=int, default=30000)
    ap.add_argument("--min-value-count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = make_curated_split(a.rule, train_count=a.train_count, target=a.target, test_count=a.test_count,
                           pool_size=a.pool_size, seed=a.seed, min_value_count=a.min_value_count)
    save_split(s, a.out)
    c = s["meta"]["curated"]
    print(f"{a.out}: coverage {c['start_coverage']} → {c['coverage']} (목표 {c['target']}, 도달 {c['reached']}, "
          f"반복 {c['iterations']}, 값 최소 등장 {c['min_value_count']})")
    if not c["reached"]:
        raise SystemExit("목표 coverage에 도달하지 못했습니다")


if __name__ == "__main__":
    main()
