from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def unrank_combination(rank: int, n: int, m: int) -> tuple:
    if not 0 <= rank < math.comb(n, m):
        raise ValueError(f"조합 번호 범위 밖: {rank}")
    out, x = [], 1
    for k in range(m, 0, -1):
        while True:
            c = math.comb(n - x, k - 1)
            if rank < c:
                out.append(x)
                x += 1
                break
            rank -= c
            x += 1
    return tuple(out)


def rank_combination(comb, n: int) -> int:
    comb = sorted(comb)
    m, rank, prev = len(comb), 0, 0
    for i, v in enumerate(comb):
        for x in range(prev + 1, v):
            rank += math.comb(n - x, m - i - 1)
        prev = v
    return rank


def make_split(n: int, m: int, train_count: int | None = None, train_frac: float | None = None,
               test_count: int = 20000, orders_per_set: int = 1, seed: int = 0) -> dict:
    total = math.comb(n, m)
    if (train_count is None) == (train_frac is None):
        raise ValueError("train_count와 train_frac 중 하나만 지정")
    if train_count is None:
        train_count = round(total * train_frac)
    if train_count < 1 or test_count < 0 or train_count + test_count > total:
        raise ValueError(f"집합 수 오류: train {train_count} + test {test_count} > 전체 {total}")
    if orders_per_set < 1 or orders_per_set > math.factorial(m):
        raise ValueError("orders_per_set 범위 오류")
    rng = random.Random(seed)
    ranks = rng.sample(range(total), train_count + test_count)
    tr_ranks, te_ranks = ranks[:train_count], ranks[train_count:]

    def rows(rank_list, tag):
        r = random.Random(f"{seed}-{tag}-orders")
        xs, ids = [], []
        for rk in rank_list:
            base = list(unrank_combination(rk, n, m))
            chosen = set()
            while len(chosen) < orders_per_set:
                p = base[:]
                r.shuffle(p)
                chosen.add(tuple(p))
            for p in sorted(chosen, key=lambda _: r.random()):
                xs.append(p)
                ids.append(rk)
        return np.array(xs, dtype=np.int16).reshape(-1, m), np.array(ids, dtype=np.int64)

    train_x, train_id = rows(tr_ranks, "train")
    test_x, test_id = rows(te_ranks, "test")
    meta = dict(format_version=FORMAT_VERSION, n=n, m=m, total_sets=total,
                train_sets=train_count, test_sets=test_count,
                train_frac=train_count / total, orders_per_set=orders_per_set, seed=seed,
                train_rows=len(train_x), test_rows=len(test_x))
    return dict(meta=meta, train=train_x, test=test_x, train_set=train_id, test_set=test_id)


def save_split(split: dict, out: str | Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for k in ("train", "test", "train_set", "test_set"):
        np.save(out / f"{k}.npy", split[k])
    (out / "meta.json").write_text(json.dumps(split["meta"], indent=2, ensure_ascii=False))


def load_split(path: str | Path) -> dict:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"데이터 형식 버전이 다릅니다: {meta.get('format_version')}")
    d = dict(meta=meta)
    for k in ("train", "test", "train_set", "test_set"):
        d[k] = np.load(path / f"{k}.npy")
    return d


def compute_targets(rule, xs: np.ndarray) -> np.ndarray:
    out = np.empty_like(xs)
    cache = {}
    for i, row in enumerate(xs.tolist()):
        key = tuple(row)
        if key not in cache:
            cache[key] = rule(key)
        out[i] = cache[key]
    return out


def default_name(n, m, train_count, test_count, orders, seed):
    return f"n{n}_m{m}_tr{train_count}_te{test_count}_o{orders}_s{seed}"


def main():
    ap = argparse.ArgumentParser(description="값 집합 단위 train/test split")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--m", type=int, default=5)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--train-count", type=int, help="train 값 집합 개수")
    g.add_argument("--train-frac", type=float, help="train 값 집합 비율 (예: 0.01)")
    ap.add_argument("--test-count", type=int, default=20000)
    ap.add_argument("--orders-per-set", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="출력 디렉터리 (기본 data/<자동 이름>)")
    a = ap.parse_args()
    if a.train_count is None and a.train_frac is None:
        a.train_frac = 0.01
    split = make_split(a.n, a.m, a.train_count, a.train_frac, a.test_count, a.orders_per_set, a.seed)
    meta = split["meta"]
    out = a.out or f"data/{default_name(a.n, a.m, meta['train_sets'], a.test_count, a.orders_per_set, a.seed)}"
    save_split(split, out)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
