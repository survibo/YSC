from __future__ import annotations

import numpy as np

from src.rules import COMMUTATIVE_OPS, Rule, ref_pairs


def cell_keys(rule: Rule, xs: np.ndarray) -> np.ndarray:
    if len(rule.steps) != 1 or rule.steps[0].code != "REF":
        raise ValueError(f"오답 섞기는 단일 REF 규칙만 지원함: {rule.spec}")
    st, n = rule.steps[0], rule.n
    comm = st.p("op") in COMMUTATIVE_OPS
    out = np.empty(xs.shape, dtype=np.int64)
    for r, row in enumerate(xs.tolist()):
        for i, (a, b) in enumerate(ref_pairs(st, row, n)):
            if comm and a > b:
                a, b = b, a
            out[r, i] = (a - 1) * n + (b - 1)
    return out


def make_noise(rule: Rule, x_tr: np.ndarray, y_tr: np.ndarray, frac: float, mode: str, seed: int = 777) -> dict:
    N, m = y_tr.shape
    n = rule.n
    rng = np.random.default_rng(seed)
    keys = cell_keys(rule, x_tr)
    y = y_tr.copy()
    mask = np.zeros((N, m), dtype=bool)
    cell_wrong: dict[int, int] = {}
    target = int(round(frac * N))
    if target == 0:
        pass
    elif mode == "scatter":
        rows = rng.choice(N, target, replace=False)
        pos = rng.integers(0, m, target)
        wrong = (y_tr[rows, pos] + rng.integers(1, n, target)) % n
        y[rows, pos] = wrong
        mask[rows, pos] = True
        for k, w in zip(keys[rows, pos].tolist(), wrong.tolist()):
            cell_wrong.setdefault(k, w)
    elif mode == "cell":
        uniq, counts = np.unique(keys, return_counts=True)
        order = rng.permutation(len(uniq))
        got = 0
        for j in order:
            if got >= target:
                break
            k = int(uniq[j])
            sel = keys == k
            true_v = int(y_tr[sel][0])
            w = (true_v + int(rng.integers(1, n))) % n
            y[sel] = w
            mask |= sel
            cell_wrong[k] = w
            got += int(counts[j])
    else:
        raise ValueError(f"알 수 없는 오답 방식: {mode}")
    info = dict(mode=mode, frac=frac, seed=seed, tokens=int(mask.sum()), rows=int(mask.any(1).sum()),
                cells=len(cell_wrong), train_rows=N, total_cells=int(len(np.unique(keys))))
    return dict(y_noisy=y, mask=mask, cell_wrong=cell_wrong, keys=keys, info=info)


def test_cells(rule: Rule, x_te: np.ndarray, cell_wrong: dict) -> tuple:
    keys = cell_keys(rule, x_te)
    wrong = np.full(keys.shape, -1, dtype=np.int64)
    for k, w in cell_wrong.items():
        wrong[keys == k] = w
    return wrong >= 0, wrong
