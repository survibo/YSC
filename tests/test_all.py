import json
import math

import pytest
import torch

from src.data import load_split, make_split, rank_combination, save_split, unrank_combination
from src.model import build_model
from src.rules import PRESETS, Rule, RuleError, analyze, sample_rules


def test_rule_lab_demo_matches_web():
    r = Rule("Q:first > I:add(c=3) > B:left(k=8) > R:add", 50)
    assert r((7, 2, 49, 13, 25, 1)) == (10, 24, 36, 12, 18, 13)


def test_rule_lab_steps():
    n = 50
    assert Rule("A:high", n).raw([3, 1, 4, 5, 2]) == [5, 1, 4, 2, 3]
    assert Rule("A:low", n).raw([3, 1, 4, 5, 2]) == [1, 5, 2, 4, 3]
    assert Rule("D:odd", n).raw([1, 2, 3, 4, 5]) == [5, 2, 3, 4, 1]
    assert Rule("E:shuffle", n).raw([1, 2, 3, 4, 5]) == [1, 4, 2, 5, 3]
    assert Rule("E:inverse", n).raw([1, 4, 2, 5, 3]) == [1, 2, 3, 4, 5]
    assert Rule("AF:diff", n).raw([1, 3, 6, 10, 15]) == [5, 10, 17, 26, 197]
    assert Rule("AG", n).raw([1, 2, 3, 4, 5]) == [2, 1, 4, 3, 5]
    assert Rule("P:product", n).raw([10, 23, 7]) == [1, 6, 7]
    assert Rule("BA:largest", n).raw([1, 12, 13]) == [1, 3, 13]
    assert Rule("BC:addG", n).raw([4, 6, 8]) == [6, 8, 10]
    assert Rule("AO:add(period=2|5|3)", n).raw([1, 1, 1, 1]) == [3, 6, 4, 3]


def test_r_requires_q():
    with pytest.raises(RuleError):
        Rule("R:add", 50)


@pytest.mark.parametrize("name,inv,eq", [
    ("sort", 120, 1), ("add_total", 1, 120), ("add_first", 1, 24),
    ("nbsum", 1, 5), ("add_pos", 1, 1), ("l1_sortnext", 120, 1), ("l1_map", 1, 1),
])
def test_symmetry_sizes(name, inv, eq):
    info = analyze(name, 50, 5, n_probe=60)
    assert (info["inv_size"], info["eq_size"]) == (inv, eq)


def test_effective_depth_controls():
    assert analyze("ctrl_add_mul_sort", n_probe=60)["effective_depth"] == 1
    assert analyze("ctrl_sort_rot_rev", n_probe=60)["effective_depth"] == 1
    assert analyze("add_c", n_probe=60)["trivial"]


def test_all_presets_parse():
    for name in PRESETS:
        Rule(name, 50)((7, 2, 49, 13, 25))


def test_sample_rules():
    rs = sample_rules(5, seed=1, n_probe=40)
    assert len(rs) == 5
    assert all(not r["trivial"] and not r["pointwise"] for r in rs)


def test_unrank_roundtrip():
    for rk in [0, 1, 300, math.comb(12, 4) - 1]:
        c = unrank_combination(rk, 12, 4)
        assert rank_combination(c, 12) == rk


def test_split_disjoint(tmp_path):
    s = make_split(12, 4, train_count=50, test_count=40, orders_per_set=2, seed=3)
    assert len(s["train"]) == 100 and len(s["test"]) == 80
    assert not set(s["train_set"].tolist()) & set(s["test_set"].tolist())
    for row, rk in zip(s["train"].tolist(), s["train_set"].tolist()):
        assert rank_combination(row, 12) == rk and len(set(row)) == 4
    save_split(s, tmp_path / "d")
    assert load_split(tmp_path / "d")["meta"]["train_rows"] == 100


@pytest.mark.parametrize("kind", ["m0", "minv", "meq", "m0eq", "gpt"])
def test_model_shapes_and_symmetry(kind):
    torch.manual_seed(0)
    x = torch.stack([torch.randperm(20)[:5] for _ in range(4)])
    y = torch.randint(0, 20, (4, 5))
    mdl = build_model(kind, 20, 5, d=32, n_layer=1, n_head=2).eval()
    a = mdl.logits(x, y)
    assert a.shape == (4, 5, 20) and mdl.predict(x).shape == (4, 5)
    p = torch.tensor([3, 0, 4, 1, 2])
    b = mdl.logits(x[:, p], y)
    if kind == "minv":
        assert torch.allclose(a, b, atol=1e-5)
    if kind == "meq":
        assert torch.allclose(a[:, p], b, atol=1e-5)


def test_train_and_resume(tmp_path):
    import train
    s = make_split(12, 4, train_count=60, test_count=30, seed=0)
    save_split(s, tmp_path / "d")
    args = ["--data", str(tmp_path / "d"), "--rule", "l1_first", "--model", "m0", "--d", "32",
            "--layers", "1", "--heads", "2", "--steps", "20", "--eval-every", "10",
            "--runs-dir", str(tmp_path / "runs"), "--name", "r", "--quiet", "--device", "cpu"]
    summ = train.main(args)
    assert summ["last_step"] == 20
    summ = train.main(["--resume", str(tmp_path / "runs" / "r" / "ckpt_last.pt"), "--steps", "30",
                       "--quiet", "--device", "cpu"])
    assert summ["last_step"] == 30
    rows = (tmp_path / "runs" / "r" / "metrics.csv").read_text().strip().splitlines()
    steps = [int(r.split(",")[0]) for r in rows[1:]]
    assert steps == sorted(steps) and steps[-1] == 30
    info = json.loads((tmp_path / "runs" / "r" / "rule_info.json").read_text())
    assert info["eq_size"] == 6


def test_train_multi_matches_single(tmp_path):
    import train
    import train_multi
    s = make_split(12, 4, train_count=60, test_count=30, seed=0)
    save_split(s, tmp_path / "d")
    common = ["--data", str(tmp_path / "d"), "--model", "m0", "--d", "32", "--layers", "1", "--heads", "2",
              "--steps", "20", "--eval-every", "10", "--n-eval", "-1", "--quiet", "--device", "cpu"]
    for rule, seed in [("l1_first", 0), ("sort", 2)]:
        train.main(common + ["--rule", rule, "--seed", str(seed), "--runs-dir", str(tmp_path / "single")])
    (tmp_path / "m.json").write_text(json.dumps([{"rule": "l1_first", "seed": 0}, {"rule": "sort", "seed": 2}]))
    train_multi.main(common + ["--members", str(tmp_path / "m.json"), "--runs-dir", str(tmp_path / "multi"),
                               "--rows-budget", "64"])
    for name in ["l1_first_m0_d_wd1_s0", "sort_m0_d_wd1_s2"]:
        a = (tmp_path / "single" / name / "metrics.csv").read_text().splitlines()
        b = (tmp_path / "multi" / name / "metrics.csv").read_text().splitlines()
        cut = lambda rows: [",".join(r.split(",")[3:9]) for r in rows]
        assert cut(a) == cut(b)
