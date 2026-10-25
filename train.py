from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import compute_targets, load_split
from src.metrics import SymmetryProbe, evaluate, norms
from src.model import KINDS, build_model
from src.rules import PRESETS, Rule, analyze, op_coverage

CSV_FIELDS = ["step", "lr", "elapsed",
              "train_loss", "test_loss", "train_tok", "test_tok", "train_exact", "test_exact",
              "train_exact_tf", "test_exact_tf",
              "param_norm", "emb_norm", "sym_viol_tok", "sym_viol_seq", "sym_viol_tv"]


def log_grid(max_step):
    out, base = set(), 1
    while base <= max_step:
        for k in (1, 2, 5):
            if k * base <= max_step:
                out.add(k * base)
        base *= 10
    return out


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    sys.exit("GPU를 찾지 못했습니다. CUDA_VISIBLE_DEVICES 값과 nvidia-smi를 확인하세요. "
             "CPU로 돌리려면 --device cpu를 명시하세요.")


def make_autocast(device, dtype):
    if device.type == "cuda" and dtype in ("bf16", "fp16"):
        dt = torch.bfloat16 if dtype == "bf16" else torch.float16
        return lambda: torch.autocast("cuda", dtype=dt)
    return contextlib.nullcontext


def rule_name(spec):
    if spec in PRESETS:
        return spec
    return "r" + hashlib.sha1(spec.encode()).hexdigest()[:8]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="수열 변환 규칙 grokking 학습")
    ap.add_argument("--data", help="data.py로 만든 split 디렉터리")
    ap.add_argument("--rule", help="규칙 스펙 또는 프리셋 이름")
    ap.add_argument("--model", choices=KINDS, default="m0")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tie", choices=["auto", "yes", "no"], default="auto",
                    help="임베딩과 출력 head 공유 (auto: gpt만 공유)")
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=0.0, help="0이면 끔")
    ap.add_argument("--batch-size", type=int, default=-1, help="-1: full batch, 양수: 복원추출")
    ap.add_argument("--steps", type=int, default=100000, help="최대 step (resume 시 최종 step)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=4096, help="평가에 쓸 train/test 행 수 (-1: 전부)")
    ap.add_argument("--eval-batch", type=int, default=8192)
    ap.add_argument("--sym-probe", type=int, default=512, help="대칭 위반 점수에 쓸 test 입력 수")
    ap.add_argument("--sym-sigmas", type=int, default=8)
    ap.add_argument("--fit-thr", type=float, default=0.99, help="t_fit: train exact 기준")
    ap.add_argument("--gen-thr", type=float, default=0.95, help="t_gen: test exact 기준")
    ap.add_argument("--sym-thr", type=float, default=0.05, help="t_sym: 대칭 위반(칸) 기준")
    ap.add_argument("--stop-after-gen", type=int, default=-1,
                    help="t_gen 이후 이만큼 더 학습하고 멈춤 (-1: 멈추지 않음)")
    ap.add_argument("--budget-mult", type=float, default=0.0,
                    help="t_fit의 이 배수까지 t_gen이 없으면 멈춤 (0: 끔)")
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--ckpt-log", action="store_true", help="1,2,5,10,20,... step 체크포인트도 저장")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="CUDA에서만 적용. MPS/CPU는 fp32")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--name", help="run 디렉터리 이름")
    ap.add_argument("--resume", help="ckpt_last.pt 경로")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = resume["config"]
        keep = {"steps", "resume", "device", "compile", "quiet", "eval_batch", "stop_after_gen",
                "budget_mult", "ckpt_every"}
        for k, v in saved.items():
            if k not in keep:
                setattr(args, k, v)
        run_dir = Path(args.resume).parent
    if not args.data or not args.rule:
        sys.exit("--data와 --rule이 필요합니다")

    split = load_split(args.data)
    meta = split["meta"]
    n, m = meta["n"], meta["m"]
    rule = Rule(args.rule, n)

    if not resume:
        name = args.name or (f"{rule_name(args.rule)}_{args.model}_{Path(args.data).name}"
                             f"_wd{args.wd:g}_s{args.seed}")
        run_dir = Path(args.runs_dir) / name
        if run_dir.exists():
            sys.exit(f"이미 있는 run 디렉터리입니다: {run_dir} (--resume 또는 --name 사용)")
        run_dir.mkdir(parents=True)

    info_path = run_dir / "rule_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
    else:
        info = analyze(rule.spec, n, m)
        info_path.write_text(json.dumps(info, ensure_ascii=False))

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    autocast = make_autocast(device, args.dtype)

    def tensors(xs):
        ys = compute_targets(rule, xs)
        x = torch.as_tensor(xs.astype(np.int64) - 1, device=device)
        y = torch.as_tensor(ys.astype(np.int64) - 1, device=device)
        return x, y

    coverage = op_coverage(rule, split["train"].tolist())
    tr_x, tr_y = tensors(split["train"])
    te_x, te_y = tensors(split["test"])
    g = torch.Generator().manual_seed(args.seed + 1)

    def subset(x, y, k):
        if k < 0 or k >= len(x):
            return x, y
        idx = torch.randperm(len(x), generator=g)[:k].to(device)
        return x[idx], y[idx]

    ev_tr = subset(tr_x, tr_y, args.n_eval)
    ev_te = subset(te_x, te_y, args.n_eval)
    probe_x, probe_y = subset(te_x, te_y, args.sym_probe)
    group = info["inv_group"] if info["sym_kind"] == "invariant" else info["eq_group"]
    sym_probe = SymmetryProbe(info["sym_kind"], group, probe_x, probe_y, args.sym_sigmas, args.seed)

    tie = None if args.tie == "auto" else args.tie == "yes"
    model = build_model(args.model, n, m, args.d, args.layers, args.heads, tie, args.init_std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                            weight_decay=args.wd)
    lr_at = lambda s: args.lr * min(1.0, (s + 1) / max(1, args.warmup))
    batch_gen = torch.Generator(device="cpu").manual_seed(args.seed + 2)

    step, elapsed = 0, 0.0
    state = dict(t_fit=None, t_gen=None, t_sym=None)
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        step, elapsed, state = resume["step"], resume["elapsed"], resume["state"]
        batch_gen.set_state(resume["batch_gen"])
        torch.set_rng_state(resume["torch_rng"])
    fmodel = torch.compile(model) if args.compile else model

    config = {k: v for k, v in vars(args).items()}
    config.update(rule_spec=rule.spec, n=n, m=m, run_dir=str(run_dir), device=str(device),
                  n_params=sum(p.numel() for p in model.parameters()), data_meta=meta)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    csv_path = run_dir / "metrics.csv"
    if resume and csv_path.exists():
        rows = list(csv.DictReader(csv_path.open()))
        rows = [r for r in rows if int(r["step"]) <= step]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
    else:
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, CSV_FIELDS).writeheader()

    grid = log_grid(args.steps)

    def save_ckpt(path):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, elapsed=elapsed,
                        state=state, batch_gen=batch_gen.get_state(), torch_rng=torch.get_rng_state(),
                        config=vars(args)), path)

    def write_summary(stop_reason=None):
        t_fit, t_gen = state["t_fit"], state["t_gen"]
        summ = dict(rule=rule.spec, rule_name=rule_name(args.rule), model=args.model, seed=args.seed,
                    data=str(args.data), train_sets=meta["train_sets"], wd=args.wd,
                    sym_kind=info["sym_kind"], inv_size=info["inv_size"], eq_size=info["eq_size"],
                    dep_mean=info["dep_mean"], effective_depth=info["effective_depth"],
                    nominal_depth=info["nominal_depth"], op_coverage=coverage,
                    t_fit=t_fit, t_gen=t_gen, t_sym=state["t_sym"],
                    delay=(t_gen - t_fit) if (t_fit is not None and t_gen is not None) else None,
                    ratio=(t_gen / t_fit) if (t_fit and t_gen is not None) else None,
                    censored=t_gen is None, last_step=step, stop_reason=stop_reason,
                    fit_thr=args.fit_thr, gen_thr=args.gen_thr)
        (run_dir / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
        return summ

    def do_eval():
        a = evaluate(fmodel, *ev_tr, args.eval_batch, autocast)
        b = evaluate(fmodel, *ev_te, args.eval_batch, autocast)
        s = sym_probe(fmodel, autocast)
        nm = norms(model)
        row = dict(step=step, lr=lr_at(step), elapsed=round(elapsed, 2),
                   train_loss=a["loss"], test_loss=b["loss"], train_tok=a["tok_acc"], test_tok=b["tok_acc"],
                   train_exact=a["exact"], test_exact=b["exact"],
                   train_exact_tf=a["exact_tf"], test_exact_tf=b["exact_tf"],
                   param_norm=nm["param_norm"], emb_norm=nm["emb_norm"],
                   sym_viol_tok=s["viol_tok"], sym_viol_seq=s["viol_seq"], sym_viol_tv=s["viol_tv"])
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, CSV_FIELDS).writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        if state["t_fit"] is None and a["exact"] >= args.fit_thr:
            state["t_fit"] = step
        if state["t_gen"] is None and b["exact"] >= args.gen_thr:
            state["t_gen"] = step
        if state["t_sym"] is None and not math.isnan(s["viol_tok"]) and s["viol_tok"] <= args.sym_thr:
            state["t_sym"] = step
        if not args.quiet:
            print(f"step {step:>7d}  train {a['exact']:.3f}  test {b['exact']:.3f}  "
                  f"loss {a['loss']:.3g}/{b['loss']:.3g}  norm {nm['param_norm']:.1f}  "
                  f"sym {s['viol_tok']:.3f}  [{elapsed:.0f}s]", flush=True)
        return a, b

    model.train()
    if step == 0:
        do_eval()
        write_summary()
    stop_reason = None
    N = len(tr_x)
    t0 = time.time()
    while step < args.steps:
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        if args.batch_size < 0:
            xb, yb = tr_x, tr_y
        else:
            idx = torch.randint(0, N, (args.batch_size,), generator=batch_gen).to(device)
            xb, yb = tr_x[idx], tr_y[idx]
        with autocast():
            lg = fmodel.logits(xb, yb)
        loss = F.cross_entropy(lg.float().reshape(-1, n), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        if step % args.eval_every == 0 or step in grid or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.time() - t0
            do_eval()
            write_summary()
            t0 = time.time()
            if args.stop_after_gen >= 0 and state["t_gen"] is not None \
                    and step >= state["t_gen"] + args.stop_after_gen:
                stop_reason = "after_gen"
            if args.budget_mult > 0 and state["t_fit"] is not None and state["t_gen"] is None \
                    and step >= args.budget_mult * max(1, state["t_fit"]):
                stop_reason = "budget"
        if args.ckpt_log and step in grid:
            save_ckpt(run_dir / f"ckpt_{step:08d}.pt")
        if step % args.ckpt_every == 0:
            save_ckpt(run_dir / "ckpt_last.pt")
        if stop_reason:
            break
    elapsed += time.time() - t0
    save_ckpt(run_dir / "ckpt_last.pt")
    summ = write_summary(stop_reason or "max_steps")
    if not args.quiet:
        print(json.dumps(summ, ensure_ascii=False))
    return summ


if __name__ == "__main__":
    main()
