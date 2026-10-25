from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap

from src.data import compute_targets, load_split
from src.metrics import SymmetryProbe
from src.model import KINDS, build_model
from src.noise import make_noise, test_cells
from src.rules import Rule, analyze, op_coverage
from train import CSV_FIELDS, log_grid, make_autocast, pick_device, rule_name


def member_name(rule, model, data, wd, seed):
    return f"{rule_name(rule)}_{model}_{Path(data).name}_wd{wd:g}_s{seed}"


def group_id(names):
    return "g" + hashlib.sha1("\n".join(sorted(names)).encode()).hexdigest()[:10]


class MemberView:

    def __init__(self, base, params):
        self.base, self.params = base, params

    def logits(self, x, y=None):
        return functional_call(self.base, self.params, (x, y))

    def predict(self, x):
        return functional_call(self.base, self.params, (x,), {"predict": True})

    def eval(self):
        pass

    def train(self):
        pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="여러 run 묶음 학습")
    ap.add_argument("--data")
    ap.add_argument("--members", help="members.json 경로")
    ap.add_argument("--model", choices=KINDS, default="m0")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tie", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0, help="members.json에 wd가 없을 때 기본값")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=-1, help="-1: full batch, 양수: run별 복원추출")
    ap.add_argument("--train-mask", type=int, default=0,
                    help="k>0: train 예제마다 출력 칸 k개만 loss에 넣는다(고정, 예제별 무작위). train 정확도도 그 칸만 잰다")
    ap.add_argument("--noise-frac", type=float, default=0.0,
                    help="train에 섞을 오답 토큰 수 = round(frac × train 예제 수). 0이면 오답 없음")
    ap.add_argument("--noise-mode", choices=["scatter", "cell"], default="scatter",
                    help="scatter: 예제마다 한 칸, cell: 표 칸 단위로 일관된 오답 (src/noise.py)")
    ap.add_argument("--noise-seed", type=int, default=777)
    ap.add_argument("--loss", choices=["ce", "mse"], default="ce",
                    help="학습 손실. mse = 출력 logit과 one-hot 정답의 제곱 오차 합. 평가 loss는 항상 CE")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=4096)
    ap.add_argument("--rows-budget", type=int, default=16384,
                    help="한 번에 계산할 (run 수 × 행 수) 상한. 넘으면 나눠 계산하고 gradient를 모은다 (결과 같음)")
    ap.add_argument("--sym-probe", type=int, default=512)
    ap.add_argument("--sym-sigmas", type=int, default=8)
    ap.add_argument("--sym-every", type=int, default=1000,
                    help="대칭 위반 점수를 재는 간격(step). 그 사이 평가에서는 빈 값. 0이면 재지 않음")
    ap.add_argument("--fit-thr", type=float, default=0.99)
    ap.add_argument("--gen-thr", type=float, default=0.95)
    ap.add_argument("--sym-thr", type=float, default=0.05)
    ap.add_argument("--stop-after-gen", type=int, default=-1)
    ap.add_argument("--budget-mult", type=float, default=0.0)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--ckpt-log", action="store_true")
    ap.add_argument("--ckpt-dense", type=int, default=0,
                    help="--ckpt-log와 함께: N step마다 run별 체크포인트도 저장 (내부 분석을 촘촘히)")
    ap.add_argument("--ckpt-early-every", type=int, default=0,
                    help="--ckpt-log와 함께: 처음 --ckpt-early-until step까지는 이 간격으로 저장 (초반 오답 암기·교정 추적)")
    ap.add_argument("--ckpt-early-until", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--resume", help="runs/_groups/<묶음>/ckpt_last.pt")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        keep = {"steps", "resume", "device", "quiet", "rows_budget", "stop_after_gen", "budget_mult", "sym_every",
                "ckpt_every"}
        for k, v in resume["config"].items():
            if k not in keep:
                setattr(args, k, v)
        members = resume["members"]
    else:
        if not args.data or not args.members:
            sys.exit("--data와 --members가 필요합니다")
        members = json.loads(Path(args.members).read_text())
        if isinstance(members, dict):
            members = members["members"]
        for mb in members:
            mb.setdefault("wd", args.wd)
            mb["seed"] = int(mb["seed"])
            mb["wd"] = float(mb["wd"])
            mb["name"] = member_name(mb["rule"], args.model, args.data, mb["wd"], mb["seed"])

    split = load_split(args.data)
    meta = split["meta"]
    n, m = meta["n"], meta["m"]
    device = pick_device(args.device)
    autocast = make_autocast(device, args.dtype)
    runs_dir = Path(args.runs_dir)
    gid = group_id([mb["name"] for mb in members])
    gdir = runs_dir / "_groups" / gid
    gdir.mkdir(parents=True, exist_ok=True)
    K = len(members)

    x_tr = torch.as_tensor(split["train"].astype(np.int64) - 1, device=device)
    x_te = torch.as_tensor(split["test"].astype(np.int64) - 1, device=device)
    g = torch.Generator().manual_seed(1)
    pick = lambda N, k: (torch.arange(N) if k < 0 or k >= N else torch.randperm(N, generator=g)[:k]).to(device)
    i_tr, i_te = pick(len(x_tr), args.n_eval), pick(len(x_te), args.n_eval)
    mask_tr = None
    if args.train_mask > 0:
        g = torch.Generator().manual_seed(12345)
        r = torch.rand(len(x_tr), meta["m"], generator=g)
        mask_tr = (r.argsort(1).argsort(1) < args.train_mask).to(device)
    i_probe = pick(len(x_te), args.sym_probe)

    rules, infos, Y_tr, Y_te, probes, run_dirs, covs = [], [], [], [], [], [], []
    noises = []
    train_list = split["train"].tolist()
    for mb in members:
        rule = Rule(mb["rule"], n)
        rd = runs_dir / mb["name"]
        if not resume and rd.exists() and (rd / "metrics.csv").exists():
            prev = json.loads((rd / "config.json").read_text()).get("group") if (rd / "config.json").exists() else None
            if prev != gid:
                sys.exit(f"이미 있는 run입니다: {rd}")
        rd.mkdir(parents=True, exist_ok=True)
        ip = rd / "rule_info.json"
        info = json.loads(ip.read_text()) if ip.exists() else analyze(rule.spec, n, m)
        ip.write_text(json.dumps(info, ensure_ascii=False))
        ytr_np = compute_targets(rule, split["train"]).astype(np.int64) - 1
        if args.noise_frac > 0:
            nz = make_noise(rule, split["train"], ytr_np, args.noise_frac, args.noise_mode, args.noise_seed)
            te_mask, te_wrong = test_cells(rule, split["test"], nz["cell_wrong"])
            noises.append(dict(y_true=torch.as_tensor(ytr_np, device=device),
                               mask=torch.as_tensor(nz["mask"], device=device),
                               te_mask=torch.as_tensor(te_mask, device=device),
                               te_wrong=torch.as_tensor(te_wrong, device=device)))
            (rd / "noise.json").write_text(json.dumps(dict(nz["info"], test_tokens=int(te_mask.sum()),
                                                           test_rows=int(te_mask.any(1).sum())), ensure_ascii=False))
            ytr_np = nz["y_noisy"]
        ytr = torch.as_tensor(ytr_np, device=device)
        yte = torch.as_tensor(compute_targets(rule, split["test"]).astype(np.int64) - 1, device=device)
        group = info["inv_group"] if info["sym_kind"] == "invariant" else info["eq_group"]
        probes.append(SymmetryProbe(info["sym_kind"], group, x_te[i_probe], yte[i_probe],
                                    args.sym_sigmas, mb["seed"]))
        covs.append(op_coverage(rule, train_list))
        rules.append(rule); infos.append(info); Y_tr.append(ytr); Y_te.append(yte); run_dirs.append(rd)
        cfg = {k: v for k, v in vars(args).items()}
        cfg.update(rule=mb["rule"], seed=mb["seed"], wd=mb["wd"], rule_spec=rule.spec, n=n, m=m,
                   run_dir=str(rd), device=str(device), data_meta=meta, group=gid, trainer="train_multi")
        (rd / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        if not resume:
            with (rd / "metrics.csv").open("w", newline="") as f:
                csv.DictWriter(f, CSV_FIELDS).writeheader()
    Y_tr, Y_te = torch.stack(Y_tr), torch.stack(Y_te)

    tie = None if args.tie == "auto" else args.tie == "yes"
    models = []
    for mb in members:
        torch.manual_seed(mb["seed"])
        models.append(build_model(args.model, n, m, args.d, args.layers, args.heads, tie, args.init_std).to(device))
    base = copy.deepcopy(models[0])
    params, _ = stack_module_state(models)
    del models
    n_params = sum(v[0].numel() for v in params.values())
    wd = torch.tensor([mb["wd"] for mb in members], device=device)

    def make_opt(ps):
        kw = dict(lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0)
        if device.type == "cuda":
            kw["fused"] = True
        return torch.optim.AdamW(list(ps.values()), **kw)

    opt = make_opt(params)
    active = list(range(K))
    step, elapsed = 0, 0.0
    states = [dict(t_fit=None, t_gen=None, t_sym=None, done=False, stop_reason=None) for _ in members]
    if resume:
        active, step, elapsed, states = resume["active"], resume["step"], resume["elapsed"], resume["states"]
        params = {k: v.to(device).requires_grad_() for k, v in resume["params"].items()}
        opt = make_opt(params)
        opt.load_state_dict(resume["opt"])
        for i, mb in enumerate(members):
            p = run_dirs[i] / "metrics.csv"
            rows = [r for r in csv.DictReader(p.open()) if int(r["step"]) <= step] if p.exists() else []
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)

    lr_at = lambda s: args.lr * min(1.0, (s + 1) / max(1, args.warmup))
    batch_gen = torch.Generator().manual_seed(12345)
    grid = log_grid(args.steps)

    def f_logits(p, x, y):
        return functional_call(base, p, (x, y))

    def f_pred(p, x):
        return functional_call(base, p, (x,), {"predict": True})

    def save_group_ckpt():
        torch.save(dict(params={k: v.detach().cpu() for k, v in params.items()}, opt=opt.state_dict(),
                        active=active, step=step, elapsed=elapsed, states=states, members=members,
                        config=vars(args)), gdir / "ckpt_last.pt")

    def save_member_ckpt(slot, i):
        mdl = copy.deepcopy(base)
        with torch.no_grad():
            for name, p in mdl.named_parameters():
                p.copy_(params[name][slot])
        torch.save(dict(model=mdl.state_dict(), step=step, config=json.loads((run_dirs[i] / "config.json").read_text())),
                    run_dirs[i] / f"ckpt_{step:08d}.pt")

    def write_summary(i):
        st, info, mb = states[i], infos[i], members[i]
        t_fit, t_gen = st["t_fit"], st["t_gen"]
        summ = dict(rule=rules[i].spec, rule_name=rule_name(mb["rule"]), model=args.model, seed=mb["seed"],
                    data=str(args.data), train_sets=meta["train_sets"], wd=mb["wd"],
                    sym_kind=info["sym_kind"], inv_size=info["inv_size"], eq_size=info["eq_size"],
                    dep_mean=info["dep_mean"], effective_depth=info["effective_depth"],
                    nominal_depth=info["nominal_depth"], op_coverage=covs[i], t_fit=t_fit, t_gen=t_gen, t_sym=st["t_sym"],
                    delay=(t_gen - t_fit) if (t_fit is not None and t_gen is not None) else None,
                    ratio=(t_gen / t_fit) if (t_fit and t_gen is not None) else None,
                    censored=t_gen is None, last_step=step, stop_reason=st["stop_reason"],
                    fit_thr=args.fit_thr, gen_thr=args.gen_thr, group=gid)
        (run_dirs[i] / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))

    @torch.no_grad()
    def eval_split(x, Y, idx, mask=None):
        A = len(active)
        Ya = Y[active][:, idx]
        xs = x[idx]
        ms = mask[idx] if mask is not None else None
        loss = torch.zeros(A, device=device)
        tok = torch.zeros(A, device=device); ex = torch.zeros(A, device=device)
        tok_tf = torch.zeros(A, device=device); ex_tf = torch.zeros(A, device=device)
        chunk = max(64, args.rows_budget // A)
        for s in range(0, len(idx), chunk):
            xb, yb = xs[s:s + chunk], Ya[:, s:s + chunk]
            with autocast():
                lg = vmap(f_logits, in_dims=(0, None, 0))(params, xb, yb).float()
                pr = vmap(f_pred, in_dims=(0, None))(params, xb)
            mb = ms[s:s + chunk].unsqueeze(0) if ms is not None else torch.ones_like(yb[:1], dtype=torch.bool)
            loss += (F.cross_entropy(lg.permute(0, 3, 1, 2), yb, reduction="none") * mb).sum((1, 2))
            am = lg.argmax(-1)
            tok_tf += ((am == yb) & mb).float().sum((1, 2)); ex_tf += ((am == yb) | ~mb).all(-1).float().sum(1)
            tok += ((pr == yb) & mb).float().sum((1, 2)); ex += ((pr == yb) | ~mb).all(-1).float().sum(1)
        N = len(idx)
        T = float(ms.sum()) if ms is not None else N * m
        return dict(loss=loss / T, tok=tok / T, exact=ex / N, tok_tf=tok_tf / T, exact_tf=ex_tf / N)

    nan_sym = dict(viol_tok=math.nan, viol_seq=math.nan, viol_tv=math.nan)

    NOISE_FIELDS = ["step", "mem", "fixed", "other", "true_rank2", "true_rank_med", "true_prob",
                    "clean_rows_exact", "test_cell_tokens", "test_spread", "test_true_on_cells", "test_clean_rows_exact"]
    if noises:
        nz0 = noises[0]
        nrows = nz0["mask"].any(1).nonzero().squeeze(1)
        gq = torch.Generator().manual_seed(4321)
        crow = (~nz0["mask"].any(1)).nonzero().squeeze(1).cpu()
        crow = crow[torch.randperm(len(crow), generator=gq)[:2048]].to(device)
        trow = nz0["te_mask"].any(1).nonzero().squeeze(1).cpu()
        trow = trow[torch.randperm(len(trow), generator=gq)[:4096]].to(device)
        tcrow = (~nz0["te_mask"].any(1)).nonzero().squeeze(1).cpu()
        tcrow = tcrow[torch.randperm(len(tcrow), generator=gq)[:4096]].to(device)
        for i in range(K):
            if not (noises[i]["mask"] == nz0["mask"]).all():
                sys.exit("묶음 안 run들의 오답 자리가 다릅니다 (규칙이 다르면 묶지 마세요)")
            if not resume:
                with (run_dirs[i] / "noise.csv").open("w", newline="") as f:
                    csv.DictWriter(f, NOISE_FIELDS).writeheader()

    @torch.no_grad()
    def predict_rows(x, Y, rows, want_logits=False):
        A = len(active)
        chunk = max(64, args.rows_budget // A)
        prs, lgs = [], []
        for s in range(0, len(rows), chunk):
            r = rows[s:s + chunk]
            with autocast():
                prs.append(vmap(f_pred, in_dims=(0, None))(params, x[r]))
                if want_logits:
                    lgs.append(vmap(f_logits, in_dims=(0, None, 0))(params, x[r], Y[active][:, r]).float())
        pr = torch.cat(prs, 1)
        return (pr, torch.cat(lgs, 1)) if want_logits else pr

    @torch.no_grad()
    def noise_eval():
        nz = noises[0]
        msk = nz["mask"][nrows]
        y_true = nz["y_true"][nrows]
        pr, lg = predict_rows(x_tr, Y_tr, nrows, want_logits=True)
        y_wrong = Y_tr[active][:, nrows]
        sel = msk.unsqueeze(0).expand_as(pr)
        tot = msk.sum().clamp(min=1)
        mem = ((pr == y_wrong) & sel).sum((1, 2)) / tot
        fixed = ((pr == y_true) & sel).sum((1, 2)) / tot
        lt = lg.gather(-1, y_true.unsqueeze(0).expand_as(pr).unsqueeze(-1))
        rank = (lg > lt).sum(-1) + 1
        prob = torch.softmax(lg, -1).gather(-1, y_true.unsqueeze(0).expand_as(pr).unsqueeze(-1)).squeeze(-1)
        r2 = ((rank == 2) & sel).sum((1, 2)) / tot
        pmean = (prob * sel).sum((1, 2)) / tot
        rmed = [rank[a][msk].float().median().item() for a in range(len(active))]
        pc = predict_rows(x_tr, Y_tr, crow)
        clean_ex = (pc == nz["y_true"][crow]).all(-1).float().mean(1)
        if len(trow):
            pt = predict_rows(x_te, Y_te, trow)
            tm, tw = nz["te_mask"][trow], nz["te_wrong"][trow]
            tt = tm.sum().clamp(min=1)
            spread = ((pt == tw) & tm).sum((1, 2)) / tt
            ttrue = ((pt == Y_te[active][:, trow]) & tm).sum((1, 2)) / tt
            ntok = int(nz["te_mask"].sum())
        else:
            spread = ttrue = torch.full((len(active),), math.nan, device=device); ntok = 0
        ptc = predict_rows(x_te, Y_te, tcrow)
        tclean = (ptc == Y_te[active][:, tcrow]).all(-1).float().mean(1)
        vals = dict(mem=mem, fixed=fixed, true_rank2=r2, true_prob=pmean, clean_rows_exact=clean_ex,
                    test_spread=spread, test_true_on_cells=ttrue, test_clean_rows_exact=tclean)
        vals = {k: v.tolist() for k, v in vals.items()}
        for slot, i in enumerate(active):
            row = {k: vals[k][slot] for k in vals}
            row.update(step=step, other=1 - row["mem"] - row["fixed"], true_rank_med=rmed[slot], test_cell_tokens=ntok)
            with (run_dirs[i] / "noise.csv").open("a", newline="") as f:
                csv.DictWriter(f, NOISE_FIELDS).writerow(
                    {k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in row.items()})

    def do_eval():
        if noises:
            noise_eval()
        a, b = eval_split(x_tr, Y_tr, i_tr, mask_tr), eval_split(x_te, Y_te, i_te)
        a = {k: v.tolist() for k, v in a.items()}
        b = {k: v.tolist() for k, v in b.items()}
        with torch.no_grad():
            sq = sum(v.detach().float().pow(2).flatten(1).sum(1) for v in params.values())
            pns = sq.sqrt().tolist()
            ens = params["tok_emb.weight"].detach().float().flatten(1).norm(dim=1).tolist()
        do_sym = args.sym_every > 0 and (step % args.sym_every == 0 or step in grid)
        for slot, i in enumerate(active):
            if do_sym:
                p_i = {k: v[slot].detach() for k, v in params.items()}
                s = probes[i](MemberView(base, p_i), autocast)
            else:
                s = nan_sym
            pn, en = pns[slot], ens[slot]
            row = dict(step=step, lr=lr_at(step), elapsed=round(elapsed, 2),
                       train_loss=a["loss"][slot], test_loss=b["loss"][slot],
                       train_tok=a["tok"][slot], test_tok=b["tok"][slot],
                       train_exact=a["exact"][slot], test_exact=b["exact"][slot],
                       train_exact_tf=a["exact_tf"][slot], test_exact_tf=b["exact_tf"][slot],
                       param_norm=pn, emb_norm=en,
                       sym_viol_tok=s["viol_tok"], sym_viol_seq=s["viol_seq"], sym_viol_tv=s["viol_tv"])
            with (run_dirs[i] / "metrics.csv").open("a", newline="") as f:
                csv.DictWriter(f, CSV_FIELDS).writerow(
                    {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
            st = states[i]
            if st["t_fit"] is None and row["train_exact"] >= args.fit_thr:
                st["t_fit"] = step
            if st["t_gen"] is None and row["test_exact"] >= args.gen_thr:
                st["t_gen"] = step
            if st["t_sym"] is None and not math.isnan(s["viol_tok"]) and s["viol_tok"] <= args.sym_thr:
                st["t_sym"] = step
            if args.stop_after_gen >= 0 and st["t_gen"] is not None and step >= st["t_gen"] + args.stop_after_gen:
                st["done"], st["stop_reason"] = True, "after_gen"
            if args.budget_mult > 0 and st["t_fit"] is not None and st["t_gen"] is None \
                    and step >= args.budget_mult * max(1, st["t_fit"]):
                st["done"], st["stop_reason"] = True, "budget"
            write_summary(i)
        if not args.quiet:
            ex_tr = " ".join(f"{v:.2f}" for v in a["exact"].tolist())
            ex_te = " ".join(f"{v:.2f}" for v in b["exact"].tolist())
            print(f"step {step:>7d}  active {len(active)}/{K}  train [{ex_tr}]  test [{ex_te}]  [{elapsed:.0f}s]",
                  flush=True)

    def compact():
        nonlocal params, opt, active
        keep = [slot for slot, i in enumerate(active) if not states[i]["done"]]
        if len(keep) == len(active):
            return
        if not keep:
            active = []
            return
        keep_t = torch.tensor(keep, dtype=torch.long, device=device)
        old_state = {name: opt.state.get(p, {}) for name, p in params.items()}
        params = {k: v.detach()[keep_t].clone().requires_grad_() for k, v in params.items()}
        opt = make_opt(params)
        for name, p in params.items():
            st = old_state[name]
            if st:
                opt.state[p] = {"step": st["step"].clone(),
                                "exp_avg": st["exp_avg"][keep_t].clone(),
                                "exp_avg_sq": st["exp_avg_sq"][keep_t].clone()}
        active = [active[s] for s in keep]

    if step == 0:
        do_eval()
    N = len(x_tr)
    t0 = time.time()
    while step < args.steps and active:
        lr = lr_at(step)
        for gr in opt.param_groups:
            gr["lr"] = lr
        Ya = Y_tr[active] if len(active) < K else Y_tr
        opt.zero_grad(set_to_none=True)
        if args.batch_size < 0:
            B, shared = N, True
        else:
            idx = torch.randint(0, N, (len(active), args.batch_size), generator=batch_gen).to(device)
            B, shared = args.batch_size, False
        chunk = max(64, args.rows_budget // len(active))
        for s in range(0, B, chunk):
            with autocast():
                if shared:
                    yb = Ya[:, s:s + chunk]
                    lg = vmap(f_logits, in_dims=(0, None, 0))(params, x_tr[s:s + chunk], yb)
                else:
                    ib = idx[:, s:s + chunk]
                    yb = torch.gather(Ya, 1, ib.unsqueeze(-1).expand(-1, -1, m))
                    lg = vmap(f_logits, in_dims=(0, 0, 0))(params, x_tr[ib], yb)
            if args.loss == "mse":
                lf = lg.float()
                ce = (lf - F.one_hot(yb, lf.shape[-1]).to(lf.dtype)).pow(2).sum(-1)
            else:
                ce = F.cross_entropy(lg.float().permute(0, 3, 1, 2), yb, reduction="none")
            if mask_tr is None:
                (ce.sum((1, 2)) / (B * m)).sum().backward()
            else:
                mb = mask_tr[s:s + chunk].unsqueeze(0) if shared else mask_tr[ib]
                (( ce * mb).sum((1, 2)) / (B * args.train_mask)).sum().backward()
        with torch.no_grad():
            w = wd[active]
            for v in params.values():
                v.mul_((1 - lr * w).view(-1, *([1] * (v.dim() - 1))).to(v.dtype))
        opt.step()
        step += 1
        if step % args.eval_every == 0 or step in grid or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.time() - t0
            do_eval()
            compact()
            t0 = time.time()
        if args.ckpt_log and (step in grid or (args.ckpt_dense and step % args.ckpt_dense == 0)
                              or (args.ckpt_early_every and step <= args.ckpt_early_until
                                  and step % args.ckpt_early_every == 0)):
            for slot, i in enumerate(active):
                save_member_ckpt(slot, i)
        if step % args.ckpt_every == 0:
            save_group_ckpt()
    elapsed += time.time() - t0
    for i in active:
        states[i]["stop_reason"] = states[i]["stop_reason"] or "max_steps"
        write_summary(i)
    save_group_ckpt()
    if not args.quiet:
        print(f"끝: step {step}, {elapsed:.0f}s, run {K}개")
    return [json.loads((rd / "summary.json").read_text()) for rd in run_dirs]


if __name__ == "__main__":
    main()
