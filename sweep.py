from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import rule_name


def is_done(run_dir: Path) -> bool:
    summ = run_dir / "summary.json"
    return summ.exists() and bool(json.loads(summ.read_text()).get("stop_reason"))


def multi_jobs(a, rules, extra):
    from train_multi import group_id, member_name
    runs = Path(a.runs_dir)
    groups_dir = runs / "_groups"
    in_group, jobs = set(), []
    for mfile in sorted(groups_dir.glob("*/members.json")) if groups_dir.exists() else []:
        spec = json.loads(mfile.read_text())
        names = [mb["name"] for mb in spec["members"]]
        in_group.update(names)
        if all(is_done(runs / nm) for nm in names):
            continue
        ckpt = mfile.parent / "ckpt_last.pt"
        if ckpt.exists():
            cmd = [sys.executable, "train_multi.py", "--resume", str(ckpt), "--quiet"]
            cmd += [x for x in extra if x != "--quiet"]
        else:
            cmd = spec["cmd"]
        jobs.append((mfile.parent.name, cmd, mfile.parent))

    pending = {}
    for data, model in itertools.product(a.data, a.models):
        for rule, wd, seed in itertools.product(rules, a.wds, a.seeds):
            nm = member_name(rule, model, data, wd, seed)
            if nm in in_group or is_done(runs / nm):
                continue
            pending.setdefault((data, model), []).append(dict(rule=rule, seed=seed, wd=wd, name=nm))
    for (data, model), mbs in pending.items():
        for s in range(0, len(mbs), a.group_size):
            chunk = mbs[s:s + a.group_size]
            gid = group_id([mb["name"] for mb in chunk])
            gdir = groups_dir / gid
            cmd = [sys.executable, "train_multi.py", "--data", data, "--model", model,
                   "--members", str(gdir / "members.json"), "--runs-dir", a.runs_dir, "--quiet"] + extra
            if not a.dry_run:
                gdir.mkdir(parents=True, exist_ok=True)
                (gdir / "members.json").write_text(json.dumps(dict(members=chunk, cmd=cmd), ensure_ascii=False,
                                                              indent=1))
            jobs.append((gid, cmd, gdir))
    return jobs


def _arg(cmd, flag, default=None):
    return cmd[cmd.index(flag) + 1] if flag in cmd else default


def estimate_cost(cmd):
    try:
        data = _arg(cmd, "--data")
        rows = json.loads((Path(data) / "meta.json").read_text())["train_rows"] if data else 1000
        bs = int(_arg(cmd, "--batch-size", -1))
        if bs > 0:
            rows = bs
        mem = _arg(cmd, "--members")
        k = len(json.loads(Path(mem).read_text())["members"]) if mem and Path(mem).exists() else 1
        d, L = int(_arg(cmd, "--d", 128)), int(_arg(cmd, "--layers", 2))
        steps = int(_arg(cmd, "--steps", 10000))
        return rows * k * (d / 128) ** 2 * (L / 2) * steps
    except Exception:
        return 0


class GpuLoad:
    PATH = Path("/tmp/jutam_gpu_load.json")

    def __enter__(self):
        import fcntl
        self.f = open(str(self.PATH) + ".lock", "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        try:
            self.d = json.loads(self.PATH.read_text())
        except Exception:
            self.d = {}
        import os
        def alive(pid):
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        self.d = {g: [p for p in ps if alive(p)] for g, ps in self.d.items()}
        return self

    def pick(self, gpus):
        return min(gpus, key=lambda g: (len(self.d.get(g, [])), gpus.index(g)))

    def add(self, gpu, pid):
        self.d.setdefault(gpu, []).append(pid)

    def __exit__(self, *exc):
        import fcntl
        self.PATH.write_text(json.dumps(self.d))
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()

def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="split 디렉터리 (여러 개 가능)")
    ap.add_argument("--rules", nargs="*", default=[], help="규칙 스펙 또는 프리셋 이름")
    ap.add_argument("--rules-file", help="sample_rules.py 출력 JSON")
    ap.add_argument("--models", nargs="+", default=["m0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--wds", nargs="+", type=float, default=[1.0])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--jobs", type=int, default=1, help="동시에 돌릴 학습 수")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--multi", action="store_true",
                    help="같은 데이터·모델의 run을 묶어 train_multi.py 한 프로세스로 학습")
    ap.add_argument("--group-size", type=int, default=32, help="--multi에서 한 묶음의 최대 run 수")
    ap.add_argument("--gpus", help="쓸 GPU 번호 목록 (예: 0,1,2,3) 또는 all. 주면 --jobs는 GPU 하나당 동시 실행 수")
    ap.add_argument("--shard", help="여러 서버에 나눠 돌릴 때 i/N (예: 0/3, 1/3, 2/3). 작업 목록의 i번째, i+N번째...만 실행")
    a = ap.parse_args(argv)

    rules = list(a.rules)
    if a.rules_file:
        rules += [r["spec"] for r in json.loads(Path(a.rules_file).read_text())]
    if not rules:
        sys.exit("--rules 또는 --rules-file이 필요합니다")

    jobs = multi_jobs(a, rules, extra) if a.multi else []
    for data, rule, model, wd, seed in ([] if a.multi else
                                        itertools.product(a.data, rules, a.models, a.wds, a.seeds)):
        name = f"{rule_name(rule)}_{model}_{Path(data).name}_wd{wd:g}_s{seed}"
        run_dir = Path(a.runs_dir) / name
        summ = run_dir / "summary.json"
        if summ.exists() and json.loads(summ.read_text()).get("stop_reason"):
            continue
        cmd = [sys.executable, "train.py", "--data", data, "--rule", rule, "--model", model,
               "--wd", str(wd), "--seed", str(seed), "--runs-dir", a.runs_dir, "--name", name, "--quiet"]
        if (run_dir / "ckpt_last.pt").exists():
            cmd = [sys.executable, "train.py", "--resume", str(run_dir / "ckpt_last.pt"), "--quiet"]
            cmd += [x for x in extra if x != "--quiet"]
        else:
            cmd += extra
        jobs.append((name, cmd, run_dir))

    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    gpus = None
    if a.gpus:
        if a.gpus == "all":
            import torch
            gpus = [str(g) for g in range(torch.cuda.device_count())]
        else:
            gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
        if not gpus:
            sys.exit("GPU를 찾지 못했습니다")
    print(f"실행할 {'묶음' if a.multi else 'run'} {len(jobs)}개" + (f", GPU {','.join(gpus)} × {a.jobs}" if gpus else ""))
    if a.dry_run:
        for _, cmd, _ in jobs:
            print(" ".join(cmd))
        return

    Path(a.runs_dir).mkdir(parents=True, exist_ok=True)
    manifest = Path(a.runs_dir) / "sweep_log.jsonl"

    import os

    jobs.sort(key=lambda j: -estimate_cost(j[1]))

    def run(job):
        name, cmd, run_dir = job
        log = Path(a.runs_dir) / f"{name}.log"
        env = dict(os.environ)
        t0 = time.time()
        with log.open("a") as f:
            if gpus:
                cap = int(os.environ.get("JUTAM_MAX_PER_GPU", "0") or 0)
                while True:
                    with GpuLoad() as load:
                        gpu = load.pick(gpus)
                        if cap <= 0 or len(load.d.get(gpu, [])) < cap:
                            env["CUDA_VISIBLE_DEVICES"] = gpu
                            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                                    cwd=Path(__file__).parent, env=env)
                            load.add(gpu, proc.pid)
                            break
                    time.sleep(15)
            else:
                gpu = None
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=Path(__file__).parent, env=env)
            rc = proc.wait()
        rec = dict(name=name, returncode=rc, seconds=round(time.time() - t0, 1), gpu=gpu, cmd=cmd)
        with manifest.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = "완료" if rc == 0 else f"실패({rc}) → {log}"
        print(f"[{status}] {name} {rec['seconds']}s" + (f" (GPU {gpu})" if gpu is not None else ""), flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=a.jobs * (len(gpus) if gpus else 1)) as ex:
        rcs = list(ex.map(run, jobs))
    fails = sum(1 for r in rcs if r != 0)
    print(f"끝: {len(rcs) - fails}개 성공, {fails}개 실패")


if __name__ == "__main__":
    main()
