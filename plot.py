from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#8a6d3b"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"


def load(run_dir: Path):
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    col = lambda k: [float(r[k]) if r[k] not in ("", "nan") else float("nan") for r in rows]
    summ = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    return dict(step=[max(1, int(r["step"])) for r in rows], col=col, summary=summ)


def style(ax, title, xlog, ylog=False):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("step", color=MUTED, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="grokking.png")
    ap.add_argument("--linear-x", action="store_true")
    ap.add_argument("--labels", nargs="*", help="run별 범례 이름 (기본: 디렉터리 이름)")
    a = ap.parse_args()
    dirs = sorted({Path(p) for pat in a.runs for p in glob.glob(pat) if (Path(p) / "metrics.csv").exists()})
    if not dirs:
        raise SystemExit("metrics.csv가 있는 run이 없습니다")
    if len(dirs) > len(SERIES):
        print(f"run {len(dirs)}개 중 앞 {len(SERIES)}개 그림")
        dirs = dirs[: len(SERIES)]
    xlog = not a.linear_x
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if fam in names:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_acc, ax_loss, ax_norm, ax_sym = axes.ravel()
    for i, d in enumerate(dirs):
        r = load(d)
        c = SERIES[i]
        label = a.labels[i] if a.labels and i < len(a.labels) else d.name
        ax_acc.plot(r["step"], r["col"]("train_exact"), color=c, lw=1.5, ls=":")
        ax_acc.plot(r["step"], r["col"]("test_exact"), color=c, lw=2, label=label)
        ax_loss.plot(r["step"], r["col"]("train_loss"), color=c, lw=1.5, ls=":")
        ax_loss.plot(r["step"], r["col"]("test_loss"), color=c, lw=2)
        ax_norm.plot(r["step"], r["col"]("param_norm"), color=c, lw=2)
        ax_sym.plot(r["step"], r["col"]("sym_viol_tok"), color=c, lw=2)
        tg = r["summary"].get("t_gen")
        if tg:
            ax_acc.axvline(max(1, tg), color=c, lw=1, ls="--", alpha=0.6)
    style(ax_acc, "수열 전체 일치 정확도 (실선 test, 점선 train, 세로선 t_gen)", xlog)
    style(ax_loss, "loss (실선 test, 점선 train)", xlog, ylog=True)
    style(ax_norm, "전체 weight norm", xlog)
    style(ax_sym, "대칭 위반 (어긋난 출력 칸 비율)", xlog)
    ax_acc.set_ylim(-0.02, 1.02)
    ax_sym.set_ylim(-0.02, 1.02)
    ax_acc.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(a.out, dpi=150)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
