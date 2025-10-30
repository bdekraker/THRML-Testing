import glob
import json
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt


plt.style.use("dark_background")
BG_COLOR = "#141414"
FG_COLOR = "#f0f0f0"


def load_all(pattern="outputs/bench_*.json"):
    data = []
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                d = json.load(f)
                d["_path"] = path
                data.append(d)
        except Exception:
            pass
    return data


def group_by_blocking(data):
    groups = {}
    for d in data:
        groups.setdefault(d["blocking"], []).append(d)
    return groups


def agg(rows, key):
    vals = [r[key] for r in rows]
    return stats.mean(vals), (stats.pstdev(vals) if len(vals) > 1 else 0.0), vals


def scatter_with_labels(ax, xs, ys, labels, colors):
    for x, y, lbl, color in zip(xs, ys, labels, colors):
        ax.scatter(x, y, color=color, edgecolors="#050505", s=70, linewidths=0.8)
        ax.text(x * 1.01 + 1e-6, y, lbl, color=color, fontsize=9, va="center")


def pad_limits(vals, frac=0.08):
    lo, hi = min(vals), max(vals)
    if lo == hi:
        if hi == 0:
            lo, hi = -1.0, 1.0
        else:
            lo, hi = lo * 0.99, hi * 1.01
    span = hi - lo
    return lo - frac * span, hi + frac * span


def set_modern_axes(ax):
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")
    ax.tick_params(colors=FG_COLOR)
    ax.xaxis.label.set_color(FG_COLOR)
    ax.yaxis.label.set_color(FG_COLOR)
    ax.title.set_color(FG_COLOR)


def main():
    data = load_all()
    if not data:
        print("No outputs/bench_*.json found.")
        return
    groups = group_by_blocking(data)

    xs_sps, ys_ess, lbls = [], [], []
    xs_sps_lag, ys_lag, lbls2 = [], [], []

    for blk, rows in groups.items():
        for r in rows:
            xs_sps.append(r["samples_per_sec"])
            ys_ess.append(r.get("ess_per_sec", 0.0))
            lbls.append(f"{blk} (seed {r.get('seed', '?')})")

            xs_sps_lag.append(r["samples_per_sec"])
            ys_lag.append(r["lag1_autocorr"])
            lbls2.append(f"{blk} (seed {r.get('seed', '?')})")

    color_palette = plt.cm.inferno(np.linspace(0.2, 0.9, max(len(xs_sps), 1)))

    fig1, ax1 = plt.subplots(figsize=(6.5, 4.0), facecolor=BG_COLOR)
    set_modern_axes(ax1)
    scatter_with_labels(ax1, xs_sps, ys_ess, lbls, color_palette[:len(xs_sps)])
    ax1.set_xlabel("samples/sec (↑ faster)")
    ax1.set_ylabel("ESS/sec via AR(1) τ_int (↑ mixes better per wall-time)")
    ax1.set_title("Blocking trade-off: throughput vs effective mixing (Ising 28×28, CPU)")
    ax1.grid(True, alpha=0.25, color="#2a2a2a", linestyle="--", linewidth=0.7)
    ax1.set_xlim(*pad_limits(xs_sps))
    ax1.set_ylim(*pad_limits(ys_ess))
    fig1.tight_layout()
    fig1.savefig("outputs/bench_tradeoff_ess.png", dpi=150)
    print("Saved outputs/bench_tradeoff_ess.png")

    fig2, ax2 = plt.subplots(figsize=(6.5, 4.0), facecolor=BG_COLOR)
    set_modern_axes(ax2)
    scatter_with_labels(ax2, xs_sps_lag, ys_lag, lbls2, color_palette[:len(xs_sps_lag)])
    ax2.set_xlabel("samples/sec (↑ faster)")
    ax2.set_ylabel("lag-1 autocorr (↓ mixes better)")
    ax2.set_title("Blocking trade-off: raw lag-1 vs throughput (Ising 28×28, CPU)")
    ax2.grid(True, alpha=0.25, color="#2a2a2a", linestyle="--", linewidth=0.7)
    ax2.set_xlim(*pad_limits(xs_sps_lag))
    ax2.set_ylim(*pad_limits(ys_lag))
    fig2.tight_layout()
    fig2.savefig("outputs/bench_tradeoff.png", dpi=150)
    print("Saved outputs/bench_tradeoff.png")


if __name__ == "__main__":
    main()
