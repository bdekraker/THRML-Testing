import glob
import json
import statistics as stats
from collections import defaultdict

import matplotlib.pyplot as plt


plt.style.use("dark_background")
BG_COLOR = "#141414"
FG_COLOR = "#f0f0f0"
BLOCK_COLORS = {
    "checkerboard": "#8dd3c7",
    "random": "#ffd92f",
}


def load_all(pattern="outputs/bench_*.json"):
    data = []
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                for row in loaded:
                    row["_path"] = path
                    data.append(row)
            else:
                loaded["_path"] = path
                data.append(loaded)
        except Exception:
            pass
    return data


def agg(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals:
        return None, None, []

    return stats.mean(vals), (stats.pstdev(vals) if len(vals) > 1 else 0.0), vals


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


def scatter_with_labels(ax, xs, ys, labels, colors):
    if not xs:
        return

    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_span = x_span if x_span != 0 else 1.0
    y_span = y_span if y_span != 0 else 1.0
    base_dx = 0.05 * x_span
    offset_cache = defaultdict(int)

    for x, y, lbl, color in zip(xs, ys, labels, colors):
        ax.scatter(x, y, color=color, edgecolors="#050505", s=70, linewidths=0.8, zorder=3)
        key = (round(x, 4), round(y, 4))
        idx = offset_cache[key]
        offset_cache[key] += 1
        sign = -1 if idx % 2 == 0 else 1
        magnitude = (idx // 2 + 1) * 0.03 * y_span
        dy = sign * magnitude
        text_x = x + base_dx
        text_y = y + dy
        ax.annotate(
            lbl,
            (x, y),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=8,
            color=color,
            va="center",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="#1f1f1f",
                ec=color,
                lw=0.6,
                alpha=0.85,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.8,
                alpha=0.7,
                shrinkA=4,
                shrinkB=2,
            ),
        )


def build_color(blocking):
    return BLOCK_COLORS.get(blocking, "#f781bf")


def aggregate_by(data, keys):
    grouped = defaultdict(list)
    for row in data:
        try:
            k = tuple(row[key] for key in keys)
        except KeyError:
            continue
        grouped[k].append(row)
    return grouped


def plot_tradeoff(data):
    xs = [row["samples_per_sec"] for row in data]
    ys = [row.get("ess_per_sec", 0.0) for row in data]
    labels = [
        f"{row['blocking']} | seed {row.get('seed', '?')} | sps={row.get('steps_per_sample')} | J={row.get('J'):.2f} | chains={row.get('num_chains')}"
        for row in data
    ]
    colors = [build_color(row["blocking"]) for row in data]

    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor=BG_COLOR)
    set_modern_axes(ax)
    scatter_with_labels(ax, xs, ys, labels, colors)
    ax.set_xlabel("samples/sec (↑ faster)")
    ax.set_ylabel("ESS/sec via AR(1) τ_int (↑ mixes better per wall-time)")
    ax.set_title("Blocking trade-off: throughput vs effective mixing (Ising 28×28, CPU)")
    ax.grid(True, alpha=0.25, color="#2a2a2a", linestyle="--", linewidth=0.7)
    ax.set_xlim(*pad_limits(xs))
    ax.set_ylim(*pad_limits(ys))
    fig.tight_layout()
    fig.savefig("outputs/bench_tradeoff_ess.png", dpi=150)
    print("Saved outputs/bench_tradeoff_ess.png")


def plot_tradeoff_lag(data):
    xs = [row["samples_per_sec"] for row in data]
    ys = [row["lag1_autocorr"] for row in data]
    labels = [
        f"{row['blocking']} | seed {row.get('seed', '?')} | sps={row.get('steps_per_sample')} | J={row.get('J'):.2f} | chains={row.get('num_chains')}"
        for row in data
    ]
    colors = [build_color(row["blocking"]) for row in data]

    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor=BG_COLOR)
    set_modern_axes(ax)
    scatter_with_labels(ax, xs, ys, labels, colors)
    ax.set_xlabel("samples/sec (↑ faster)")
    ax.set_ylabel("lag-1 autocorr (↓ mixes better)")
    ax.set_title("Blocking trade-off: raw lag-1 vs throughput (Ising 28×28, CPU)")
    ax.grid(True, alpha=0.25, color="#2a2a2a", linestyle="--", linewidth=0.7)
    ax.set_xlim(*pad_limits(xs))
    ax.set_ylim(*pad_limits(ys))
    fig.tight_layout()
    fig.savefig("outputs/bench_tradeoff.png", dpi=150)
    print("Saved outputs/bench_tradeoff.png")


def plot_sps_trends(data):
    grouped = aggregate_by(data, ("blocking", "steps_per_sample"))
    by_block = defaultdict(list)
    for (blocking, sps), rows in grouped.items():
        samples_mean, samples_std, _ = agg(rows, "samples_per_sec")
        ess_mean, ess_std, _ = agg(rows, "ess_per_sec")
        if samples_mean is None or ess_mean is None:
            continue
        by_block[blocking].append(
            (sps, samples_mean, samples_std, ess_mean, ess_std)
        )

    if not by_block:
        return

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2), facecolor=BG_COLOR)
    set_modern_axes(ax1)
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    ax2.tick_params(colors="#dcdcdc")
    ax2.yaxis.label.set_color("#dcdcdc")

    for blocking, rows in by_block.items():
        rows.sort(key=lambda r: r[0])
        sps_vals = [r[0] for r in rows]
        samples_mean = [r[1] for r in rows]
        samples_std = [r[2] for r in rows]
        ess_mean = [r[3] for r in rows]
        ess_std = [r[4] for r in rows]
        color = build_color(blocking)

        ax1.errorbar(
            sps_vals,
            samples_mean,
            yerr=samples_std,
            color=color,
            marker="o",
            label=f"{blocking} samples/sec",
            linestyle="-",
            linewidth=1.8,
            capsize=3,
        )
        ax2.errorbar(
            sps_vals,
            ess_mean,
            yerr=ess_std,
            color=color,
            marker="s",
            label=f"{blocking} ESS/sec",
            linestyle="--",
            linewidth=1.8,
            capsize=3,
        )

    ax1.set_xlabel("steps_per_sample (block Gibbs updates between reads)")
    ax1.set_ylabel("samples/sec", color=FG_COLOR)
    ax2.set_ylabel("ESS/sec", color="#dcdcdc")
    ax1.set_title("Sweep over steps_per_sample")
    ax1.grid(True, alpha=0.25, color="#2a2a2a", linestyle="--", linewidth=0.7)
    fig.tight_layout()

    lines_labels = ax1.get_legend_handles_labels()
    lines_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_labels[0] + lines_labels2[0],
        lines_labels[1] + lines_labels2[1],
        loc="upper right",
        fontsize=9,
    )

    fig.savefig("outputs/bench_sps_trends.png", dpi=150)
    print("Saved outputs/bench_sps_trends.png")


def plot_chain_scaling(data):
    grouped = aggregate_by(data, ("blocking", "num_chains"))
    by_block = defaultdict(list)
    for (blocking, chains), rows in grouped.items():
        ess_mean, ess_std, _ = agg(rows, "ess_per_sec")
        samp_mean, samp_std, _ = agg(rows, "samples_per_sec")
        if ess_mean is None or samp_mean is None:
            continue
        by_block[blocking].append((chains, ess_mean, ess_std, samp_mean, samp_std))

    if not by_block:
        return

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2), facecolor=BG_COLOR)
    set_modern_axes(ax1)
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    ax2.tick_params(colors="#dcdcdc")
    ax2.yaxis.label.set_color("#dcdcdc")

    for blocking, rows in by_block.items():
        rows.sort(key=lambda r: r[0])
        chains = [r[0] for r in rows]
        ess_mean = [r[1] for r in rows]
        ess_std = [r[2] for r in rows]
        samp_mean = [r[3] for r in rows]
        samp_std = [r[4] for r in rows]
        color = build_color(blocking)

        ax1.errorbar(
            chains,
            ess_mean,
            yerr=ess_std,
            color=color,
            marker="o",
            label=f"{blocking} ESS/sec",
            linestyle="-",
            linewidth=1.8,
            capsize=3,
        )
        ax2.errorbar(
            chains,
            samp_mean,
            yerr=samp_std,
            color=color,
            marker="s",
            label=f"{blocking} samples/sec",
            linestyle="--",
            linewidth=1.8,
            capsize=3,
        )

    ax1.set_xlabel("num_chains run in parallel")
    ax1.set_ylabel("ESS/sec", color=FG_COLOR)
    ax2.set_ylabel("samples/sec", color="#dcdcdc")
    ax1.set_title("Chain-level parallelism payoff")
    ax1.grid(True, alpha=0.25, color="#2a2a2a", linestyle="--", linewidth=0.7)
    fig.tight_layout()

    lines_labels = ax1.get_legend_handles_labels()
    lines_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_labels[0] + lines_labels2[0],
        lines_labels[1] + lines_labels2[1],
        loc="upper left",
        fontsize=9,
    )

    fig.savefig("outputs/bench_chain_scaling.png", dpi=150)
    print("Saved outputs/bench_chain_scaling.png")


def plot_J_sweep(data):
    grouped = aggregate_by(data, ("blocking", "J"))
    by_block = defaultdict(list)
    for (blocking, J_val), rows in grouped.items():
        ess_mean, ess_std, _ = agg(rows, "ess_per_sec")
        lag_mean, lag_std, _ = agg(rows, "lag1_autocorr")
        if ess_mean is None or lag_mean is None:
            continue
        by_block[blocking].append((J_val, ess_mean, ess_std, lag_mean, lag_std))

    if not by_block:
        return

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2), facecolor=BG_COLOR)
    set_modern_axes(ax1)
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    ax2.tick_params(colors="#dcdcdc")
    ax2.yaxis.label.set_color("#dcdcdc")

    for blocking, rows in by_block.items():
        rows.sort(key=lambda r: r[0])
        J_vals = [r[0] for r in rows]
        ess_mean = [r[1] for r in rows]
        ess_std = [r[2] for r in rows]
        lag_mean = [r[3] for r in rows]
        lag_std = [r[4] for r in rows]
        color = build_color(blocking)

        ax1.errorbar(
            J_vals,
            ess_mean,
            yerr=ess_std,
            color=color,
            marker="o",
            label=f"{blocking} ESS/sec",
            linestyle="-",
            linewidth=1.8,
            capsize=3,
        )
        ax2.errorbar(
            J_vals,
            lag_mean,
            yerr=lag_std,
            color=color,
            marker="s",
            label=f"{blocking} lag-1",
            linestyle="--",
            linewidth=1.8,
            capsize=3,
        )

    ax1.set_xlabel("Coupling strength J")
    ax1.set_ylabel("ESS/sec", color=FG_COLOR)
    ax2.set_ylabel("lag-1 autocorr", color="#dcdcdc")
    ax1.set_title("Effect of coupling strength on mixing")
    ax1.grid(True, alpha=0.25, color="#2a2a2a", linestyle="--", linewidth=0.7)
    fig.tight_layout()

    lines_labels = ax1.get_legend_handles_labels()
    lines_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_labels[0] + lines_labels2[0],
        lines_labels[1] + lines_labels2[1],
        loc="upper left",
        fontsize=9,
    )

    fig.savefig("outputs/bench_J_sweep.png", dpi=150)
    print("Saved outputs/bench_J_sweep.png")


def main():
    data = load_all()
    if not data:
        print("No outputs/bench_*.json found.")
        return

    plot_tradeoff(data)
    plot_tradeoff_lag(data)
    plot_sps_trends(data)
    plot_chain_scaling(data)
    plot_J_sweep(data)


if __name__ == "__main__":
    main()
