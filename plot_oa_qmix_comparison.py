import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(values, window=5):
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / window
    # edge-padded moving average of same length
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_run(sacred_root: Path, alg: str, map_name: str):
    metrics_path = sacred_root / alg / map_name / "1" / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    steps = np.asarray(data["test_return_mean"]["steps"], dtype=float) / 1e6
    ret = np.asarray(data["test_return_mean"]["values"], dtype=float)
    won = np.asarray(data["test_battle_won_mean"]["values"], dtype=float) * 100.0
    draw = np.asarray(data["test_battle_draw_mean"]["values"], dtype=float) * 100.0
    kills = np.asarray(data["test_dead_enemies_mean"]["values"], dtype=float)
    return {
        "steps": steps,
        "return": ret,
        "win_rate": won,
        "undefeated": won + draw,
        "kills": kills,
    }


def main():
    epymarl_root = Path(__file__).resolve().parent
    sacred_root = epymarl_root / "results" / "sacred"
    out_path = epymarl_root / "results" / "oa_qmix_vs_qmix_maca_2m.png"

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8))

    styles = {
        "qmix": {"label": "QMIX Baseline", "color": "#4C78A8"},
        "oa_qmix": {"label": "OA-QMIX (Heuristic Profile)", "color": "#E45756"},
    }

    for row_idx, (map_name, max_enemies) in enumerate([("3v3", 3), ("5v5", 5)]):
        runs = {alg: load_run(sacred_root, alg, map_name) for alg in ("qmix", "oa_qmix")}

        panels = [
            ("return", f"MaCA {map_name} — Test Return", "Test Return", None),
            ("win_rate", f"MaCA {map_name} — Test Win Rate (%)", "Win Rate (%)", (-2, 75)),
            ("kills", f"MaCA {map_name} — Enemy Kills (/{max_enemies})", "Mean Enemy Kills", (-0.1, max_enemies + 0.3)),
        ]

        for col_idx, (metric_key, title, ylabel, ylim) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            for alg in ("qmix", "oa_qmix"):
                r = runs[alg]
                x = r["steps"]
                y = r[metric_key]
                y_sm = smooth(y, window=5)
                c = styles[alg]["color"]
                ax.plot(x, y, color=c, alpha=0.22, linewidth=1.1)
                ax.plot(x, y_sm, color=c, linewidth=2.3, label=f"{styles[alg]['label']} (5-pt MA)")
            if metric_key == "kills":
                ax.axhline(max_enemies, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Environment Steps (Millions)")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.0, 2.02)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, linestyle="--", alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower right", frameon=True)

    fig.suptitle(
        "OA-QMIX vs QMIX on MaCA UAV Combat Arena (3v3 & 5v5, 2M Steps, 40 Greedy Eval Episodes / 50k Steps)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
