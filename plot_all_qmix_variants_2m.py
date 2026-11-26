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
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_run(sacred_root: Path, alg: str, map_name: str):
    metrics_path = sacred_root / alg / map_name / "1" / "metrics.json"
    if not metrics_path.exists():
        return None
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
    out_path = epymarl_root / "results" / "all_qmix_variants_maca_2m.png"

    alg_specs = [
        ("qmix", "QMIX Baseline", "#4C78A8", "--", 1.9),
        ("oa_qmix", "OA-QMIX (Heuristic)", "#F58518", "--", 1.9),
        ("llm_oa_qmix", "LLM-OA-QMIX (QLearner)", "#54A24B", "-", 2.1),
        ("Finetuned-QMIX", "Finetuned-QMIX (PyMARL2)", "#72B7B2", "-.", 2.2),
        (
            "Finetuned-LLM-OA-QMIX",
            "Finetuned-LLM-OA-QMIX (Ours)",
            "#E45756",
            "-",
            2.6,
        ),
    ]

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(17.5, 13.0))

    scenarios = [("3v3", 3), ("5v5", 5), ("10v10", 10)]
    for row_idx, (map_name, _) in enumerate(scenarios):
        panels = [
            ("return", f"MaCA {map_name} — Test Return", "Test Return", None),
            (
                "win_rate",
                f"MaCA {map_name} — Test Win Rate (%)",
                "Win Rate (%)",
                (-2, 82),
            ),
            (
                "undefeated",
                f"MaCA {map_name} — Undefeated Win+Draw (%)",
                "Win + Draw (%)",
                (-2, 100),
            ),
        ]
        for col_idx, (metric_key, title, ylabel, ylim) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            for alg, label, color, ls, lw in alg_specs:
                run = load_run(sacred_root, alg, map_name)
                if run is None:
                    continue
                x = run["steps"]
                y = run[metric_key]
                y_sm = smooth(y, window=5)
                ax.plot(x, y, color=color, alpha=0.14, linewidth=0.9)
                ax.plot(
                    x,
                    y_sm,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    label=f"{label} (5-pt MA)",
                )
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Environment Steps (Millions)")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.0, 2.02)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=10.0,
        bbox_to_anchor=(0.5, 0.008),
    )
    fig.suptitle(
        "MaCA UAV Combat Arena All-Scenario Benchmark (3v3 / 5v5 / 10v10, 2M Steps)",
        fontsize=14.5,
        fontweight="bold",
        y=0.988,
    )
    fig.tight_layout(rect=(0, 0.042, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved master plot to {out_path}")


if __name__ == "__main__":
    main()
