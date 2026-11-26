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


def load_5v5_run(sacred_root: Path, alg: str):
    metrics_path = sacred_root / alg / "5v5" / "1" / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    steps = np.asarray(data["test_return_mean"]["steps"], dtype=float) / 1e6
    ret = np.asarray(data["test_return_mean"]["values"], dtype=float)
    won = np.asarray(data["test_battle_won_mean"]["values"], dtype=float) * 100.0
    draw = np.asarray(data["test_battle_draw_mean"]["values"], dtype=float) * 100.0
    kills = np.asarray(data["test_dead_enemies_mean"]["values"], dtype=float)
    deaths = np.asarray(data["test_dead_allies_mean"]["values"], dtype=float)
    return {
        "steps": steps,
        "return": ret,
        "win_rate": won,
        "undefeated": won + draw,
        "kills": kills,
        "surviving_allies": 5.0 - deaths,
    }


def main():
    epymarl_root = Path(__file__).resolve().parent
    sacred_root = epymarl_root / "results" / "sacred"
    card_dir = (
        epymarl_root
        / "results"
        / "card"
        / "llm_oa_qmix_seed804783067_5v5_2026-11-25_21-15-48"
    )
    out_path = epymarl_root / "results" / "llm_oa_qmix_vs_baselines_5v5_2m.png"

    algs = [
        ("qmix", "QMIX Baseline", "#4C78A8"),
        ("oa_qmix", "OA-QMIX (Heuristic Profile)", "#E45756"),
        ("llm_oa_qmix", "LLM-OA-QMIX (deepseek-flash)", "#54A24B"),
    ]
    runs = {alg: load_5v5_run(sacred_root, alg) for alg, _, _ in algs}

    calls_path = card_dir / "llm_profile_calls.jsonl"
    call_vecs = []
    with calls_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                pr = rec.get("parsed_response") or {}
                vec = pr.get("behavior_vector") or []
                if len(vec) == 8:
                    call_vecs.append(vec)
    call_vecs = np.asarray(call_vecs, dtype=float)

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.2,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.2))

    panels = [
        (axes[0, 0], "win_rate", "MaCA 5v5 — Test Win Rate (%)", "Win Rate (%)", (-2, 80)),
        (
            axes[0, 1],
            "undefeated",
            "MaCA 5v5 — Undefeated Rate Win+Draw (%)",
            "Win + Draw (%)",
            (-2, 100),
        ),
        (axes[0, 2], "return", "MaCA 5v5 — Test Return", "Test Return", None),
        (axes[1, 0], "kills", "MaCA 5v5 — Mean Enemy Kills (/5)", "Enemy Kills", (0, 5.2)),
        (
            axes[1, 1],
            "surviving_allies",
            "MaCA 5v5 — Surviving Friendly UAVs (/5)",
            "Surviving Allies",
            (-0.1, 5.1),
        ),
    ]

    for ax, key, title, ylabel, ylim in panels:
        for alg, label, color in algs:
            r = runs[alg]
            x = r["steps"]
            y = r[key]
            y_sm = smooth(y, window=5)
            ax.plot(x, y, color=color, alpha=0.20, linewidth=1.1)
            ax.plot(x, y_sm, color=color, linewidth=2.3, label=f"{label} (5-pt MA)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Environment Steps (Millions)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 2.02)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.3)
        if key == "win_rate":
            ax.legend(loc="upper left", frameon=True)

    # Panel 6: LLM Opponent Profile 8D Behavior Vector Evolution across 73 API calls
    ax_vec = axes[1, 2]
    call_idx = np.arange(1, len(call_vecs) + 1)
    dim_specs = [
        (0, "focus_fire", "#E45756"),
        (1, "low_hp_chase", "#F58518"),
        (4, "ally_damage_share", "#B279A2"),
        (5, "survival_pressure", "#4C78A8"),
        (6, "grouping", "#54A24B"),
        (3, "kiting_distance", "#72B7B2"),
        (2, "target_switching", "#9D755D"),
    ]
    for dim_i, name, col in dim_specs:
        ax_vec.plot(
            call_idx,
            smooth(call_vecs[:, dim_i], window=5),
            label=name,
            color=col,
            linewidth=1.9,
        )
    ax_vec.set_title("LLM Opponent Profile Evolution (73 Calls)", fontweight="bold")
    ax_vec.set_xlabel("LLM Profile Refresh Call Index (every 128 eps)")
    ax_vec.set_ylabel("Inferred Score [0, 1] (5-call MA)")
    ax_vec.set_ylim(-0.02, 1.05)
    ax_vec.grid(True, linestyle="--", alpha=0.3)
    ax_vec.legend(loc="center right", ncol=2, fontsize=8.0, frameon=True)

    fig.suptitle(
        "MaCA 5v5 Benchmark: QMIX vs OA-QMIX vs LLM-OA-QMIX (2M Steps, Same QLearner Backbone)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 5v5 3-way comparison plot to {out_path}")


if __name__ == "__main__":
    main()
