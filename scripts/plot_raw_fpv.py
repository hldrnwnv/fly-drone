"""Render issue #2 infographic exclusively from saved benchmark receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(run_dir: Path) -> None:
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    names = ["visual", "connectome", "degree_shuffled_0", "degree_shuffled_1",
             "degree_shuffled_2"]
    labels = ["Visual detector", "MaleCNS", "Shuffled 0", "Shuffled 1", "Shuffled 2"]
    summary = result["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.91))
    y = np.arange(len(names))
    width = 0.36
    nominal = [summary[name]["by_scenario"]["nominal"]["gate_passes"] for name in names]
    dropout = [summary[name]["by_scenario"]["dropout_30"]["gate_passes"] for name in names]
    n_per_scenario = summary["visual"]["by_scenario"]["nominal"]["flights"]
    axes[0, 0].barh(y - width / 2, nominal, height=width, facecolor="white",
                    edgecolor="black", hatch="///", label="Normal")
    axes[0, 0].barh(y + width / 2, dropout, height=width, facecolor="0.65",
                    edgecolor="black", hatch="...", label="30% frame dropout")
    axes[0, 0].set(yticks=y, yticklabels=labels, xlabel=f"Gate passes / {n_per_scenario} flights",
                   xlim=(0, n_per_scenario + 1), title="Closed loop, matched new starts")
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(loc="lower right", fontsize=9)
    for i, (a, b) in enumerate(zip(nominal, dropout)):
        axes[0, 0].text(a + 0.08, i - width / 2, str(a), va="center", fontsize=9)
        axes[0, 0].text(b + 0.08, i + width / 2, str(b), va="center", fontsize=9)

    accuracy = [summary[name]["open_loop_direction_correct"] /
                summary[name]["open_loop_direction_total"] for name in names]
    axes[0, 1].barh(y, accuracy, color="white", edgecolor="black", hatch="xx")
    axes[0, 1].set(yticks=y, yticklabels=labels, xlabel="Correct turn direction / eligible decisions",
                   xlim=(0, 1.14), title="Open loop, identical saved RGB frames")
    axes[0, 1].invert_yaxis()
    for i, name in enumerate(names):
        axes[0, 1].text(accuracy[i] + 0.015, i,
                        f'{summary[name]["open_loop_direction_correct"]}/'
                        f'{summary[name]["open_loop_direction_total"]}',
                        va="center", fontsize=9)

    p95 = [summary[name]["closed_loop_decision_p95_ms"] for name in names]
    axes[1, 0].barh(y, p95, facecolor="0.8", edgecolor="black", hatch="\\\\")
    axes[1, 0].set(yticks=y, yticklabels=labels, xlabel="95th-percentile decision time (ms)",
                   xlim=(0, max(p95) * 1.3), title="Camera + controller; 400 ms deadline")
    axes[1, 0].invert_yaxis()
    for i, value in enumerate(p95):
        axes[1, 0].text(value + max(p95) * 0.015, i, f"{value:.1f}", va="center", fontsize=9)

    keys = [key.split("/", 1)[1] for key in result["runs"] if key.startswith("visual/")]
    matrix = np.array([[int(result["runs"][f"{name}/{key}"]["reached"]) for key in keys]
                       for name in names])
    axes[1, 1].imshow(matrix, cmap="Greys_r", vmin=0, vmax=1, aspect="auto")
    axes[1, 1].set(yticks=y, yticklabels=labels, xticks=np.arange(len(keys)),
                   xticklabels=[f'{"N" if key.startswith("nominal") else "D"}'
                                f'{"L" if "/left/" in key else "R"}'
                                f'{key.rsplit("/", 1)[-1]}' for key in keys],
                   xlabel="Scenario / side / new start", title="Each flight: ✓ pass, × fail")
    axes[1, 1].tick_params(axis="x", labelrotation=90, labelsize=8)
    for row in range(len(names)):
        for col in range(len(keys)):
            axes[1, 1].text(col, row, "✓" if matrix[row, col] else "×",
                            ha="center", va="center", fontsize=10,
                            color="black" if matrix[row, col] else "white")
    protocol = result["protocol"]
    fig.text(0.02, 0.015,
             "Input: 320×180 RGB; one coloured gate visible. Controls: fixed visual detector, "
             "MaleCNS v1.0, three independently shuffled graphs.\n"
             f"Neural readouts: same training frames, {protocol['ticks_per_frame']} ticks/frame; "
             f"{protocol['train_count']} train + {protocol['validation_count']} validation images. "
             "N=normal, D=30% dropout; L/R=gate side. Altitude and speed assisted.",
             fontsize=9, va="bottom")
    fig.savefig(run_dir / "infographic.png", dpi=180)
    svg = run_dir / "infographic.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/raw-fpv"))
    plot(parser.parse_args().run_dir)
