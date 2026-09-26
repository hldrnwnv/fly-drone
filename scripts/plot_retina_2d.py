"""Render the issue #1 infographic from the saved map and trial receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(run_dir: Path) -> None:
    map_report = json.loads((run_dir / "map.json").read_text(encoding="utf-8"))
    trials = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["trials"]
    grating_trials = json.loads((run_dir / "grating" / "results.json").read_text(encoding="utf-8"))["trials"]
    seeds = sorted({t["seed"] for t in trials})
    n_seeds = len(seeds)
    lookup = {(t["seed"], t["mode"], t["eye"], t["stimulus"]): t for t in trials}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "retina-2d"})
    fig = plt.figure(figsize=(13, 12), facecolor="#f7f9fc")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.9, 2.2, 4.1],
                           hspace=0.62, wspace=0.30, left=0.11, right=0.97,
                           top=0.95, bottom=0.08)
    title = fig.add_subplot(grid[0, :])
    title.axis("off")
    title.text(0, 0.88, "MaleCNS v1.0: проверка двумерного зрительного входа",
               fontsize=18, fontweight="bold", color="#102a43")
    title.text(0, 0.61, f"Таблица колонок → карта рецепторов → 2D / прежний 1D вход → одинаковые кадры и {n_seeds} новых зерен",
               fontsize=11, color="#334e68")
    title.text(0, 0.36, "Условия: пустой кадр, равномерная яркость, неподвижная полоса, движение ← и →; каждый глаз отдельно",
               fontsize=10, color="#52667a")
    grating_lookup = {(t["seed"], t["stimulus"]): t for t in grating_trials
                      if t["mode"] == "2d" and t["eye"] == "R"}
    consistent = []
    for contrast in ("high", "low"):
        count = sum(grating_lookup[(seed, f"left_to_right_{contrast}")]["T4_T5_spikes"]["T4b_R"] >
                    grating_lookup[(seed, f"right_to_left_{contrast}")]["T4_T5_spikes"]["T4b_R"]
                    for seed in seeds)
        consistent.append(count)
    bar_count = sum(lookup[(seed, "2d", "R", "left_to_right")]["T4_T5_spikes"]["T4b_R"] >
                    lookup[(seed, "2d", "R", "right_to_left")]["T4_T5_spikes"]["T4b_R"]
                    for seed in seeds)
    title.text(0, 0.10,
               f"Проверка T4b_R: полоса {bar_count}/{n_seeds}, решётка с общим первым кадром {consistent[0]}/{n_seeds} и {consistent[1]}/{n_seeds} (высокий/низкий контраст)",
               fontsize=10, fontweight="bold", color="#9c2f20")

    coverage = fig.add_subplot(grid[1, 0], facecolor="white")
    palette = {"direct": "#1971c2", "inferred_l1": "#e67700", "unmapped": "#868e96"}
    labels = {"direct": "R7/R8: источник", "inferred_l1": "R1–R6: через L1", "unmapped": "без позиции"}
    for row, eye in enumerate("LR"):
        offset = 0
        for kind in ("direct", "inferred_l1", "unmapped"):
            amount = map_report["counts"][eye][kind]
            coverage.barh(row, amount, left=offset, height=0.55,
                          color=palette[kind], label=labels[kind] if row == 0 else None)
            if amount > 200:
                coverage.text(offset + amount / 2, row, str(amount), va="center",
                              ha="center", color="white", fontweight="bold", fontsize=10)
            offset += amount
    coverage.set_yticks([0, 1], ["Левый глаз", "Правый глаз"])
    coverage.invert_yaxis()
    coverage.set_xlabel("Фоторецепторы, шт.")
    coverage.set_title("Покрытие опубликованной картой", loc="left", fontweight="bold")
    coverage.legend(loc="lower left", bbox_to_anchor=(0, -0.38), ncol=2, frameon=False, fontsize=8)
    coverage.spines[["top", "right"]].set_visible(False)

    totals = fig.add_subplot(grid[1, 1], facecolor="white")
    conditions = ["blank", "bright", "stationary", "left_to_right", "right_to_left"]
    condition_labels = ["Пусто", "Ярко", "Статично", "Слева → вправо", "Справа → влево"]
    x = np.arange(len(conditions))
    for eye, shift, color, marker in (("L", -0.08, "#1971c2", "o"),
                                      ("R", 0.08, "#e67700", "s")):
        values = [[sum(n for k, n in lookup[(seed, "2d", eye, stimulus)]["T4_T5_spikes"].items()
                       if k.endswith("_" + eye)) for seed in seeds] for stimulus in conditions]
        means = np.array([np.mean(v) for v in values])
        sds = np.array([np.std(v, ddof=1) for v in values])
        totals.errorbar(x + shift, means, yerr=sds, fmt=marker, color=color,
                        capsize=3, label=f"{eye}: среднее ± SD")
    totals.set_xticks(x, condition_labels, rotation=20, ha="right")
    totals.set_ylabel("Спайки T4/T5 за 50 тактов")
    totals.set_title("2D: контрольные условия", loc="left", fontweight="bold")
    totals.legend(frameon=False, fontsize=8)
    totals.spines[["top", "right"]].set_visible(False)

    direction = fig.add_subplot(grid[2, :], facecolor="white")
    groups = [f"{kind} {eye}" for eye in "LR" for kind in
              ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")]
    y = np.arange(len(groups))
    for mode, shift, color, marker in (("2d", -0.15, "#1971c2", "o"),
                                       ("1d", 0.15, "#c05b00", "s")):
        means, sds = [], []
        for label in groups:
            kind, eye = label.split()
            key = f"{kind}_{eye}"
            values = [lookup[(seed, mode, eye, "left_to_right")]["T4_T5_spikes"][key] -
                      lookup[(seed, mode, eye, "right_to_left")]["T4_T5_spikes"][key]
                      for seed in seeds]
            means.append(np.mean(values))
            sds.append(np.std(values, ddof=1))
        direction.errorbar(means, y + shift, xerr=sds, fmt=marker,
                           color=color, capsize=2, markersize=4, label=f"{mode.upper()}: среднее ± SD")
    direction.axvline(0, color="#52667a", linewidth=1)
    direction.set_yticks(y, groups)
    direction.invert_yaxis()
    direction.set_xlabel("Спайки слева→вправо минус справа→влево за 50 тактов")
    direction.set_title(f"Разница направлений по типу клеток и глазу; n={n_seeds} парных зерен",
                        loc="left", fontweight="bold")
    direction.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2)
    direction.grid(axis="x", color="#e9ecef")
    direction.spines[["top", "right"]].set_visible(False)
    fig.text(0.11, 0.02,
             "Источник: MaleCNS v1.0, таблица optic-column-type-assignments; данные и код: runs/retina-2d/. "
             "Координаты в кадре нормированы по глазу, углы зрения не калиброваны.",
             fontsize=8, color="#52667a")
    fig.savefig(run_dir / "infographic.png", dpi=180, facecolor=fig.get_facecolor())
    svg = run_dir / "infographic.svg"
    fig.savefig(svg, facecolor=fig.get_facecolor())
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/retina-2d"))
    render(parser.parse_args().run_dir)
