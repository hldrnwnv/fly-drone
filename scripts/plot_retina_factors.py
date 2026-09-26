"""Draw issue #8 infographic from the audited result receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAP_LABELS = {"original": "Исходная", "direct_only": "Только R7/R8",
              "shuffle_801": "Перестановка 801", "shuffle_802": "Перестановка 802"}
BLOCK_LABELS = {"main": "База: 4, 0", "phase_only": "Фаза: 4, ¼",
                "period_only": "Период: 3, 0", "holdout": "Оба: 3, ¼"}


def render(run_dir: Path) -> None:
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows = data["comparisons"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "retina-factorial"})
    fig = plt.figure(figsize=(14, 11), facecolor="#f7f9fc")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.9, 2.7, 2.5], hspace=0.48,
                           wspace=0.31, left=0.09, right=0.97, top=0.95, bottom=0.09)
    title = fig.add_subplot(grid[0, :])
    title.axis("off")
    title.text(0, 0.9, "MaleCNS v1.0: карта или скорость создают ответ T4b_R?",
               fontsize=17, fontweight="bold", color="#102a43")
    title.text(0, 0.56, "Вход: кадры 64×64 → 6006 рецепторов → фиксированный flybrain → спайки T4/T5",
               fontsize=11, color="#334e68")
    title.text(0, 0.25, "Контроли: R7/R8 без R1–R6, две перестановки R1–R6, 1D, пустой и неподвижный кадр; глаза отдельно",
               fontsize=10, color="#334e68")

    maps_ax = fig.add_subplot(grid[1, :], facecolor="white")
    colors = {"original": "#166a8f", "direct_only": "#c65c2e",
              "shuffle_801": "#7050a1", "shuffle_802": "#4c7c41"}
    markers = {"original": "o", "direct_only": "s", "shuffle_801": "^",
               "shuffle_802": "D"}
    offsets = {"original": -0.24, "direct_only": -0.08,
               "shuffle_801": 0.08, "shuffle_802": 0.24}
    for contrast_index, contrast in enumerate((95, 40)):
        for speed_index, speed in enumerate((1, 2, 4)):
            x = contrast_index * 4 + speed_index
            for mode in MAP_LABELS:
                values = [row["delta_spikes"]["T4b_R"] for row in rows
                          if row["split"] == "held_out" and row["block"] == "main"
                          and row["map"] == mode and row["eye"] == "R"
                          and row["contrast"] == contrast and row["speed"] == speed]
                maps_ax.errorbar(x + offsets[mode], np.mean(values),
                                 yerr=np.std(values, ddof=1), fmt=markers[mode],
                                 capsize=3, color=colors[mode], markersize=5,
                                 label=MAP_LABELS[mode] if x == 0 else None)
    maps_ax.axhline(0, color="#52667a", linewidth=1)
    maps_ax.axvline(3.5, color="#cbd5e1", linewidth=1)
    maps_ax.set_xticks([0, 1, 2, 4, 5, 6],
                      ["1", "2", "4", "1", "2", "4"])
    maps_ax.text(1, 1.04, "Контраст ±95", transform=maps_ax.get_xaxis_transform(),
                 ha="center", fontweight="bold")
    maps_ax.text(5, 1.04, "Контраст ±40", transform=maps_ax.get_xaxis_transform(),
                 ha="center", fontweight="bold")
    maps_ax.set_xlabel("Скорость, периодов решётки за 50 кадров (1 с)")
    maps_ax.set_ylabel("Вправо − влево, спайков T4b_R / 50 тактов")
    maps_ax.set_title("Основные проверочные кадры: среднее ± SD, n=8 парных зерен 35–42",
                      loc="left", fontweight="bold", pad=20)
    maps_ax.legend(ncol=4, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, -0.3))
    maps_ax.spines[["top", "right"]].set_visible(False)
    maps_ax.grid(axis="y", color="#e9ecef")

    held = fig.add_subplot(grid[2, 0], facecolor="white")
    blocks = ("main", "phase_only", "period_only", "holdout")
    for index, block in enumerate(blocks):
        values = [row["delta_spikes"]["T4b_R"] for row in rows
                  if row["split"] == "held_out" and row["block"] == block
                  and row["map"] == "original" and row["eye"] == "R"
                  and row["contrast"] == 95 and row["speed"] == 1]
        held.scatter(np.full(len(values), index), values, marker="o",
                     facecolors="none", edgecolors="#166a8f", s=48)
        held.plot(index, np.mean(values), marker="s", color="#b33a3a", markersize=8)
    held.axhline(0, color="#52667a", linewidth=1)
    held.set_xticks(range(4), [BLOCK_LABELS[b] for b in blocks], rotation=20, ha="right")
    held.set_ylabel("Вправо − влево, спайков T4b_R / 50 тактов")
    held.set_title("Отложенные фаза и период; исходная карта, ±95, скорость 1",
                   loc="left", fontweight="bold")
    held.text(0.02, 0.98, "○ отдельное зерно   ■ среднее; n=8",
              transform=held.transAxes, va="top", fontsize=9)
    held.spines[["top", "right"]].set_visible(False)
    held.grid(axis="y", color="#e9ecef")

    coverage = fig.add_subplot(grid[2, 1], facecolor="white")
    with np.load(run_dir / "maps.npz") as saved:
        provenance = saved["provenance"]
    numbers = [int(np.sum(provenance == label)) for label in
               ("direct", "inferred_l1", "unmapped")]
    bars = coverage.barh(range(3), numbers, color=("#166a8f", "#c65c2e", "#888e99"))
    coverage.bar_label(bars, padding=4)
    coverage.set_yticks(range(3), ["R7/R8: прямая карта", "R1–R6: через L1",
                                  "Без координат"])
    coverage.invert_yaxis()
    coverage.set_xlabel("Число рецепторов из 6006")
    coverage.set_title("Что именно меняли в факторе карты", loc="left",
                       fontweight="bold")
    coverage.text(0.02, 0.05, "Перестановки сохраняют покадровое\nраспределение входа R1–R6 внутри глаза.",
                  transform=coverage.transAxes, fontsize=9, color="#334e68")
    coverage.spines[["top", "right"]].set_visible(False)

    fig.text(0.09, 0.025,
             f"Источник: MaleCNS v1.0; runs/retina-factorial/results.json; {data['trial_count']} испытаний, "
             "50 кадров × 20 мс. Искусственная проекция и LIF-модель; не полёт и не физиология живой мухи.",
             fontsize=8, color="#52667a")
    fig.savefig(run_dir / "infographic.png", dpi=180, facecolor=fig.get_facecolor())
    svg = run_dir / "infographic.svg"
    fig.savefig(svg, facecolor=fig.get_facecolor())
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/retina-factorial"))
    render(parser.parse_args().run_dir)
