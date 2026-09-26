"""Write a compact human-readable receipt for a stimulation search."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def render_report(result: dict, checkpoint: dict) -> str:
    if checkpoint["evaluations"] != result["evaluations"]:
        raise ValueError("checkpoint and results describe different evaluation counts")
    group_best: dict[str, dict] = {}
    for item in checkpoint["history"]:
        group = item["candidate"]["group"]
        if group not in group_best or item["train"]["score"] > group_best[group]["train"]["score"]:
            group_best[group] = item
    leaders = sorted(group_best.values(), key=lambda x: x["train"]["score"], reverse=True)[:10]
    lines = ["# Stimulation search report", "",
             f"- Mixed differential evolution: {result['evaluations']} candidate evaluations; "
             f"{result['neural_runs_train']} stochastic neural runs for training.",
             f"- Same frozen MaleCNS connectome in every run; seeds change model noise, "
             "not anatomical wiring.",
             "- Neural selection used motor-neuron spike asymmetry. Physics selection used "
             "separate starts with the same external altitude correction.",
             "- The final physics starts and moving-target chase were held out from evolution.",
             "- Polarity +1 excites; -1 suppresses. Side rule +1 stimulates right cells "
             "for negative roll rate and left cells for positive roll rate.",
             "", "## Best training variants by cell group", "",
             "| Group | Polarity | Side rule | Gain | Duty | Neural score | Pairs |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for item in leaders:
        c = item["candidate"]
        t = item["train"]
        lines.append(f"| `{c['group']}` | {c['polarity']:+d} | {c['orientation']:+d} | "
                     f"{c['gain']:.2f} | {c['duty']:.2f} | {t['score']:.2f} | "
                     f"{t['opposite_sign_pairs']}/{t['seeds']} |")
    lines += ["", "## Held-out neural response", "",
              "| Group | Polarity | Side rule | Score | Opposite-sign pairs |",
              "|---|---:|---:|---:|---:|"]
    for entry in result["neural_heldout"]:
        c, r = entry["candidate"], entry["result"]
        lines.append(f"| `{c['group']}` | {c['polarity']:+d} | {c['orientation']:+d} | "
                     f"{r['score']:.2f} | {r['opposite_sign_pairs']}/{r['seeds']} |")
    lines += ["", "## MuJoCo roll perturbations", "",
              "The four selection starts chose the final configuration. The eight test starts "
              "below were not used for evolution or selection.", "",
              "| Condition | Survived / 8 | Mean duration (s) | Mean absolute roll (rad) |",
              "|---|---:|---:|---:|"]
    for name, value in result["physics_test"].items():
        lines.append(f"| {name.replace('_', ' ')} | {value['survived']}/8 | "
                     f"{value['mean_duration_s']:.2f} | {value['mean_abs_roll_rad']:.3f} |")
    selected = result["physics_test"]["winner"]["runs"]
    reference = result["physics_test"]["reference_bundle"]["runs"]
    differences = np.asarray([
        a["abs_roll_integral_rad_s"] / a["duration_s"] -
        b["abs_roll_integral_rad_s"] / b["duration_s"]
        for a, b in zip(selected, reference)], dtype=float)
    observed = float(np.mean(differences))
    null = [float(np.mean(differences * np.asarray(signs)))
            for signs in itertools.product((-1, 1), repeat=len(differences))]
    one_sided_p = (sum(x <= observed for x in null) + 1) / (len(null) + 1)
    lines += ["", f"The evolved bundle had lower roll error than the fixed bundle "
              f"on {int(np.sum(differences < 0))}/8 starts. Its mean paired difference "
              f"was {observed:+.3f} rad; the exploratory exact sign-flip "
              f"one-sided p-value is {one_sided_p:.2f}. This small test does not establish "
              "a reliable improvement over the fixed bundle."]
    winner = result["winner"]
    chase = result["chase_transfer"]
    lines += ["", "## Selected interface and full-flight transfer", "",
              f"- Cell group: `{winner['group']}`; polarity {winner['polarity']:+d}; "
              f"side rule {winner['orientation']:+d}; gain {winner['gain']:.3f}; "
              f"pulse duty {winner['duty']:.3f}.",
              f"- Moving-target scene without altitude assist: **{chase['status']}** "
              f"after {chase['duration']:.2f} s; tracked={chase['tracked']}.",
              "- The gyro-to-cell rule and motor-to-rotor map are engineered. Survival in "
              "the assisted roll task does not establish autonomous flight or biological fidelity.",
              "- `results.json` contains every neural trial, physics decision, and chase trace; "
              "`checkpoint.json` contains the complete search history and resumable population.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path("runs/neural-stimulation-search"))
    args = parser.parse_args()
    result = json.loads((args.directory / "results.json").read_text())
    checkpoint = json.loads((args.directory / "checkpoint.json").read_text())
    path = args.directory / "report.md"
    path.write_text(render_report(result, checkpoint), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
