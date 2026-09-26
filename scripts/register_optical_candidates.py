"""Search provisional hex-grid registrations without looking at neural responses.

.venv/bin/python scripts/register_optical_candidates.py --fetch
All accepted maps are hypotheses: shape and dorsal-rim checks are not measured
MaleCNS optical-ray residuals. Eye-to-camera pose remains uncalibrated.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import warnings
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import rdata
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "optical-map-13-candidates"
import matplotlib.pyplot as plt

from audit_optical_map import (MALE_FILES, MAP_SHA256, SOURCE_FILES,
                               canonicalize_svg, checked_hash, checked_source)

NEIGHBORS = {(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)}
OVERLAP_FRACTION = 0.95  # exploratory topology filter, not a confidence bound
SHIFT_RADIUS = 8  # around the source/target grid-centroid difference
ANNOTATIONS_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/"
                   "connectome-data/flat-connectome/"
                   "body-annotations-male-cns-v1.0-minconf-0.5.feather")
ANNOTATIONS_SHA256 = "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"


def lattice_symmetries() -> list[np.ndarray]:
    matrices = []
    for entries in itertools.product((-1, 0, 1), repeat=4):
        matrix = np.asarray(entries, dtype=np.int64).reshape(2, 2)
        if abs(round(np.linalg.det(matrix))) == 1 and {
            tuple(matrix @ np.asarray(offset)) for offset in NEIGHBORS
        } == NEIGHBORS:
            matrices.append(matrix)
    if len(matrices) != 12:
        raise ValueError("expected 12 hex-lattice symmetries")
    return matrices


def source_eyes(cache: Path) -> dict[str, dict]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing constructor for R class")
        raw = rdata.read_rda(cache / "20240701.RData")
        neighborhood = rdata.read_rda(cache / "20240701_nb.RData")
        published_pairs = rdata.read_rda(cache / "eyemap.RData")
    all_rays = np.asarray(raw["ucl_rot_sm"], dtype=np.float64)
    lens_order = np.argsort(np.asarray(raw["i_match"], dtype=np.int64))
    sorted_rays = all_rays[lens_order]
    left_mask = np.asarray(raw["ind_left_lens"], dtype=bool)
    fafb_pairs = np.asarray(published_pairs["eyemap"], dtype=np.int64)
    fafb_rays = np.asarray(published_pairs["ucl_rot_sm"], dtype=np.float64)
    if len(fafb_pairs) != len(fafb_rays) or not np.array_equal(
            sorted_rays[~left_mask][fafb_pairs[:, 1] - 1], fafb_rays):
        raise ValueError("source right-eye ray ordering disagrees with published FAFB pair data")
    result = {}
    for eye, mask, key in (("L", left_mask, "ind_xy_left"),
                           ("R", ~left_mask, "ind_xy_right")):
        indices = np.asarray(neighborhood[key], dtype=np.int64)
        rays = sorted_rays[mask][indices[:, 0] - 1]
        if len(np.unique(indices[:, 0])) != len(indices) or len(indices) != int(mask.sum()):
            raise ValueError(f"incomplete or duplicated source lens indices: {eye}")
        if not np.allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-6):
            raise ValueError(f"non-unit published rays: {eye}")
        if len({tuple(row) for row in indices[:, 1:]}) != len(indices):
            raise ValueError(f"duplicate source grid coordinates: {eye}")
        result[eye] = {"grid": indices[:, 1:], "lens_ids": indices[:, 0], "rays": rays}
    return result


def male_columns(workbook: Path) -> dict[str, dict]:
    result = {}
    for eye, sheet in (("L", "Left OL"), ("R", "Right OL")):
        table = pd.read_excel(workbook, sheet_name=sheet)
        grid = table["column"].str.extract(r"^ME_[LR]_col_(\d+)_(\d+)$")
        if grid.isna().any().any() or len(grid.drop_duplicates()) != len(grid):
            raise ValueError(f"invalid or duplicate MaleCNS column: {eye}")
        coords = grid.astype(np.int64).to_numpy()
        dra = coords[table["column_type"].eq("DRA").to_numpy()]
        result[eye] = {"grid": coords, "dra": dra}
    return result


def annotation_stats(data: Path, *, fetch: bool) -> dict:
    path = data / "raw" / Path(ANNOTATIONS_URL).name
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"{path}: rerun with --fetch")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(ANNOTATIONS_URL, timeout=90) as response:
            path.write_bytes(response.read())
    checked_hash(path, ANNOTATIONS_SHA256)
    table = feather.read_table(path, columns=["type", "assignedOlHex1", "assignedOlHex2"])
    frame = table.to_pandas()
    located = frame["assignedOlHex1"].notna() & frame["assignedOlHex2"].notna()
    counts = {}
    for cell_type in ("L1", "T4a", "T4b", "T5a", "T5b"):
        selected = frame["type"] == cell_type
        counts[cell_type] = {"total": int(selected.sum()),
                             "both_hex_coordinates": int((selected & located).sum())}
    return {"url": ANNOTATIONS_URL, "sha256": ANNOTATIONS_SHA256,
            "counts": counts}


def search_eye(source: dict, male: dict, eye: str) -> tuple[list[dict], list[dict]]:
    source_grid = source["grid"]
    male_grid = male["grid"]
    male_set = {tuple(row) for row in male_grid}
    source_rays = {tuple(row): source["rays"][i] for i, row in enumerate(source_grid)}
    all_rows = []
    accepted = []
    for symmetry_id, matrix in enumerate(lattice_symmetries()):
        transformed = source_grid @ matrix.T
        center = np.rint(male_grid.mean(axis=0) - transformed.mean(axis=0)).astype(np.int64)
        inverse = np.linalg.inv(matrix).astype(np.int64)
        for dq in range(-SHIFT_RADIUS, SHIFT_RADIUS + 1):
            for dp in range(-SHIFT_RADIUS, SHIFT_RADIUS + 1):
                shift = center + (dq, dp)
                mapped = {tuple(row + shift) for row in transformed}
                overlap = len(mapped & male_set)
                dorsal_grid = (male["dra"] - shift) @ inverse.T
                dorsal_rays = [source_rays.get(tuple(row)) for row in dorsal_grid]
                dorsal_present = [ray for ray in dorsal_rays if ray is not None]
                dorsal_positive = sum(float(ray[2]) > 0 for ray in dorsal_present)
                suitable = (overlap >= math.ceil(OVERLAP_FRACTION * len(source_grid))
                            and dorsal_positive == len(male["dra"]))
                record = {
                    "eye": eye, "symmetry_id": symmetry_id,
                    "a11": int(matrix[0, 0]), "a12": int(matrix[0, 1]),
                    "a21": int(matrix[1, 0]), "a22": int(matrix[1, 1]),
                    "shift_q": int(shift[0]), "shift_p": int(shift[1]),
                    "overlap_columns": overlap,
                    "source_facets": len(source_grid),
                    "male_columns": len(male_grid),
                    "dra_present": len(dorsal_present),
                    "dra_positive_z": dorsal_positive,
                    "dra_total": len(male["dra"]),
                    "accepted_exploratory": suitable,
                }
                all_rows.append(record)
                if suitable:
                    accepted.append(record)
    accepted.sort(key=lambda row: (-row["overlap_columns"], row["symmetry_id"],
                                   row["shift_q"], row["shift_p"]))
    if not accepted:
        raise ValueError(f"no exploratory grid registrations: {eye}")
    return all_rows, accepted


def assigned_rays(source: dict, candidate: dict) -> dict[tuple[int, int], tuple[int, np.ndarray]]:
    matrix = np.array([[candidate["a11"], candidate["a12"]],
                       [candidate["a21"], candidate["a22"]]], dtype=np.int64)
    shift = np.array([candidate["shift_q"], candidate["shift_p"]], dtype=np.int64)
    coords = source["grid"] @ matrix.T + shift
    return {tuple(row): (int(source["lens_ids"][i]), source["rays"][i])
            for i, row in enumerate(coords)}


def angular_span(vectors: list[np.ndarray]) -> float:
    angles = []
    for first, second in itertools.combinations(vectors, 2):
        angles.append(math.degrees(math.acos(float(np.clip(first @ second, -1, 1)))))
    return max(angles)


def quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    p50, p95 = np.quantile(values, [0.5, 0.95])
    return {"n_columns": len(values), "median_deg": round(float(p50), 3),
            "p95_deg": round(float(p95), 3), "max_deg": round(float(max(values)), 3)}


def build_outputs(output: Path, sources: dict, males: dict, selected: dict,
                  receptor_map: Path) -> dict:
    with np.load(receptor_map) as data:
        ids = data["ids"]
        sides = data["side"]
        hex_columns = data["hex_columns"]
        provenance = data["provenance"]
    n_candidates = min(len(selected[eye]) for eye in "LR")
    ray_arrays = np.full((n_candidates, len(ids), 3), np.nan, dtype=np.float32)
    column_rows = []
    spread_rows = []
    stats = {}
    for eye in "LR":
        src = sources[eye]
        male = males[eye]
        male_set = {tuple(row) for row in male["grid"]}
        boundary = {point for point in male_set if any(
            (point[0] + dq, point[1] + dp) not in male_set for dq, dp in NEIGHBORS)}
        grids = [assigned_rays(src, candidate) for candidate in selected[eye][:n_candidates]]
        per_rank = []
        for rank, grid in enumerate(grids, start=1):
            matched_columns = male_set & grid.keys()
            eligible = (sides == eye) & np.isfinite(hex_columns).all(axis=1)
            matched_receptors = 0
            for i in np.flatnonzero(eligible):
                pair = grid.get(tuple(hex_columns[i].astype(np.int64)))
                if pair is not None:
                    ray_arrays[rank - 1, i] = pair[1]
                    matched_receptors += 1
            for q, p in sorted(matched_columns):
                lens_id, ray = grid[(q, p)]
                column_rows.append((eye, rank, q, p, lens_id,
                                    *(f"{axis:.8f}" for axis in ray),
                                    "boundary" if (q, p) in boundary else "interior"))
            per_rank.append({"rank": rank, "matched_male_columns": len(matched_columns),
                             "matched_receptors": matched_receptors,
                             "matched_direct_receptors": int(np.sum(
                                 (sides == eye) & (provenance == "direct") &
                                 np.isfinite(ray_arrays[rank - 1]).all(axis=1))),
                             "matched_inferred_l1_receptors": int(np.sum(
                                 (sides == eye) & (provenance == "inferred_l1") &
                                 np.isfinite(ray_arrays[rank - 1]).all(axis=1)))})
        interior_spread = []
        boundary_spread = []
        for q, p in sorted(male_set):
            entries = [grid.get((q, p)) for grid in grids]
            if any(entry is None for entry in entries):
                span = None
            else:
                span = angular_span([entry[1] for entry in entries])
                (boundary_spread if (q, p) in boundary else interior_spread).append(span)
            spread_rows.append((eye, q, p, "boundary" if (q, p) in boundary else "interior",
                                "" if span is None else f"{span:.6f}"))
        stats[eye] = {"ranked_coverage": per_rank,
                      "candidate_angle_span_degrees": {
                          "definition": "maximum pairwise great-circle angle among accepted exploratory registrations, where every candidate maps a MaleCNS column; not error against a male measurement",
                          "interior": quantiles(interior_spread),
                          "boundary": quantiles(boundary_spread)},
                      "columns_missing_in_at_least_one_candidate":
                          len(male_set) - len(interior_spread) - len(boundary_spread)}
    with (output / "candidate_columns.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("eye", "candidate_rank", "male_hex1", "male_hex2", "female_lens_id",
                         "ray_x", "ray_y", "ray_z", "male_column_region"))
        writer.writerows(column_rows)
    with (output / "candidate_spread.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("eye", "male_hex1", "male_hex2", "male_column_region", "angular_span_deg"))
        writer.writerows(spread_rows)
    np.savez_compressed(output / "candidate_rays.npz", ids=ids, side=sides,
                        rays_by_rank=ray_arrays,
                        note="Provisional female facet rays indexed by independently ranked eye maps; no FPV head pose")
    return stats


def plot(result: dict, svg: Path, png: Path) -> None:
    left = result["candidates"]["L"][0]
    right = result["candidates"]["R"][0]
    n_candidates = result["candidate_count_by_eye"]
    fig = plt.figure(figsize=(11, 7), layout="constrained", facecolor="#fbfaf7")
    layout = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.2])
    header = fig.add_subplot(layout[0, :])
    header.axis("off")
    header.set_title("№13 · Кандидаты оптической карты", loc="left", fontsize=17, weight="bold")
    for x, label in [
        (0.01, f"µCT самки\nЛ {left['source_facets']} / П {right['source_facets']} фасетки\n"
               "измеренные лучи"),
        (0.34, "12 поворотов/зеркал × сдвиги\nконтроль: дорсальная область DRA\nпоиск без спайков"),
        (0.72, f"MaleCNS v1.0\nЛ {left['male_columns']} / П {right['male_columns']} колонки\n"
               f"{n_candidates['L']} / {n_candidates['R']} кандидата"),
    ]:
        header.text(x, 0.58, label, transform=header.transAxes, fontsize=10, va="center",
                    bbox={"boxstyle": "round,pad=0.6", "fc": "#e6eced", "ec": "#34434a"})
    header.text(0.295, 0.57, "→", transform=header.transAxes, fontsize=21, ha="center")
    header.text(0.665, 0.57, "→", transform=header.transAxes, fontsize=21, ha="center")
    header.text(0.5, 0.1, "Инверсия дорсальной стороны отвергается; угол глаза к FPV-камере ещё не проверен",
                transform=header.transAxes, ha="center", fontsize=10)
    bars = fig.add_subplot(layout[1, 0])
    positions = np.arange(3)
    for index, eye in enumerate("LR"):
        matches = [item["overlap_columns"] for item in result["candidates"][eye]]
        bars.bar(positions + (index - 0.5) * 0.32, matches, width=0.3,
                 color=("#528ca1", "#d8b15b")[index], hatch=("", "//")[index],
                 label=("Левый глаз", "Правый глаз")[index])
        for pos, count in zip(positions + (index - 0.5) * 0.32, matches):
            bars.text(pos, count + 6, str(count), ha="center", fontsize=9)
    bars.set_xticks(positions, ["1", "2", "3"])
    bars.set_ylim(0, 1000)
    bars.set_xlabel("Ранг кандидата")
    bars.set_ylabel("Совпавшие фасетки с колонками (n)")
    bars.set_title("Форма решётки: 3 варианта на глаз")
    bars.legend(frameon=False, fontsize=9)
    bars.spines[["top", "right"]].set_visible(False)
    notes = fig.add_subplot(layout[1, 1])
    notes.axis("off")
    notes.text(0, 0.96, "Разброс углов между вариантами", fontsize=13, weight="bold", va="top")
    for y, eye, label in ((0.8, "L", "Левый"), (0.56, "R", "Правый")):
        stats = result["map_stats"][eye]["candidate_angle_span_degrees"]
        inside, edge = stats["interior"], stats["boundary"]
        notes.text(0, y, f"{label}: внутри {inside['median_deg']:.2f}° "
                   f"(n={inside['n_columns']} колонок)\n"
                   f"           край {edge['median_deg']:.2f}° "
                   f"(n={edge['n_columns']} колонок)", fontsize=11, va="top")
    maximum = max(result["map_stats"][eye]["candidate_angle_span_degrees"][region]["max_deg"]
                  for eye in "LR" for region in ("interior", "boundary"))
    notes.text(0, 0.29, f"Максимум: {maximum:.2f}° · источник: сохранённые лучи\n"
               "0 независимо проверенных пар с MaleCNS\n0 отложенных нейронных проб", fontsize=11, va="top")
    notes.text(0, 0.07, "Разброс вариантов ≠ ошибка относительно мужского глаза.",
               fontsize=10, style="italic", va="top")
    fig.savefig(svg, bbox_inches="tight")
    canonicalize_svg(svg)
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/flybrain"))
    parser.add_argument("--map", type=Path, default=Path("runs/retina-2d/map.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/optical-map-13"))
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    cache = args.data / "raw" / "eyemap-t4"
    for name in SOURCE_FILES:
        checked_source(cache, name, fetch=args.fetch)
    for name, digest in MALE_FILES.items():
        checked_hash(args.data / name, digest)
    checked_hash(args.map, MAP_SHA256)
    sources = source_eyes(cache)
    males = male_columns(args.data / "raw/optic-columns.xlsx")
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    selected = {}
    inverted_controls = {}
    for eye in "LR":
        search_rows, selected[eye] = search_eye(sources[eye], males[eye], eye)
        all_rows.extend(search_rows)
        inverted = [row for row in search_rows if row["dra_present"] > 0 and
                    row["dra_positive_z"] == 0]
        inverted_controls[eye] = max(inverted, key=lambda row: row["overlap_columns"])
    with (args.output / "candidate_search.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    stats = build_outputs(args.output, sources, males, selected, args.map)
    result = {
        "status": "exploratory_grid_candidates_only; optical residual and camera pose unvalidated",
        "selection": "enumerate 12 hex-lattice symmetries and integer shifts within ±8 of the centroid difference; require at least 95% source facet overlap, all MaleCNS DRA columns matched to measured rays with positive dorsal z; rank by overlap only",
        "selection_uses_neural_responses": False,
        "source_files_sha256": SOURCE_FILES,
        "male_files_sha256": MALE_FILES,
        "male_annotation_audit": annotation_stats(args.data, fetch=args.fetch),
        "source_grid_to_male_grid": "[hex1,hex2]^T = A [female_q,female_p]^T + shift",
        "candidate_count_by_eye": {eye: len(selected[eye]) for eye in "LR"},
        "candidates": {eye: selected[eye] for eye in "LR"},
        "dorsal_inverted_shape_controls": inverted_controls,
        "map_stats": stats,
    }
    result_path = args.output / "candidates.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    plot(json.loads(result_path.read_text(encoding="utf-8")),
         args.output / "candidate_infographic.svg", args.output / "candidate_infographic.png")
    print(json.dumps({"counts": result["candidate_count_by_eye"], "stats": stats},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
