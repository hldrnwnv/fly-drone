"""Audit whether published female eye rays can be registered to MaleCNS v1.0.

FLY_DATA=$PWD/data/flybrain .venv/bin/python scripts/audit_optical_map.py --fetch
The source RData files are cached under data/flybrain/raw/eyemap-t4, not committed.
No source ray is assigned to a MaleCNS body ID without a validated registration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from math import atan, degrees, radians, tan
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdata

SOURCE_COMMIT = "99d2a43123db636cedb55af9ff31a59657e7d17e"
SOURCE_BASE = f"https://raw.githubusercontent.com/reiserlab/eyemap_T4/{SOURCE_COMMIT}/"
SOURCE_FILES = {
    "data/microCT/20240701.RData": "8408447f8c5798fbc6d751de79ef4983d08c342a500c9a55b2a3b4b38d92deec",
    "data/eyemap.RData": "c3f63c69afdc3f381cdafabb1d376b8e25ec5a71bce67550ae42fcdd79cf618e",
    "proc_eyemap.R": "35c0d8d48a67ea6bd11237464f01a6e87f13146e07baef7ad863ec1e8795df6d",
    "proc_uCT.R": "ba96990c55887a5e046323f5ea6e58e676f92de5909c0ef08ba10a4bc9acf65f",
}
MALE_FILES = {
    "raw/optic-columns.xlsx": "d4af1cacb751036f7e84bfecc9bec79ca010066ac066559c29b566003ec080d3",
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}
MAP_SHA256 = "a224fe3afb3fa99ee2f4080ad7bf909b3788e787ffc613ce8dea874a71b000f8"
NEIGHBORS = ((1, 0), (0, 1), (1, 1))  # each undirected hex-grid edge once


def checked_source(cache: Path, source_name: str, *, fetch: bool) -> Path:
    path = cache / Path(source_name).name
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"{path}: rerun with --fetch")
        cache.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_BASE + source_name, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_FILES[source_name]:
        raise ValueError(f"source checksum mismatch: {path}: {digest}")
    return path


def checked_hash(path: Path, expected: str) -> str:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch: {path}: {actual}")
    return actual


def published_eye_stats(cache: Path) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing constructor for R class")
        raw = rdata.read_rda(cache / "20240701.RData")
        matched = rdata.read_rda(cache / "eyemap.RData")
    left = np.asarray(raw["ind_left_lens"], dtype=bool)
    lens = np.asarray(matched["lens_ixy"], dtype=np.int64)
    pairs = np.asarray(matched["eyemap"], dtype=np.int64)
    rays = np.asarray(matched["ucl_rot_sm"], dtype=float)
    if len(pairs) != len(rays) or len(set(pairs[:, 1])) != len(pairs):
        raise ValueError("published one-to-one lens/ray map is inconsistent")
    if not np.allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-6):
        raise ValueError("published rays are not unit vectors")
    lens_grid = {int(row[0]): (int(row[1]), int(row[2])) for row in lens}
    grid_rays = {lens_grid[int(lens_id)]: rays[i] for i, lens_id in enumerate(pairs[:, 1])}
    if len(grid_rays) != len(pairs):
        raise ValueError("published grid has duplicate matched positions")
    adjacent_angles = []
    for (q, p), ray in grid_rays.items():
        for dq, dp in NEIGHBORS:
            other = grid_rays.get((q + dq, p + dp))
            if other is not None:
                adjacent_angles.append(degrees(float(np.arccos(np.clip(ray @ other, -1, 1)))))
    if not adjacent_angles:
        raise ValueError("no adjacent published rays")
    q05, median, q95 = np.quantile(adjacent_angles, [0.05, 0.5, 0.95])
    return {
        "specimen": "female whole-head microCT; published eye-to-FAFB match is right eye",
        "facets_by_eye": {"L": int(left.sum()), "R": int((~left).sum())},
        "published_right_eye_grid": int(len(lens)),
        "published_right_eye_to_fafb_pairs": int(len(pairs)),
        "neighbor_sensitivity_degrees": {
            "definition": "great-circle separation of adjacent matched right-eye rays; one illustrative grid-step misregistration, not a measured MaleCNS error",
            "n_edges": len(adjacent_angles),
            "median": round(float(median), 3),
            "p05": round(float(q05), 3),
            "p95": round(float(q95), 3),
        },
    }


def male_map_stats(map_path: Path, csv_path: Path) -> dict:
    with np.load(map_path) as data:
        rows = {name: data[name] for name in data.files}
    length = len(rows["ids"])
    if any(len(value) != length for value in rows.values()):
        raise ValueError("receptor map arrays have different lengths")
    if len(np.unique(rows["ids"])) != length:
        raise ValueError("duplicate receptor body ID")
    valid = np.isfinite(rows["hex_columns"]).all(axis=1)
    if not np.array_equal(valid, np.isfinite(rows["xy"]).all(axis=1)):
        raise ValueError("source column and projected-pixel coverage disagree")
    if np.any((rows["provenance"] == "unmapped") != ~valid):
        raise ValueError("unmapped provenance differs from column coverage")
    by_eye = {}
    for eye in "LR":
        selected = rows["side"] == eye
        columns = np.unique(rows["hex_columns"][selected & valid], axis=0)
        by_eye[eye] = {
            "receptors_total": int(selected.sum()),
            "direct_column_receptors": int(np.sum(selected & (rows["provenance"] == "direct"))),
            "inferred_l1_receptors": int(np.sum(selected & (rows["provenance"] == "inferred_l1"))),
            "unmapped_receptors": int(np.sum(selected & ~valid)),
            "distinct_located_columns": int(len(columns)),
            "validated_optical_rays": 0,
        }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("body_id", "eye", "receptor_type", "hex1", "hex2", "column_provenance",
                         "l1_partner_id", "optical_ray_status", "female_lens_id", "angular_residual_deg"))
        for i in range(length):
            q, p = rows["hex_columns"][i]
            writer.writerow((int(rows["ids"][i]), str(rows["side"][i]),
                             str(rows["receptor_type"][i]),
                             "" if not valid[i] else int(q), "" if not valid[i] else int(p),
                             str(rows["provenance"][i]),
                             "" if rows["l1_partner_id"][i] < 0 else int(rows["l1_partner_id"][i]),
                             "unregistered", "", ""))
    return {"receptors_total": length, "by_eye": by_eye,
            "optical_registration": "unavailable: no validated MaleCNS-to-facet landmarks or body-ID correspondence",
            "angular_residual_degrees": None, "peripheral_uncertainty_degrees": None}


def workbook_stats(path: Path) -> dict:
    sheets = {}
    for eye, sheet in (("R", "Right OL"), ("L", "Left OL")):
        table = pd.read_excel(path, sheet_name=sheet)
        expected = {"column", "L1", "R7", "R8", "column_type"}
        if not expected.issubset(table.columns):
            raise ValueError(f"missing fields in {sheet}")
        names = table["column"].astype(str)
        if not names.str.match(f"ME_{eye}_col_[0-9]+_[0-9]+$").all():
            raise ValueError(f"invalid column labels in {sheet}")
        sheets[eye] = {
            "columns": int(len(table)),
            "dorsal_rim_columns": int((table["column_type"] == "DRA").sum()),
            "fields": list(table.columns),
            "measured_facet_ray_field": False,
            "equator_or_meridian_correspondence_field": False,
        }
    return sheets


def camera_stats(xml_path: Path, width: int = 320, height: int = 180) -> dict:
    camera = ElementTree.parse(xml_path).find(".//camera[@name='fpv']")
    if camera is None:
        raise ValueError("FPV camera missing")
    fovy = float(camera.attrib["fovy"])
    focal_pixels = height / (2 * tan(radians(fovy) / 2))
    fovx = 2 * degrees(atan(width / height * tan(radians(fovy) / 2)))
    axes = [float(value) for value in camera.attrib["xyaxes"].split()]
    right, up = np.asarray(axes[:3]), np.asarray(axes[3:])
    forward = -np.cross(right, up)
    return {"resolution_pixels": [width, height], "position_m_in_quad_frame":
            [float(value) for value in camera.attrib["pos"].split()],
            "vertical_fov_degrees": fovy, "horizontal_fov_degrees": round(fovx, 3),
            "focal_length_pixels": round(focal_pixels, 3),
            "right_in_quad_frame": right.tolist(), "up_in_quad_frame": up.tolist(),
            "forward_in_quad_frame": forward.tolist(),
            "pixel_ray_formula": "normalize([1, -(u-W/2)/f, (H/2-v)/f]) in quad frame; pixel centers u,v; f=H/(2*tan(fovy/2))",
            "unknown_transform": "quad frame to male head/eye coordinates"}


def plot(result: dict, svg: Path, png: Path) -> None:
    male = result["male_map"]["by_eye"]
    source = result["published_eye"]
    facets = source["facets_by_eye"]
    validated_rays = sum(male[eye]["validated_optical_rays"] for eye in "LR")
    fig = plt.figure(figsize=(11, 7.2), layout="constrained", facecolor="#fbfaf7")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    ax.set_title("№13 · Проверка оптической регистрации", fontsize=17, loc="left", weight="bold")
    boxes = [
        (0.01, f"Оптика самки, µCT\nЛ {facets['L']} / П {facets['R']} фасеток\n"
               f"П → FAFB: {source['published_right_eye_to_fafb_pairs']} пар"),
        (0.35, f"MaleCNS v1.0\nЛ {male['L']['distinct_located_columns']} / "
               f"П {male['R']['distinct_located_columns']} колонок\nНет парных ориентиров"),
        (0.69, f"Камера FPV\n{' × '.join(map(str, result['camera']['resolution_pixels']))} пикс. · "
               f"{result['camera']['vertical_fov_degrees']:g}° вертикально\nПоза головы/глаз неизвестна"),
    ]
    for x, label in boxes:
        ax.text(x, 0.56, label, transform=ax.transAxes, va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.7", "fc": "#e6eced", "ec": "#34434a"})
    ax.text(0.286, 0.55, "→", transform=ax.transAxes, fontsize=22, ha="center")
    ax.text(0.626, 0.55, "→", transform=ax.transAxes, fontsize=22, ha="center")
    ax.text(0.5, 0.12, "Запланированы после регистрации: зеркало · поворот · сдвиг · старая карта · порядок кадров",
            transform=ax.transAxes, ha="center", fontsize=10)
    bars = fig.add_subplot(grid[1, 0])
    names = ["Левый", "Правый"]
    direct = [male[e]["direct_column_receptors"] for e in "LR"]
    inferred = [male[e]["inferred_l1_receptors"] for e in "LR"]
    missing = [male[e]["unmapped_receptors"] for e in "LR"]
    positions = np.arange(2)
    bars.bar(positions, direct, color="#528ca1", label="Прямая колонка R7/R8")
    bars.bar(positions, inferred, bottom=direct, color="#d8b15b", hatch="//", label="Вывод через L1")
    bars.bar(positions, missing, bottom=np.array(direct) + inferred,
             color="#dadada", hatch="xx", label="Колонка не найдена")
    bars.set_xticks(positions, names)
    bars.set_ylabel("Рецепторы MaleCNS (n)")
    bars.set_title(f"Покрытие колонками; лучи: {validated_rays} / {result['male_map']['receptors_total']}")
    bars.legend(fontsize=8, frameon=False, loc="upper left")
    bars.spines[["top", "right"]].set_visible(False)
    notes = fig.add_subplot(grid[1, 1])
    notes.axis("off")
    angle = source["neighbor_sensitivity_degrees"]
    notes.text(0, 0.95, "Итог регистрации", fontsize=14, weight="bold", va="top")
    notes.text(0, 0.79, f"{validated_rays} проверенных пар фасетка → MaleCNS\n"
               f"Угловая ошибка не определена\n{result['neural_held_out_tests']} отложенных нейронных проб",
               fontsize=12, va="top")
    notes.text(0, 0.49,
               f"Сдвиг на одну колонку у самки:\nмедиана {angle['median']:.2f}° "
               f"(5–95%: {angle['p05']:.2f}–{angle['p95']:.2f}°; n={angle['n_edges']} рёбер)",
               fontsize=11, va="top")
    notes.text(0, 0.2, "Это чувствительность к сдвигу,\nне измеренная ошибка MaleCNS.",
               fontsize=10, va="top", style="italic")
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
                   encoding="utf-8")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/flybrain"))
    parser.add_argument("--map", type=Path, default=Path("runs/retina-2d/map.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/optical-map-13"))
    parser.add_argument("--fetch", action="store_true", help="download pinned primary files if missing")
    args = parser.parse_args()
    cache = args.data / "raw" / "eyemap-t4"
    for name in SOURCE_FILES:
        checked_source(cache, name, fetch=args.fetch)
    for name, digest in MALE_FILES.items():
        checked_hash(args.data / name, digest)
    checked_hash(args.map, MAP_SHA256)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "question": "Can female facet rays be assigned to MaleCNS v1.0 receptor body IDs?",
        "status": "registration_not_validated; neural test gated",
        "source_commit": SOURCE_COMMIT,
        "source_files_sha256": SOURCE_FILES,
        "malecns_files_sha256": MALE_FILES,
        "existing_map_sha256": MAP_SHA256,
        "published_eye": published_eye_stats(cache),
        "male_workbook": workbook_stats(args.data / "raw/optic-columns.xlsx"),
        "male_map": male_map_stats(args.map, args.output / "receptor_audit.csv"),
        "camera": camera_stats(Path("assets/fpv_quad.xml")),
        "neural_held_out_tests": 0,
        "controls_and_held_out": "not run because no admissible source-to-MaleCNS registration",
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(json.loads(result_path.read_text(encoding="utf-8")),
         args.output / "infographic.svg", args.output / "infographic.png")
    print(json.dumps({"status": result["status"], "male_map": result["male_map"],
                      "published_eye": result["published_eye"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
