"""Build an auditable, partial 2-D MaleCNS v1.0 receptor-to-column map.

Run with FLY_DATA=$PWD/data/flybrain .venv/bin/python scripts/build_retina_map.py
The optic-column workbook comes from the pinned MaleCNS supplement; flybrain's
normalized graph is used only to infer R1-6 -> annotated L1 partners.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy import sparse

from fly_drone.retina_2d import normalized_eye_coordinates

SOURCE_URL = ("https://raw.githubusercontent.com/flyconnectome/2025malecns/"
              "67767d2233657983993ff6c2be48e836a935863c/"
              "supplemental_data/optic-column-type-assignments-v1.0.xlsx")
SOURCE_SHA256 = "d4af1cacb751036f7e84bfecc9bec79ca010066ac066559c29b566003ec080d3"


def build(data: Path, output: Path) -> dict:
    workbook = data / "raw" / "optic-columns.xlsx"
    if not workbook.exists():
        workbook.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_URL, timeout=30) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError("downloaded optic-column workbook SHA-256 differs from pinned source")
        workbook.write_bytes(content)
    if hashlib.sha256(workbook.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("optic-column workbook SHA-256 differs from pinned source")
    # Use the parser shipped with the pinned flybrain version, which reads both eyes.
    columns_by_body = importlib.import_module("flybrain.build").optic_columns(workbook)
    with np.load(data / "brain.npz") as meta:
        ids = meta["ids"]
        visual = meta["visual"]
        side = meta["side"]
        kind = meta["cell_type"]
        azimuth = meta["azimuth"]
    receptor_ids = ids[visual]
    receptor_side = side[visual]
    receptor_type = kind[visual]
    columns = np.full((len(visual), 2), np.nan, dtype=np.float32)
    provenance = np.full(len(visual), "unmapped", dtype="U12")
    partner_id = np.full(len(visual), -1, dtype=np.int64)

    for i, body in enumerate(receptor_ids):
        entry = columns_by_body.get(int(body))
        if entry is not None and entry[0] == receptor_side[i] and receptor_type[i] in ("R7", "R8"):
            columns[i] = entry[1:]
            provenance[i] = "direct"

    l1 = np.array([i for i, body in enumerate(ids)
                   if kind[i] == "L1" and int(body) in columns_by_body], dtype=np.int32)
    r16 = np.flatnonzero(receptor_type == "R1-6")
    graph = sparse.load_npz(data / "weights.npz")
    partners = abs(graph[l1][:, visual[r16]]).tocsc()
    for j, receptor in enumerate(r16):
        start, stop = partners.indptr[j:j + 2]
        candidates = partners.indices[start:stop]
        strengths = partners.data[start:stop]
        same_side = [k for k, pos in enumerate(candidates)
                     if columns_by_body[int(ids[l1[pos]])][0] == receptor_side[receptor]]
        if not same_side:
            continue
        ranked = sorted(same_side, key=lambda k: (-strengths[k], int(ids[l1[candidates[k]]])))
        if len(ranked) > 1 and strengths[ranked[0]] <= 1.05 * strengths[ranked[1]]:
            continue  # no guessed tie-break coordinates
        target = int(ids[l1[candidates[ranked[0]]]])
        columns[receptor] = columns_by_body[target][1:]
        provenance[receptor] = "inferred_l1"
        partner_id[receptor] = target

    xy = normalized_eye_coordinates(columns, receptor_side)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=receptor_ids, visual=visual, side=receptor_side,
                        receptor_type=receptor_type, hex_columns=columns, xy=xy,
                        provenance=provenance, l1_partner_id=partner_id,
                        legacy_azimuth=azimuth)
    counts = {s: {p: int(np.sum((receptor_side == s) & (provenance == p)))
                  for p in ("direct", "inferred_l1", "unmapped")} for s in "LR"}
    receipt = {"dataset": "MaleCNS v1.0 via flybrain 0.1.0",
               "workbook_url": SOURCE_URL, "workbook_sha256": SOURCE_SHA256,
               "coordinates": "source hex1/hex2 columns; x=hex1-hex2, y=-(hex1+hex2), normalized within eye",
               "inference": "R1-6: strongest same-side annotated L1 partner in flybrain normalized graph; withhold within 5 percent top tie",
               "limits": "L1 column is an inferred sampling locus, not the R1-6 ommatidium or an angular optical axis; physical camera orientation and degrees per pixel uncalibrated",
               "unmapped_drive": "constant background luminance", "counts": counts}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/flybrain"))
    parser.add_argument("--output", type=Path, default=Path("runs/retina-2d/map.npz"))
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.output), indent=2))
