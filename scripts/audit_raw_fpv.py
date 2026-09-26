"""Check saved issue #2 frame receipts and paired-controller invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(run_dir: Path) -> dict:
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    names = ("visual", "connectome", "degree_shuffled_0", "degree_shuffled_1",
             "degree_shuffled_2")
    with np.load(run_dir / "corpus_frames.npz") as corpus:
        train, validation = corpus["train"], corpus["validation"]
        for split, frames in (("train", train), ("validation", validation)):
            records = result["corpus"][split]
            assert len(records) == len(frames)
            for frame, record in zip(frames, records, strict=True):
                assert hashlib.sha256(frame.tobytes()).hexdigest() == record["frame_sha256"]
        train_hashes = {record["frame_sha256"] for record in result["corpus"]["train"]}
        validation_hashes = {record["frame_sha256"] for record in result["corpus"]["validation"]}
        assert not train_hashes & validation_hashes
    replayed = 0
    dropouts = 0
    with np.load(run_dir / "open_loop_frames.npz") as archive:
        for episode, by_controller in result["open_loop"].items():
            key = episode.replace("/", "_")
            frames, valid = archive[key], archive[key + "_valid"]
            reference = result["runs"][f"visual/{episode}"]
            observations = reference["control_observations"]
            assert len(frames) == len(valid) == len(observations)
            first_hash = observations[0]["frame_sha256"]
            for i, (frame, available, observation) in enumerate(
                    zip(frames, valid, observations, strict=True)):
                assert observation["frame_sha256"] not in train_hashes | validation_hashes
                assert bool(available) == (observation["reason"] != "dropout")
                if available:
                    assert hashlib.sha256(frame.tobytes()).hexdigest() == observation["frame_sha256"]
                    replayed += 1
                else:
                    dropouts += 1
                for name in names:
                    decision = by_controller[name]["decisions"][i]
                    assert decision["frame_sha256"] == observation["frame_sha256"]
                    assert decision["reason"] == observation["reason"]
            for name in names:
                run = result["runs"][f"{name}/{episode}"]
                assert run["initial_xy"] == reference["initial_xy"]
                assert run["initial_yaw"] == reference["initial_yaw"]
                assert run["control_observations"][0]["frame_sha256"] == first_hash
                assert len(by_controller[name]["decisions"]) == len(observations)
    adjacency = [result["fits"][name]["adjacency_sha256"] for name in names[1:]]
    assert len(set(adjacency)) == len(adjacency)
    for name in names[1:]:
        with np.load(run_dir / f"{name}_readout.npz") as readout:
            assert len(readout["group_names"]) == 18
            assert len(readout["weights"]) == 55
    report = {"episodes": len(result["open_loop"]), "controllers": len(names),
              "training_frames_verified": len(train), "validation_frames_verified": len(validation),
              "open_loop_frames_verified": replayed, "dropout_markers": dropouts,
              "same_first_frame_and_pose": True, "distinct_graph_hashes": True}
    (run_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/raw-fpv"))
    print(json.dumps(audit(parser.parse_args().run_dir), indent=2))
