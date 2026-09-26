"""Run a short unmodified Shiu FlyWire model trial from a local checkout.

This checks the published model separately. FlyWire lacks the wing motor cells
used by wing-direct, so this script does not control the quad.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import brian2 as b2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="checkout of philshiu/Drosophila_brain_model")
    parser.add_argument("--output", type=Path, default=Path("runs/shiu-smoke/results.json"))
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--stimulus-index", type=int, default=0)
    args = parser.parse_args()
    if args.duration_ms <= 0:
        parser.error("duration-ms must be positive")
    source = args.source.resolve()
    spec = importlib.util.spec_from_file_location("shiu_original_model", source / "model.py")
    if spec is None or spec.loader is None:
        parser.error(f"Shiu model.py missing from {source}")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    b2.prefs.codegen.target = "numpy"
    b2.seed(7)
    start = time.perf_counter()
    neurons, synapses, spikes = model.create_model(
        source / "2023_03_23_completeness_630_final.csv",
        source / "2023_03_23_connectivity_630_final.parquet",
        model.default_params,
    )
    setup_s = time.perf_counter() - start
    if not 0 <= args.stimulus_index < len(neurons):
        parser.error("stimulus-index outside neuron population")
    inputs, neurons = model.poi(neurons, [args.stimulus_index], [], model.default_params)
    network = b2.Network(neurons, synapses, spikes, *inputs)
    network.run(args.duration_ms * b2.ms)
    result = {
        "source": "philshiu/Drosophila_brain_model original model.py and v630 FlyWire data",
        "commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                          text=True).strip(),
        "neurons": len(neurons), "synapses": len(synapses),
        "stimulus_index": args.stimulus_index,
        "stimulus": "original PoissonInput, 150 Hz, 250 x 0.275 mV",
        "duration_ms": args.duration_ms,
        "total_spikes": int(spikes.num_spikes),
        "stimulated_neuron_spikes": int(np.count_nonzero(
            np.asarray(spikes.i) == args.stimulus_index)),
        "setup_s": setup_s, "total_wall_s": time.perf_counter() - start,
        "drone_connected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
