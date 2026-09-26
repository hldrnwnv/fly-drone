# FPV camera benchmark — 2026-09-25

## Protocol

MuJoCo 3.14.0, one 320×180 FPV camera and one four-motor quad model. A fixed colour detector converts each frame to a target-bearing angle. All controllers use the same camera, 0.4 s decision interval, altitude/speed stabilizer, two gate layouts, and matched start pose and dropout seed within each trial. Five start poses per layout are evaluated both normally and with 30% randomly dropped frames. The real MaleCNS graph and three independently degree-shuffled graphs each receive 96 identical synthetic bearing samples for readout training, 32 held-out samples, and 20 neural ticks per decision. The shuffled graphs preserve each neuron's connection counts, but not cell-type-specific or spatial wiring. The geometric controller has the same 400 ms deadline but uses fewer operations.

## Results

| Controller | Gate passes | Normal | 30% dropout | Same-frame direction | Mean decision time |
|---|---:|---:|---:|---:|---:|
| Geometry | 20/20 | 10/10 | 10/10 | 155/164 | 6.9 ms |
| MaleCNS connectome | 20/20 | 10/10 | 10/10 | 150/164 | 46.9 ms |
| Shuffled graph 0 | 1/20 | 1/10 | 0/10 | 80/164 | 36.2 ms |
| Shuffled graph 1 | 0/20 | 0/10 | 0/10 | 84/164 | 39.4 ms |
| Shuffled graph 2 | 6/20 | 4/10 | 2/10 | 90/164 | 39.7 ms |

The same-frame test replays the exact geometric-flight bearing sequence to every controller; frame hashes are saved for each observation. All 20 first-frame hashes and start poses match between geometry, MaleCNS and each shuffled network. There were no recorded 400 ms deadline misses. The real graph's held-out direction readout was correct on 22/25 nonneutral samples. Raw trajectories, observations, frame hashes, model fits and individual run results are in `results.json`; readout weights are saved as `*_readout.npz`.

## Interpretation

The original graph works better than these three globally shuffled controls for this specific LC10a-to-DNa02 steering adapter. It shows **no flight advantage over the geometric controller** here; both pass every tested gate, and geometry is faster and slightly more accurate on the replayed observations. The dropout test cannot establish superior robustness because both pass all ten dropout flights. The colour detector supplies a precomputed angle, so this experiment **does not test whether the fly connectome handles complex raw video better**. It also does not show that the full graph or the quad model is biologically or aerodynamically faithful.

## Reproduction

From the repository root, run `FLY_DATA="$PWD/data/flybrain" .venv/bin/fly-fpv benchmark --output runs/vision-benchmark`. The seed is 7; wiring seeds are 2026–2028. Run `.venv/bin/python -m unittest discover -s tests -v` for the small integration tests.
