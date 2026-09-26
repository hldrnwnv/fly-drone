# Stage 1: Camera-guided steering

The current simple flight mode is `fly-fpv chase` in MuJoCo. The virtual FPV camera detects a colored tag on a moving target. Its measured bearing goes to a synthetic `LC10a` input of the frozen MaleCNS/flybrain model. A fitted readout of `DNa02` activity chooses the **turn direction**; the observed bearing also scales the turn command. A separate image-size rule sets forward speed, and a conventional stabilizer sets the four rotor thrusts. Target coordinates are used only to animate the scene and score the flight.

```bash
FLY_DATA="$PWD/data/flybrain" .venv/bin/fly-fpv chase --output runs/chase
```

The saved [results](results.json) use 96 training and 32 validation observations. Direction was correct on **22/25** nonneutral validation observations. In three 18-second neural flights, the target was visible at all 45 decisions per flight; **2/3** met the pursuit criterion. Their fractions of time in the follow band after 4 s were **0.646, 0.809, 0.789**; the threshold is 0.70. The geometric steering comparison met the criterion in its one run (fraction **0.814**). All four flights completed without crashing. Mean neural decision times were 42–43 ms, with no 400 ms deadline misses in these runs.

Watch [the neural flight](follow_demo.mp4) alongside [the geometric comparison](geometric.mp4). A [four-panel version with depth and optical flow](../chase-depth-flow/report.md) shows the same moving-target setup and records the exact camera decision frames used for its maps. The JSON includes camera frame hashes, detections, neural commands, speed commands, motor thrusts, and trajectories for all runs.

This is a demonstration of a **camera measurement → connectome steering readout → conventional quad stabilizer** interface. The color tag is segmented by ordinary code, and a supervised readout plus a geometric gain are fitted around the frozen graph. These few runs show no advantage over geometric steering. Direct wing-muscle-to-rotor control remains a separate research experiment.
