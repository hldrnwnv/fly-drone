# Moving-target FPV flight with depth and optical flow

Run with `FLY_DATA="$PWD/data/flybrain" .venv/bin/fly-fpv chase --depth-flow-video --output runs/chase-depth-flow` after installing `.[fpv,perception]`.

The cow moves forward along an S-shaped lateral path: its recorded position changes from `(2.2, 0.0)` to about `(13.3, 0.43)` m during the 18-second run, with lateral position spanning about −1 to +1 m. The original external camera was mounted on the following quad, making the cow look stationary. The [corrected four-panel video](follow_depth_flow.mp4) uses a **fixed world camera** and shows changing cow/drone coordinates. Compare [0 s](frame_0s.png), [6 s](frame_6s.png), and [17 s](frame_17s.png). The cow is a rigid visual proxy whose legs do not animate.

The four panels show FPV, the fixed world view, Depth Anything V2 Small relative inverse depth, and RAFT Small optical flow. The displayed depth is **relative**, not meters. Flow describes image displacement between successive camera decisions, not a physical wind or velocity sensor.

The locally generated `sensory_maps.npz` contains the exact 45 FPV RGB frames used at steering decisions, their depth outputs, and 44 valid optical-flow estimates. This large array is excluded from Git and can be regenerated with the command above. Each saved RGB frame's SHA-256 matches the corresponding camera observation in [results.json](results.json). The first frame has no preceding image, so its flow panel is blank. Maps update every 0.4 s and remain visible between decisions. A [six-second still](frame_6s.png) shows all panels at once.

Steering is unchanged: a color-tag detector yields bearing, a frozen MaleCNS readout chooses turn direction, an image-size rule sets forward speed, and a conventional stabilizer controls four rotors. Depth and flow are computed **after the flight** for inspection; they are not brain inputs in this run. The mean offline inference time was about 138 ms per decision frame. Two of three neural flights met the pursuit criterion, matching the steering-only run; the geometric comparison met it in its one trial. These runs do not show a benefit from the depth or flow networks.
