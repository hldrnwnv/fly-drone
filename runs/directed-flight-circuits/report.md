# Literature-directed flight circuit probe

The [machine-readable receipt](results.json) records 126 configurations and 1,524 neural runs on one **frozen** MaleCNS/flybrain graph. Each configuration uses 12 paired model-noise seeds: six to select dose, side, and pulse duration, then six held out. Following 20 warm-up ticks, stimulation and motor spikes are measured for 30 ticks (0.6 s). The stimulation is a constant artificial voltage, not natural sensory spike timing.

Run again with:

```bash
FLY_DATA="$PWD/data/flybrain" .venv/bin/python scripts/probe_known_flight_circuits.py
```

| Input group | Anatomical / experimental reason | Selected held-out response vs matched rest, per 0.6 s |
|---|---|---:|
| Bilateral `DNg02` | Published flight-power descending population | DLM/DVM power MNs **+90.7 spikes** (six paired effects: +81 to +99); rest 14.7, stimulated 105.3 |
| Bilateral `SApp08` | A MaleCNS cell is annotated as a haltere sensory neuron | `w-cHIN` relay **+52.8 spikes**; b1/b3 left-right difference **0.0** |
| Left `DNa04` | Candidate descending wing-steering route | `w-cHIN` **+8.8 spikes**; b1/b3 left-right difference **−0.3** |
| Right `DNa05` | Same proposed route | b1/b3 left-right difference **−0.5** |
| Left `SNpp37/38` | Tegula wing campaniform receptors | i1/i2 right-left difference **−3.8 spikes** (all six paired effects negative) |
| Left `SNpp06/26` | Distal wing campaniform receptors | i1/i2 right-left difference **−0.8 spikes** |
| Left `DNa02` | Walking-steering comparison | i1/i2 right-left difference **+0.2 spikes** |

The previous differential-evolution objective measured only i1/i2 asymmetry, so it could not score the large DNg02 **power** effect. The earlier `SNpp` gyro input actually stimulates **wing** sensory neurons; it is not a reconstructed haltere pathway. `SApp08` reaches `w-cHIN` in this model, but this probe finds no corresponding b1/b3 output. `DNa04` also drives the relay without a stable b1/b3 response. The [MANC circuit analysis](https://elifesciences.org/articles/96084/figures) proposes electrical connections in this route that a chemical-synapse model may omit.

The type queries are broad: bilateral `DNg02_a`–`g` selected **29 cells** and bilateral `SApp08` selected **47 cells** in this local model. These interventions cannot identify which individual cell generated the response. The `SApp08` result especially calls for cell-level subdivision before treating it as a haltere encoding result.

Cell labels and candidate roles come from the [DNg02 flight study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9206711/), [wing sensory annotation](https://elifesciences.org/articles/107867/figures), [VFB SApp08 annotation](https://www.virtualflybrain.org/term/sapp08_r-malecns54772-vfb_jrmc1b82/), and [DNa02 walking study](https://elifesciences.org/articles/102230). The VFB entry identifies one cell in an older MaleCNS release; it does not validate every `SApp08` cell selected here. The effects are model spike counts at selected doses, not biological reflex gains or quad flight performance. Different readouts and timing remain possible; no drone flight was attempted with these interventions.
