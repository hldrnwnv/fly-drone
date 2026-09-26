# Stimulation search report

- Mixed differential evolution: 264 candidate evaluations; 2112 stochastic neural runs for training.
- Same frozen MaleCNS connectome in every run; seeds change model noise, not anatomical wiring.
- Neural selection used motor-neuron spike asymmetry. Physics selection used separate starts with the same external altitude correction.
- The final physics starts and moving-target chase were held out from evolution.
- Polarity +1 excites; -1 suppresses. Side rule +1 stimulates right cells for negative roll rate and left cells for positive roll rate.

## Best training variants by cell group

| Group | Polarity | Side rule | Gain | Duty | Neural score | Pairs |
|---|---:|---:|---:|---:|---:|---:|
| `SNpp37_38_06` | +1 | +1 | 1.00 | 1.00 | 2.30 | 4/4 |
| `SNpp_bundle` | +1 | +1 | 1.30 | 0.98 | 1.92 | 3/4 |
| `IN08B051_d` | +1 | +1 | 0.67 | 1.00 | 1.51 | 3/4 |
| `SNpp37_38` | +1 | +1 | 1.12 | 1.00 | 1.27 | 3/4 |
| `SNpp38` | +1 | +1 | 0.20 | 1.00 | 0.22 | 1/4 |
| `SNpp37` | +1 | +1 | 0.77 | 1.00 | 0.22 | 1/4 |
| `SApp08` | -1 | +1 | 0.44 | 0.65 | 0.00 | 0/4 |
| `DNa04` | -1 | +1 | 0.77 | 0.57 | 0.00 | 0/4 |
| `DNg24` | -1 | +1 | 0.21 | 0.47 | 0.00 | 0/4 |
| `DNa02` | -1 | -1 | 0.97 | 0.38 | 0.00 | 0/4 |

## Held-out neural response

| Group | Polarity | Side rule | Score | Opposite-sign pairs |
|---|---:|---:|---:|---:|
| `SNpp37_38_06` | +1 | +1 | 1.81 | 12/12 |
| `SNpp_bundle` | +1 | +1 | 1.53 | 10/12 |
| `IN08B051_d` | +1 | +1 | 0.93 | 9/12 |
| `SNpp37_38` | +1 | +1 | 0.69 | 7/12 |
| `SNpp38` | +1 | +1 | 0.00 | 0/12 |
| `DNa02` | -1 | -1 | 0.00 | 0/12 |

## MuJoCo roll perturbations

The four selection starts chose the final configuration. The eight test starts below were not used for evolution or selection.

| Condition | Survived / 8 | Mean duration (s) | Mean absolute roll (rad) |
|---|---:|---:|---:|
| winner | 8/8 | 4.00 | 0.249 |
| no stimulation | 1/8 | 2.29 | 0.489 |
| reference bundle | 8/8 | 4.00 | 0.303 |
| reversed winner | 0/8 | 1.21 | 0.485 |

The evolved bundle had lower roll error than the fixed bundle on 6/8 starts. Its mean paired difference was -0.055 rad; the exploratory exact sign-flip one-sided p-value is 0.14. This small test does not establish a reliable improvement over the fixed bundle.

## Selected interface and full-flight transfer

- Cell group: `SNpp_bundle`; polarity +1; side rule +1; gain 1.300; pulse duty 0.980.
- Moving-target scene without altitude assist: **crashed** after 3.06 s; tracked=False.
- The gyro-to-cell rule and motor-to-rotor map are engineered. Survival in the assisted roll task does not establish autonomous flight or biological fidelity.
- `results.json` contains every neural trial, physics decision, and chase trace; `checkpoint.json` contains the complete search history and resumable population.
