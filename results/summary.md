# KrakenScope automated experiment summary

Base seed: `20261201` · repetitions per scenario: `10`

| Scenario | Samples | Tracked RMSE | Steady RMSE | Raw RMSE | Improvement | P95 | Mode accuracy | Start delay | Stop delay | False switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_good | 4510 | 8.5 m | 6.0 m | 1278.2 m | +99.3% | 12.4 m | 97.0% | — | — | 3 |
| static_hard_noise | 6010 | 8.4 m | 4.9 m | 1486.2 m | +99.4% | 9.4 m | 98.3% | — | — | 2 |
| moving_straight | 4509 | 43.3 m | 27.0 m | 1874.6 m | +97.7% | 67.2 m | 97.3% | 12.7 s | — | 2 |
| moving_hard_noise | 4504 | 66.7 m | 65.4 m | 2079.1 m | +96.8% | 148.8 m | 98.0% | 10.9 s | — | 3 |
| moving_curve | 6009 | 45.1 m | 34.8 m | 2058.3 m | +97.8% | 71.6 m | 92.8% | 19.6 s | — | 2 |
| stop_move_stop | 5009 | 22.1 m | 19.9 m | 952.2 m | +97.7% | 42.6 m | 56.3% | 11.2 s | 19.8 s | 1 |
| bad_geometry | 5991 | 1201.9 m | 888.8 m | 8896.7 m | +86.5% | 1377.6 m | 84.8% | — | — | 13 |

`Tracked` is the KrakenScope output. `Raw` is the instantaneous two-bearing intersection.
Ground truth is used only after localization to calculate these metrics.
