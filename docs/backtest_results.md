# Historical Week 1 backtest

Run on 2026-09-01 with public nflverse weekly player box scores from 2021–25, PPR scoring, RB/WR/TE, and walk-forward holds outs for Weeks 1 of 2022–25. The first eligible holdout (2022) has no preceding labeled Week-1 panel, so learned baselines and conformal calibration begin in 2023. The component simulator used 3,000 draws/player for this fast reproducibility run.

| model | N | Median AE | 50% coverage | 50% width | 80% coverage | 80% width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| component MC (raw) | 1,225 | 4.45 | 41.2% | 6.88 | 74.9% | 13.40 |
| component MC + walk-forward conformal | 921 | 4.19 | 50.5% | 7.98 | 83.0% | 14.94 |
| position-normal baseline | 921 | 4.37 | 54.3% | 7.52 | 84.6% | 13.04 |
| usage nearest-neighbor baseline | 921 | 4.24 | 62.4% | 8.19 | 88.9% | 14.89 |

The raw simulator is under-dispersed. That is a useful finding: the data do not support claiming it is well calibrated as-is. The conformal version improves its median error and lands close to nominal 50%/80% coverage while preserving role-dependent distribution shapes. The nearest-neighbor baseline is conservative and better covered, but quite wide and less structurally explanatory.

For a take-home submission, recommend the **component MC + walk-forward conformal calibration** as the product model, show the raw model and baselines in the model card, and tune calibration by position only after obtaining more historical forecast weeks. The exact machine-readable metrics and calibration plot are created by `fantasy-ranges backtest` in `outputs/backtest/` and intentionally remain untracked because they are reproducible artifacts.
