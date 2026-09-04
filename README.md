# Week 1 fantasy range projections

A reproducible preseason model for projecting **Week 1 PPR fantasy-point ranges** for RBs, WRs, and TEs. It produces a distribution for each player—not just one point estimate—so the output can communicate both expected production and role uncertainty.

The public walkthrough is the best place to start: [project demo](https://pranavrajaram.github.io/bdge-take-home/). It uses the real 2026 Week 1 player pool and works through the model with concrete inputs and outputs.

## Explore the published work

- [Interactive project walkthrough](https://pranavrajaram.github.io/bdge-take-home/)
- [Full 2026 Week 1 projection table](https://pranavrajaram.github.io/bdge-take-home/projections.html)
- [Methodology](https://pranavrajaram.github.io/bdge-take-home/methodology.html)
- [Hyperparameters and sensitivity choices](https://pranavrajaram.github.io/bdge-take-home/hyperparameters.html)

## How the model works

The model works from football events to fantasy points instead of predicting prior fantasy scores directly:

`team pass/rush attempts → player opportunity share → targets/carries → catches/yards/TDs → PPR points`

Each stage is simulated many times. This lets a player with a stable role receive a narrower range than a rookie, a team changer, or a player whose prior usage was volatile. Historical player rates are shrunk toward position-level evidence when a player has a limited sample, and recent seasons receive more weight than older ones. A known Week 1 opponent adds a small, position-specific adjustment based on prior fantasy points allowed, with heavy league-average shrinkage and a ±8% cap.

Past fantasy points are used only as held-out labels when evaluating forecasts. They are not a model input. Modeling the components makes the calculation inspectable and avoids treating a previous final score as a durable skill independent of volume, role, and touchdowns.

The project compares the component simulator with two simple benchmarks: a position-normal distribution and a usage-nearest-neighbors model. Evaluation holds out each historical Week 1 and builds every feature only from prior seasons; conformal calibration then adjusts published intervals to better match their advertised coverage.

## Where the analysis lives

| File | Responsibility |
| --- | --- |
| `notebooks/week1_projection_walkthrough.ipynb` | Notebook version of the end-to-end workflow. |
| `src/fantasy_ranges/data.py` | Downloads nflverse player-game data, normalizes the schema, and assembles the projection player pool. |
| `src/fantasy_ranges/features.py` | Builds preseason priors, recency weights, team-volume expectations, and role uncertainty. |
| `src/fantasy_ranges/simulation.py` | Runs the component-level Monte Carlo simulation and converts simulated stats to PPR points. |
| `src/fantasy_ranges/backtest.py` | Runs leakage-safe, walk-forward Week 1 evaluation and interval calibration. |
| `src/fantasy_ranges/projections.py` | Generates the production projection table. |
| `src/fantasy_ranges/baselines.py` | Defines the simple benchmark models used in evaluation. |
| `docs/week1_projection_demo.html` | The presentation-ready public walkthrough. |
| `docs/methodology.html` | Detailed implementation guide with links to the relevant functions. |
| `docs/hyperparameters.html` | Rationale for model constants and sensitivity choices. |

## Reproduce locally

Python packages should be installed in a virtual environment, particularly on macOS/Homebrew systems that enforce PEP 668.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,parquet,notebook]'
```

Run the test suite:

```bash
pytest -q
```

Fetch the public historical data and run the backtest:

```bash
fantasy-ranges fetch --seasons 2021 2022 2023 2024 2025 --data-dir data/raw
fantasy-ranges backtest --input data/raw/player_games.csv --seasons 2022 2023 2024 2025 --out-dir outputs/backtest
```

Build a Week 1 projection run:

```bash
fantasy-ranges project-week1 \
  --input data/raw/player_games.csv \
  --season 2026 \
  --out-dir outputs/week1_2026 \
  --simulations 20000
```

The command applies walk-forward conformal calibration by default. Use `--no-calibrate` only to inspect the uncalibrated component model.

## Data and scope

The historical box-score source is nflverse weekly `stats_player` data. The model covers RB, WR, and TE projections under 1-PPR scoring: one point per reception, 0.1 points per rushing or receiving yard, and six points per touchdown. The data source and field choices are described in the public [methodology](https://pranavrajaram.github.io/bdge-take-home/methodology.html).

The model is intentionally a transparent preseason baseline, not a claim that it has perfect injury, depth-chart, coaching, or team-context information. Those unknowns are reflected primarily through wider ranges. Touchdowns remain particularly noisy and are deliberately shrunk toward position-level rates.
