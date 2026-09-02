# Week 1 fantasy range projections

A reproducible, leakage-safe framework for projecting **PPR fantasy-point distributions** for RBs, WRs, and TEs before Week 1. It deliberately models a stat-generating process rather than regressing directly on past fantasy points:

`team volume → player share → opportunities → efficiency → touchdowns → fantasy points`

The primary model is a partially pooled Monte Carlo simulator. It is compared with two deliberately simpler, honest baselines:

- **Position-normal:** a player-specific center plus a position-specific residual standard deviation.
- **Usage nearest-neighbors:** observed games from similar prior-season usage profiles.

The project is designed around a key Week 1 insight: a rookie or team changer can share a median with a stable veteran without sharing their interval width.

## Current 2026 Week 1 projections

The repository includes a real roster-based Week 1 run (not the synthetic demo)
in `outputs/week1_2026/projections.csv`. It covers 422 active RBs, WRs, and TEs
from the 2026 nflverse Week 1 roster snapshot. Recreate or refresh it with:

```bash
fantasy-ranges project-week1 \
  --input data/raw/player_games.csv \
  --season 2026 \
  --out-dir outputs/week1_2026 \
  --simulations 20000
```

This command applies the historical walk-forward conformal calibration by
default. Pass `--no-calibrate` only when inspecting the raw component model.

Read [`docs/methodology.md`](docs/methodology.md) for a plain-English
explanation of the model, its assumptions, output fields, and a map from each
modeling step to the responsible file/function.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,parquet]'
fantasy-ranges demo --out-dir outputs/demo
pytest -q
```

On macOS/Homebrew, installing directly with `python3 -m pip install ...` is
intentionally blocked by PEP 668. Activate the `.venv` above first; do not use
`--break-system-packages` for this project.

To open the guided walkthrough notebook, install its optional UI dependency and
launch Jupyter from the project root:

```bash
python -m pip install -e '.[notebook]'
python -m ipykernel install --user --name fantasy-ranges --display-name "Python (fantasy-ranges)"
jupyter lab notebooks/week1_projection_walkthrough.ipynb
```

In Jupyter, select the **Python (fantasy-ranges)** kernel. The notebook also
adds the local `src/` directory to its import path as a fallback, but its
dependencies must still be installed in the selected environment.

To run a real historical backtest (downloads public nflverse data; CSV works without `pyarrow`):

```bash
fantasy-ranges fetch --seasons 2021 2022 2023 2024 2025 --data-dir data/raw
fantasy-ranges backtest --input data/raw/player_games.csv --seasons 2022 2023 2024 2025 --out-dir outputs/backtest
```

For an interactive view after producing a projection CSV:

```bash
python -m pip install -e '.[app]'
streamlit run app.py -- --projections outputs/week1_2026/projections.csv
```

## Data availability and final schema

The ingestion layer uses nflverse's weekly `stats_player` release as the canonical box-score source. It supplies player-game IDs, positions, opponent, carries, targets, receptions, rushing/receiving yards and TDs, PPR fantasy points, target share, and air-yards measures. Team attempts/plays and game context should be joined from nflverse play-by-play or team stats. FTN charting is a useful *postseason historical* supplement for route participation from 2022 onward; it is not suitable as a guaranteed in-season dependency. See [`docs/data_sources.md`](docs/data_sources.md).

The normalized modeling table has one row per `player_id, season, week, team` and keeps both realized columns (only as labels) and prior-only columns used for predictions:

| Group | Fields |
| --- | --- |
| identity/context | player_id, player_name, position, season, week, team, opponent, game_id |
| realized label | fantasy_points_ppr, targets, receptions, receiving_yards, carries, rushing_yards, touchdowns |
| prior-only usage | prior_target_share, prior_rush_share, prior_targets, prior_carries, prior_routes, prior_team_pass_attempts |
| role evidence | games_sample, late_season_share, same_team, experience, rookie, team_change, role_uncertainty |
| environment | expected_team_pass_attempts, expected_team_rush_attempts, expected_points, spread |

## Modeling approach

### Recommended: hierarchical component simulator

For each player, the model builds shrinkage priors from their historical samples and positional samples. A role-uncertainty score converts into lower beta concentration: stable roles have narrow target/carry share draws; rookies, team changers, small samples, and unstable recent usage have wide draws. Each simulation draws:

1. Team pass and rush volume (negative binomial / gamma-Poisson dispersion)
2. Player target and carry shares (beta distributions)
3. Targets and carries (beta-binomial, which permits over-dispersion)
4. Receptions (beta-binomial) and yards (gamma with player/position shrinkage)
5. Rushing and receiving TDs (Poisson with a positional shrinkage rate)

The implementation uses posterior-style pseudo-count pooling rather than implying every player has enough data for a bespoke distribution.

### Backtesting protocol

`backtest` holds out each historical Week 1 and creates features strictly from earlier seasons. That makes it an actual preseason-style test. It reports median AE, pinball loss at P10/P25/P50/P75/P90, 50% and 80% coverage, and interval width. Any extension that uses prior *weeks* must use a lagged as-of feature table; the builder intentionally never uses same-game outcomes as inputs.

## What to ship in a take-home

1. Use the simulator as the candidate model and include the baselines in the appendix/table.
2. Build 2026 priors from 2025 plus decayed 2021–24 history. Supply a reviewed `data/inputs/week1_2026_roles.csv` with team, expected pass/rush volume, player role overrides, injury/rookie/team-change flags, and (optionally) consensus centers.
3. Calibrate global position/role multipliers on historical Week 1 coverage. Do this before examining 2026 outcomes.
4. Show a player table with median, P25–P75, P10–P90, `P(10+)`, `P(15+)`, `P(20+)`, and role uncertainty. Explain that touchdown uncertainty is inherently large.

## Limitations worth saying explicitly

- Box-score data does not provide complete historic routes/snaps. The code accepts routes as an optional enhancement, not a silent fabricated feature.
- Team and coaching changes are represented through wider priors unless curated external projections/role inputs are provided. A good production version should version those human inputs.
- Touchdowns are noisy. The model shrinks TD rates strongly by design; a player-specific TD rate is never treated as stable from a handful of games.
- Public weekly projection archives are not a required dependency. If a dated archive is obtained, pass its consensus center into `ProjectionRequest` and validate the residual/hybrid variant separately.

## Repository map

- `src/fantasy_ranges/data.py` — nflverse ingestion and schema normalization
- `src/fantasy_ranges/features.py` — as-of preseason features and role uncertainty
- `src/fantasy_ranges/baselines.py` — position-normal and weighted-neighbor baselines
- `src/fantasy_ranges/simulation.py` — component Monte Carlo model
- `src/fantasy_ranges/backtest.py` — leakage-safe Week 1 evaluation and calibration graphics
- `src/fantasy_ranges/demo.py` — deterministic synthetic example so CI/demo works offline
- `app.py` — lightweight distribution explorer

Data attribution: nflverse data are CC-BY-4.0; FTN-derived participation/charting data require the attribution described in their release documentation.
