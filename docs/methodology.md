# How the Week 1 projection model works

This page explains the model in plain language. The short version is: instead
of guessing one fantasy-point number, we estimate the chain of events that can
produce points and repeat that chain many thousands of times.

For example, a WR can score because his team passes often, he earns a large
share of those passes, he catches them efficiently, gains yards, and perhaps
scores. Each part is uncertain. The final projection is the collection of all
those plausible simulated games.

## What a Week 1 projection means

`P10`, `P25`, `P50`, `P75`, and `P90` are percentiles of simulated PPR fantasy
points.

| Output | Plain-English meaning |
| --- | --- |
| P10 | In 90% of simulated games, the player scored more than this. It is a reasonable low-end outcome, not a guaranteed floor. |
| P50 / median | Half of simulations are above this number and half below it. This is the best single-number summary. |
| P90 | Only 10% of simulations exceed this. It is a plausible ceiling, not a prediction of the player’s best game. |
| P(10+), P(15+), P(20+) | The share of simulated games that clear each scoring threshold. |
| role uncertainty | Low / medium / high confidence in workload, not confidence that the player is talented. |

The project uses PPR scoring: one point per reception, 0.1 per rushing or
receiving yard, and six per rushing or receiving touchdown.

## The actual 2026 Week 1 player pool

The projection file is generated from nflverse's 2026 Week 1 roster snapshot.
It includes active RBs, WRs, and TEs—not just well-known fantasy starters. That
is intentional: anyone on the active roster is a possible outcome candidate,
though players with no meaningful NFL history usually have low medians and wide
workload uncertainty.

The generated artifact is calibrated using completed historical Week 1 forecasts:

- `outputs/week1_2026/projections.csv` — 422 active RB/WR/TE projections
- `outputs/week1_2026/candidate_roster.csv` — the exact roster universe used

The roster decides **who is eligible to project**. It does not pretend to know
the exact depth chart or Week 1 workload. Those uncertainties remain in the
model. Refresh the roster shortly before lineup decisions; it is a dated public
snapshot, not a live injury feed.

## Step 1: collect historical player games

We load 2021–25 nflverse weekly player-stat files. Each raw row is a player
game. We retain RB, WR, and TE rows for the fantasy output, but aggregate team
opportunities before filtering so QB pass attempts still count toward team
volume.

| Job | File | Function | What it does |
| --- | --- | --- | --- |
| Download box scores | `src/fantasy_ranges/data.py` | `fetch_player_stats` | Downloads/caches the nflverse weekly stats CSVs. |
| Clean and aggregate | `src/fantasy_ranges/data.py` | `normalize_player_games` | Standardizes column names and creates team targets, rush attempts, pass attempts, player target share, and player rush share. |
| Download current roster | `src/fantasy_ranges/data.py` | `fetch_roster` | Downloads/caches the season roster release. |
| Select Week 1 candidates | `src/fantasy_ranges/data.py` | `week1_candidates_from_roster` | Keeps active RB/WR/TEs, maps IDs/names, and flags rookies. |

The normalized historical table has both actual outcomes (targets, yards,
touchdowns, fantasy points) and team context. Actual outcomes are labels for
history/backtesting—not features that leak into a future prediction.

## Step 2: create a preseason player prior

Week 1 is different from Week 8: there are no current-season targets or carries.
For the 2026 output, the feature builder uses only 2021–25 results.

For an established player, it calculates recency-weighted history for:

- target share and rush share;
- catches per target, yards per catch, and rushing yards per carry;
- receiving and rushing TD rates;
- sample size and recent usage volatility.

Older seasons are downweighted, so last season matters most without treating a
single season as the whole truth. A player with few games is pulled toward the
typical player at his position. This is a practical version of “partial pooling”:
we let a player be himself when there is enough evidence and otherwise borrow
strength from similar players.

Destination-team volume comes from that team’s most recent completed season.
More precisely, the feature builder takes one distinct `season, week, team`
row for every regular-season game, then calculates the simple arithmetic mean:

```text
expected_team_pass_attempts = mean(team pass attempts in the team's 2025 games)
expected_team_rush_attempts = mean(team rush attempts in the team's 2025 games)
```

QB pass attempts and QB carries are included in the team totals. The player
table repeats those totals on several player rows, so the code explicitly
deduplicates to one team-game before averaging; otherwise games with more
recorded RB/WR/TEs would be overweighted. For example, SF's 17 2025 games yield
33.76 pass attempts and 28.29 rush attempts per game. A 2026 SF player starts
with those means; a player changing teams inherits his destination team's mean,
not his old team's opportunity pool. A manually supplied team-volume input
overrides this baseline.

These are baseline means, not fixed simulation values. The simulator turns the
pass mean into a Gamma-Poisson draw (18% latent-volume coefficient of
variation) and the rush mean into a Gamma-Poisson draw (20% CV), then draws the
actual game’s attempts from those rates. Team-total/spread, opponent, QB, and
coaching projections are not currently included and should be added as dated,
auditable overrides.

| Job | File | Function | What it does |
| --- | --- | --- | --- |
| Build historical player prior | `src/fantasy_ranges/features.py` | `_player_prior` | Computes recency-weighted historical rates and samples. |
| Build final Week 1 features | `src/fantasy_ranges/features.py` | `build_preseason_features` | Enforces `season < target_season`, shrinks small samples, identifies team changes, and assigns destination-team volume. |

### Role uncertainty

Role uncertainty rises when the evidence for a stable workload is weak:

- few previous NFL games;
- rookie status;
- a new team;
- unstable prior target share; or
- a manually supplied uncertainty override.

It does **not** simply add a few points to an interval. Instead it makes the
underlying target-share and carry-share distributions less concentrated. A
stable veteran’s simulated shares remain close to his historical role more
often; a rookie’s share can range from a small role to a meaningful one. This
is why two players can have equal medians but different floors and ceilings.

## Step 3: simulate the football events

`ComponentSimulator` is the core model. It repeats the following process for a
player 10,000–20,000 times:

1. Draw a plausible number of team pass attempts and rush attempts.
2. Draw the player’s target share and carry share. Higher role uncertainty means
   a wider draw.
3. Turn shares into discrete targets and carries.
4. Draw receptions conditional on targets.
5. Draw positive, right-skewed yards conditional on receptions/carries.
6. Draw receiving and rushing touchdowns using strongly shrunk TD rates.
7. Convert the resulting stat line to PPR fantasy points.

The code uses beta-binomial draws for opportunities. In everyday terms, a plain
binomial model acts as if a player’s chance of a target is fixed for every play.
The beta-binomial model first allows that chance itself to move around from game
to game, which better represents uncertain roles. Yardage uses gamma draws so
negative yards cannot appear just because a normal distribution happened to
draw below zero.

| Job | File | Function / class | What it does |
| --- | --- | --- | --- |
| Fit position anchors | `src/fantasy_ranges/simulation.py` | `ComponentSimulator.fit` | Learns position-level catch, yardage, TD, share, and targetable-pass priors. |
| Simulate a player | `src/fantasy_ranges/simulation.py` | `ComponentSimulator.simulate` | Produces one player’s fantasy-point samples. |
| Summarize samples | `src/fantasy_ranges/simulation.py` | `Distribution.summary` | Calculates percentiles and threshold probabilities. |
| Create product table | `src/fantasy_ranges/projections.py` | `project_week1` | Runs priors + simulator and formats player-facing columns. |

## Step 4: compare with simpler models

The component model should earn its complexity. We therefore compare it against
two simpler baselines:

| Baseline | File / function | Intuition |
| --- | --- | --- |
| Position-normal | `src/fantasy_ranges/baselines.py` → `PositionNormalBaseline` | Start with a component point center and add the typical historical error width for that position. |
| Usage neighbors | `src/fantasy_ranges/baselines.py` → `UsageNearestNeighbors` | Find past player-games with similar target share, rush share, prior volume, and uncertainty; use their observed outcomes as the range. |

These are valuable controls. If the sophisticated model is less calibrated than
a simple baseline, the answer is to improve/calibrate it—not to hide the
comparison.

## Step 5: check whether the ranges are honest

Point error alone is not enough. A model could have a decent median while giving
intervals that are far too narrow. The backtest holds out Week 1 of each season
from 2022–25 and only lets the model see earlier completed seasons.

We report:

- **median absolute error:** how far the middle forecast is from reality;
- **pinball loss:** error quality at each requested percentile;
- **50% / 80% coverage:** how often the actual score landed in the claimed
  interval; and
- **interval width:** how wide that claimed interval was.

The raw component model was under-dispersed in the first backtest. That is a
useful discovery, not a failure to conceal. `QuantileConformalizer` learns an
additive correction from *earlier* Week 1 forecast errors and applies it to the
next held-out year. It improved observed coverage toward the declared levels.

| Job | File | Function / class | What it does |
| --- | --- | --- | --- |
| Build historical Week 1 examples | `src/fantasy_ranges/backtest.py` | `build_week1_panel` | Creates labeled rows with only prior-season features. |
| Run walk-forward backtest | `src/fantasy_ranges/backtest.py` | `run_week1_backtest` | Repeats fit → predict → reveal actual outcome across seasons. |
| Score distributions | `src/fantasy_ranges/backtest.py` | `evaluate` | Calculates coverage, interval width, pinball loss, and median AE. |
| Calibrate quantiles | `src/fantasy_ranges/backtest.py` | `QuantileConformalizer` | Corrects future quantiles using only previous forecast residuals. |

See `docs/backtest_results.md` for the measured historical result and
`outputs/backtest/calibration.png` for the coverage chart.

## How to refresh actual Week 1 projections

Run this from the activated project virtual environment:

```bash
fantasy-ranges project-week1 \
  --input data/raw/player_games.csv \
  --season 2026 \
  --out-dir outputs/week1_2026 \
  --simulations 20000
```

If `--roster` is omitted, the command fetches/caches the public 2026 roster
release in `data/inputs/`. To use a reviewed/offline roster file, pass
`--roster path/to/roster_2026.csv`. The CLI entry point itself is
`src/fantasy_ranges/cli.py` → `main`.

## Important limitations and sensible next improvements

This is a statistically honest Week 1 baseline, not a claim of omniscience.

1. **The roster is not a depth chart.** Active status says a player can play;
   it does not say whether he starts. Add a reviewed role/depth-chart input and
   use it as a documented prior override.
2. **No live injury/preseason feed is used.** Refresh roster/injury information
   close to kickoff and widen or suppress players with uncertain availability.
3. **No opponent/game-total inputs are yet applied.** Add schedules, projected
   team totals, spreads, and defensive context as separately dated inputs.
4. **Routes and red-zone work are not in the core public box-score model.** FTN
   charting can enhance historical priors, but its availability must be treated
   carefully and it should never be silently backfilled as a live feature.
5. **TDs remain noisy.** The model intentionally shrinks player TD rates toward
   positional averages rather than turning last year’s TD total into a promise.
6. **Calibration is historical, not clairvoyant.** The current projection run
   uses completed 2022–25 Week 1 residuals to calibrate the component model's
   quantiles. It cannot account for a new 2026-specific factor that is absent
   from the roster and historical inputs.

The guided notebook (`notebooks/week1_projection_walkthrough.ipynb`) is the
best place to see the main calculations in sequence; the package files above
are the production implementation.

For an explicit inventory of every numerical assumption, its effect, and
whether it is a tuned parameter or a prototype heuristic, see
[`hyperparameters.md`](hyperparameters.md).
