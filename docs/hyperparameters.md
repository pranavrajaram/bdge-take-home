# Model constants and hyperparameters

This is the honest inventory of the non-data values in the current prototype.
Some are scoring/output conventions, some are numerical safeguards, and some
are modeling assumptions. The modeling assumptions were selected as reasonable
starting priors and checked through the Week 1 backtest; they were not
estimated by an optimizer.

## Main assumptions

| Constant | Location | Meaning and effect | Status / what to do |
| --- | --- | --- | --- |
| 0.60 | features.py, weighted mean | Annual recency decay. For three equal-length prior seasons, the most recent season has 51.0% of the weight. | Heuristic. Tune through nested walk-forward validation over .50 to .90. |
| 48 | simulation.py, shrink | Number of prior player games at which positional pooling reaches zero. | Heuristic. Roughly three full seasons; prevents an established starter being pulled toward a prior that includes backups. |
| 14 | simulation.py, shrink | Initial position-prior pseudo-games for target/carry share, tapered linearly to zero by 48 prior games. | Heuristic. Most relevant to sparse histories. |
| 20 | shrink | Initial pseudo-games for catch rate and yards per reception/carry, tapered to zero by 48 games. | Heuristic. More pooling for limited efficiency samples. |
| 30 | shrink | Initial pseudo-games for TD rate, tapered to zero by 48 games. | Heuristic. Stronger pooling for sparse TD samples. |
| 0.18, 0.20 | simulate | Latent coefficients of variation for team pass and rush volume. | Heuristic. Estimate historical team-game residual variation instead. |
| 70, 5 | simulate | Role-share beta concentration: 70 times (1 minus uncertainty), plus 5. | Heuristic. Sets role-driven range width. |
| 45 times (1 minus uncertainty / 2) | simulate | Catch-rate beta concentration. | Heuristic. Says catch rates vary less than workload shares. |
| .42, .38 | simulate | Gamma CV for receiving yards/reception and rushing yards/carry. | Heuristic. Estimate by position/opportunity tier. |
| .64, .45, .82 | data.py, simulation.py | Targetable-pass fallback and lower/upper guardrails. | Data-quality heuristic. Material: the RB rate reaches the .82 ceiling. |
| 1.2, 1.0 | simulate | Caps receiving/rushing TD Poisson rates. | Guardrails against malformed input, not football laws. |
| 12 | features.py, matchup prior | League-average pseudo-games added to each defense-position points-allowed rate. | Heuristic shrinkage for noisy defense performance; tune only through walk-forward validation. |
| .92, 1.08 | features.py, simulation.py | Lower and upper bounds for the Week 1 opponent multiplier. | Intentional ±8% cap: matchup should refine a forecast, not dominate it. |

## Fallback values that normally do not affect the real 2026 run

These are still constants, but they only apply when data are absent or a caller
bypasses the normal feature-builder path.

| Value | Location | Used when |
| --- | --- | --- |
| target share `.15`, rush share `.10` | simulation fit | A position has no usable historical observations. |
| catch rate `.63` | simulation fit | Same missing-position fallback. |
| yards/reception `10.5`, yards/carry `4.2` | simulation fit | Same missing-position fallback. |
| rec. TD rate `.045`, rush TD rate `.025` | simulation fit | Same missing-position fallback. |
| targetable-pass rate `.64` | simulation fit / data fallback | No usable targetable-pass rate or raw pass attempts. |
| team pass/rush `34`, `25` | simulate | A caller supplies no team-volume feature at all. Real roster projections do not use these. |
| role uncertainty `.50` | simulate | A caller supplies no role-uncertainty feature. Real feature-builder output does. |
| usage-volatility `.20` | feature builder | A player has no usable volatility measurement. |
| position-normal residual SD `7.0` | baseline | A position has no fitted residual scale. Baseline only. |

The probability clip `.001–.999`, minimum beta concentration `.4`, minimum
gamma mean `.05`, minimum gamma CV `.08`, and final uncertainty clip
`.02–1.00` are numerical-validity safeguards. The uncertainty display labels
also use arbitrary cut points: low below `.33`, medium below `.66`, high at or
above `.66`; those labels do not affect the simulations.

## Exact formulas

### Recency weighting

For target season T and historical season s:

~~~text
raw_weight = 0.60 ^ (T - s)
weighted_rate = sum(raw_weight times observation) / sum(raw_weight)
~~~

The decay applies to every player-game row, not once per season. A season's
total contribution is therefore its games played multiplied by its per-game
decay. For three 17-game seasons (2023–25), the normalized weights are
18.4%, 30.6%, and 51.0%. McCaffrey's 2026 target-share calculation has a
larger 2025 share because he played only four games in 2024: 2021: 3.1%,
2022: 12.4%, 2023: 19.4%, 2024: 8.1%, and 2025: 57.2%.

### Shrinkage

~~~text
effective_strength = strength times max(0, 1 - player_games / 48)
shrunken_rate = (player_games times player_rate + effective_strength times position_rate)
                / (player_games + effective_strength)
~~~

Strength is 14 for shares, 20 for efficiency, and 30 for TD rates. These act
like pseudo-games, not literal games, and taper away for established players.

### Expected team volume

There is no separate learned team-volume model. For each 2026 player:

~~~text
expected_pass = mean(one row per destination-team game in 2025)
expected_rush = mean(one row per destination-team game in 2025)
~~~

QB attempts/carries are included. The code deduplicates to one season, week,
team row before averaging. SF's 2025 values are 33.76 pass attempts and 28.29
rush attempts across 17 games. Manual input overrides this baseline.

The simulation adds variance:

~~~text
pass attempts = Poisson(Gamma(mean=expected_pass, CV=.18))
rush attempts = Poisson(Gamma(mean=expected_rush, CV=.20))
~~~

For a Gamma-Poisson variable with mean m and latent CV c, approximate variance
is m plus (c times m) squared. SF's pass-attempt standard deviation is about
8.41, not zero.

### Role uncertainty

~~~text
sample_penalty = max(0, 1 - min(games, 32) / 32)
instability = min(target_share_sd / .25, 1)

uncertainty = max(manual_override,
    .08 + .32 times sample_penalty + .22 times team_change
        + .22 times rookie + .12 times instability)
~~~

The .08 is a stable-veteran floor; 32 is roughly two full seasons; .32, .22,
.22, and .12 are heuristic penalties; .25 is the target-share SD considered
maximally unstable. None are fitted weights.

~~~text
role_concentration = 70 times (1 - uncertainty) + 5
~~~

Lower concentration gives a wider beta-binomial target/carry share.

## Values that are not modeling assumptions

| Value | Meaning |
| --- | --- |
| 1 reception plus .1 yards plus 6 TDs | Full-PPR scoring convention. |
| P10/P25/P50/P75/P90 | Requested forecast outputs. |
| 10, 15, 20 | Product threshold probabilities only. |
| 20,000 current simulations | Monte Carlo precision/compute choice. |
| Normal z-scores in the normal baseline | Mathematical standard-normal quantiles, not tuned. |
| 100 neighbors and .08 distance floor | Nearest-neighbor baseline only, not production forecast. |
| .001 to .999 probability, .4 concentration, .05 gamma mean, .08 gamma CV floors | Numerical safeguards for sparse/bad data. |

## Calibration does and does not fix this

The output uses walk-forward conformal calibration: it adjusts final quantiles
using completed historical Week 1 residuals. That improved 80% coverage from
74.9% raw to 83.0% in the backtest. It makes final intervals more honest, but
does not prove internal values above are optimal.

## Tuning order

1. Replace targetable-pass fallback/clip and historical team volume with
   play-level opportunities plus dated totals/spreads.
2. Fit role-uncertainty weights and beta concentration to future target/carry
   share variance.
3. Tune recency decay and shrinkage strengths through nested walk-forward
   pinball loss and coverage.
4. Fit yardage and TD residual distributions by position/opportunity tier.

The proper claim today is: transparent priors plus calibrated output intervals,
not fully learned internal parameters.
