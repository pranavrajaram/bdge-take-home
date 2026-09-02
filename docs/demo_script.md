# Demo script: Week 1 Fantasy Range Projections

Use this as an 8–9 minute video script. The spoken words are conversational;
the italic Show notes describe what to put on screen.

## Before recording

Open these pages in order:

1. docs/week1_projection_demo.html
2. docs/projections.html
3. docs/methodology.html
4. docs/hyperparameters.html

Start on the main demo page. Keep the projection page available for a quick
player lookup and use the Puka calculation widget when prompted below.

## 0:00 — What this project does

> This project produces Week 1 PPR fantasy projections as ranges, not just one
> point estimate. Before Week 1, we do not know a player’s exact role, team
> play volume, efficiency, or touchdown outcome.
>
> So instead of saying “Puka Nacua will score exactly 15.8 points,” the model
> simulates many plausible stat lines. That gives us a median, a realistic low
> end, a realistic high end, and probabilities of clearing useful thresholds.
>
> I’ll walk through the data, how player priors are built, how uncertainty
> enters the simulation, how I check the ranges historically, and finally what
> the current Week 1 output looks like.

*Show: the top of week1_projection_demo.html. Point briefly to the six steps
and the supporting-page links.*

## 0:35 — Why this is not a traditional ML model

> Before getting into the pipeline, I want to explain one design choice. I did
> not start with a black-box model trained to predict fantasy points directly.
>
> The target is noisy and Week 1 is a small, unusual prediction problem. There
> are only a handful of historical Week 1s, player roles change in the
> offseason, and a single fantasy-point label combines several different
> football events: opportunity, efficiency, and touchdowns.
>
> A traditional model can still be useful as a benchmark or a future ensemble
> component. But for this project, a component model is easier to audit. I can
> show how a projection changes because of team volume, target share, carry
> share, catch rate, yards, or touchdown probability. It also gives a natural
> distribution rather than bolting an interval onto a point prediction later.
>
> I also intentionally do not feed historical fantasy points into the player
> prior. Fantasy points are an outcome, not a football role. Using them
> directly would hide whether a prior score came from repeatable workload or a
> touchdown spike. Instead, the model uses the components that create PPR
> points—targets, carries, receptions, yards, and touchdowns—and only converts
> them to fantasy points at the final step.
>
> That does not mean prior fantasy points disappear from the project. They are
> still the historical labels used to evaluate forecasts and to compare against
> the simpler baselines. They are just not used as a shortcut inside the
> production prior.

*Show: the Step 4 simulation sequence. Say “these are the parts of fantasy
points that the model estimates separately.”*

## 1:35 — Step 1: Data and the player pool

> The historical source is weekly nflverse box-score data from 2021 through
> 2025. The model works at the player-game level, but it also builds the team
> opportunity pool for each game.
>
> That matters because a player’s targets and carries need a denominator. In
> the San Francisco Week 1 example, McCaffrey had 10 targets out of 32 team
> targets, or a 31.3% target share. He had 22 of 31 team rushes, or a 71.0%
> rush share.
>
> A key implementation detail is that team pass attempts are calculated before
> filtering to RBs, WRs, and TEs. Quarterback attempts still count in the team
> total even though QBs are not projected in this version.
>
> For the actual 2026 player universe, I start with the active Week 1 roster
> snapshot and keep active RBs, WRs, and TEs. That produces 422 candidates, so
> the model is not limited to a hand-picked list of stars.

*Show: Step 1, then click Complete projections and briefly locate Puka Nacua
and Jaxon Smith-Njigba. Return to the demo.*

## 2:30 — Step 2: Player priors

> Week 1 has no current-season usage, so the projection starts from a prior:
> what we knew about the player before the season began.
>
> I use recency weighting. Each historical player-game is weighted by 0.60 to
> the number of seasons ago. For a player with three full recent seasons, the
> most recent season gets about 51% of the weight. That gives recent
> information real influence without pretending older history does not exist.
>
> McCaffrey is a useful example because the table shows the arithmetic. His
> 2025 season gets 57.2% of his weight because he missed most of 2024. The
> inputs include target share, rush share, catch rate, yards per opportunity,
> and touchdown rates.
>
> The second part of the prior is positional pooling. A rookie or a player
> with only a few games should not be treated as if a tiny sample is the whole
> truth, so the model blends that player toward the typical value for his
> position. But that blend tapers to zero by 48 prior games. An established
> player like McCaffrey or Bijan is not artificially pulled toward an RB
> average that includes backups.
>
> Team volume is separate from player talent. For a 2026 San Francisco player,
> the starting point is the average of SF’s 2025 team games: 33.76 pass
> attempts and 28.29 rush attempts. Those are starting means, not fixed
> inputs—every simulation can draw a different total.

*Show: the recency-weight table, the two-prior-phases table, and the
destination-team-volume formula.*

## 4:00 — Step 3: Role uncertainty

> The model treats uncertainty as uncertainty about workload, not as a vague
> extra number of fantasy points.
>
> It looks at sample size, rookie status, team changes, and how volatile prior
> target share was. A stable veteran gets a tighter distribution of target and
> carry shares. A rookie or a player changing teams gets a wider distribution:
> he can still earn a meaningful role, but the model admits that we know less.
>
> The contrast on screen is deliberate. McCaffrey has 61 prior games and low
> role uncertainty. Jonah Coleman has no prior NFL sample, so his possible
> workload is much wider. This affects the shape of the outcome range rather
> than mechanically subtracting points because someone is a rookie.

*Show: Step 3’s uncertainty rule and the McCaffrey / Coleman table.*

## 4:45 — Step 4: From football events to fantasy points

> The core model simulates the football events in order: team pass and rush
> attempts, player targets and carries, receptions, yards, touchdowns, and
> finally PPR points.
>
> One term that can sound confusing is targetable-pass rate. It is an
> intermediate bridge from team pass attempts to the pool of recorded targets.
> It is not a player’s target share or catch rate. The player’s own target
> share is applied after that pool is formed.
>
> Let me show the end-to-end arithmetic with Puka Nacua.

*Show: scroll to the calculation widget and choose Puka Nacua, WR / LA.*

> These are real means from 100,000 raw simulation draws. Puka averages 8.63
> targets and 6.28 receptions. The simulated mean stat line has 82.47
> receiving yards, 2.53 rushing yards, and 0.44 total touchdowns.
>
> The scoring line makes the result auditable: 6.28 reception points, 8.50
> yardage points, and 2.62 touchdown points, for a 17.40 raw mean.
>
> The raw median is 15.80. The reported P50 is 15.78 after calibration. The
> mean is higher than the median because touchdowns create occasional large
> outcomes, which pull the average up. That is why a range is more useful than
> presenting only one number.

## 6:00 — Step 5: Are the ranges honest?

> A range model should be judged on more than whether its middle estimate is
> close. It also has to be honest about uncertainty.
>
> I backtest by replaying past Week 1s. For the 2025 test, the model only gets
> 2021 through 2024 information, then I compare its forecast with the real
> 2025 Week 1 result. The same process is repeated for each available year.
>
> The most intuitive score is median absolute error: it tells us the typical
> size of the P50 miss. Coverage checks the ranges. If I label P10 to P90 an
> 80% range, the actual score should land inside it about 80% of the time.
>
> Here the raw component model covered 76% of outcomes with its advertised
> 80% range. In plain language, it was a little too confident—its ranges were
> too tight.
>
> Walk-forward calibration is the correction for that. Before forecasting a
> new year, I look only at earlier forecast errors and ask whether the
> percentiles need to move outward or inward. For the 2025 test, the
> correction learns from 2022–24 errors only; it cannot peek at the 2025 result
> it will be graded on. For 2026, it uses completed 2022–25 Week 1 errors.
>
> After calibration, the observed 80% coverage is 87%. That is now
> conservative rather than perfectly calibrated, so it is an area to improve,
> not a number I am trying to hide. The position-normal baseline is included
> as a reality check, and it performs similarly on typical point error.

*Show: the How to read the results table, the calibration explanation, then
the walk-forward backtest table. Optionally open the calibration plot.*

## 7:25 — Step 6: Current projections and takeaway

> The current output is the same process applied to the active 2026 Week 1
> pool.
>
> The table gives P10, P25, P50, P75, P90, threshold probabilities, and role
> uncertainty. The P50 is the best concise center. P10 and P90 are plausible
> low- and high-end outcomes, not guarantees.
>
> For example, Puka’s current P50 is 15.78, with a P10 of 6.37 and a P90 of
> 33.18. That is a concise statement of both opportunity and risk. Bijan’s
> mean is 17.89 and his P50 is 16.38; the difference again reflects
> right-skewed touchdown outcomes.
>
> The full 422-player table is available here, sorted by mean across
> positions. The methodology page documents the data and functions behind each
> stage, and the hyperparameter page makes the modeling choices explicit.

*Show: Complete projections, then point out the links to Methodology and
Hyperparameters.*

## 8:15 — Close

> The key contribution here is not claiming to know Week 1 perfectly. It is a
> transparent process: build as-of priors, simulate the football events that
> produce points, represent workload uncertainty directly, and measure whether
> the reported ranges behave honestly in historical holdouts.
>
> The next improvements would be dated depth-chart and injury inputs,
> opponent and team-total context, and a play-level replacement for the
> targetable-pass proxy. Those would make the priors more informed while
> preserving the same auditable simulation and evaluation framework.
