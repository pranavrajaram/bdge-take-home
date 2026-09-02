"""Offline deterministic fixture that demonstrates the whole pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import calibration_plot, run_week1_backtest
from .features import build_preseason_features
from .simulation import ComponentSimulator


def synthetic_games(seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    players = [
        ("rb_stable", "Stable RB", "RB", "AAA", .43, .05), ("rb_split", "Split RB", "RB", "AAA", .22, .08),
        ("wr_alpha", "Veteran WR", "WR", "BBB", .28, .02), ("wr_two", "WR Two", "WR", "BBB", .18, .02),
        ("te_one", "TE One", "TE", "CCC", .20, .02), ("wr_c", "WR C", "WR", "CCC", .23, .02),
        ("rb_d", "RB D", "RB", "DDD", .35, .04), ("wr_d", "WR D", "WR", "DDD", .25, .02),
    ]
    rows = []
    for season in range(2021, 2026):
        for week in range(1, 18):
            for pid, name, pos, team, share, rush_share in players:
                team_targets = int(rng.poisson(23))
                team_carries = int(rng.poisson(24))
                targets = rng.binomial(team_targets, np.clip(rng.normal(share, .05), .02, .6)) if pos != "RB" or share > .1 else 1
                carries = rng.binomial(team_carries, np.clip(rng.normal(rush_share, .06), 0, .65))
                rec = rng.binomial(targets, .65)
                rec_yards = rec * rng.gamma(5, 2.0)
                rush_yards = carries * rng.gamma(7, .62)
                rec_td = rng.poisson(targets * .045)
                rush_td = rng.poisson(carries * .028)
                rows.append({"player_id": pid, "player_name": name, "position": pos, "team": team,
                             "opponent": "OPP", "season": season, "week": week, "targets": targets,
                             "receptions": rec, "receiving_yards": rec_yards, "receiving_tds": rec_td,
                             "carries": carries, "rushing_yards": rush_yards, "rushing_tds": rush_td,
                             "passing_attempts": 35 if pos == "WR" and name == "Veteran WR" else 0})
    from .data import normalize_player_games
    return normalize_player_games(pd.DataFrame(rows))


def run_demo(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    games = synthetic_games()
    predictions, metrics = run_week1_backtest(games, range(2022, 2026), simulations=3_000)
    predictions.to_csv(out / "backtest_predictions.csv", index=False)
    metrics.to_csv(out / "backtest_metrics.csv", index=False)
    calibration_plot(predictions, out / "calibration.png")
    candidates = games.loc[(games["season"] == 2025) & (games["week"] == 1), ["player_id", "player_name", "position", "team"]].copy()
    rookie = pd.DataFrame([{"player_id": "rookie_demo", "player_name": "Rookie WR", "position": "WR", "team": "BBB", "rookie": 1,
                            "expected_fantasy_points": 16.0, "role_uncertainty": .92}])
    veteran = candidates.loc[candidates["player_id"] == "wr_alpha"].copy()
    veteran["expected_fantasy_points"] = 16.0
    veteran["role_uncertainty"] = .10
    features = build_preseason_features(games, pd.concat([veteran, rookie], ignore_index=True), 2026)
    projection, _ = ComponentSimulator(simulations=20_000).fit(games).predict(features)
    projection.to_csv(out / "projections.csv", index=False)
    return metrics, projection
