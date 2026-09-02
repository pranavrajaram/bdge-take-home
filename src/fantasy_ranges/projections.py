"""Produce deployable Week 1 projection tables from a roster snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import QuantileConformalizer, run_week1_backtest
from .features import build_preseason_features
from .simulation import ComponentSimulator


def project_week1(
    games: pd.DataFrame,
    candidates: pd.DataFrame,
    target_season: int,
    simulations: int = 20_000,
    calibrate: bool = True,
) -> pd.DataFrame:
    """Generate a player-facing projection table for a concrete Week 1 roster."""
    history = games.loc[games["season"] < target_season].copy()
    features = build_preseason_features(history, candidates, target_season)
    raw, _ = ComponentSimulator(simulations=simulations, seed=target_season).fit(history).predict(features)
    if calibrate:
        # At 2026 prediction time every 2022–25 Week 1 outcome is already known.
        # The correction is fitted only to raw forecasts from those completed years.
        historical_predictions, _ = run_week1_backtest(
            games, seasons=sorted(games.loc[games["season"] < target_season, "season"].unique()),
            simulations=min(3_000, max(1_000, simulations // 5)),
        )
        calibration_history = historical_predictions.loc[historical_predictions["model"].eq("component_mc")]
        raw = QuantileConformalizer().fit(calibration_history).transform(raw)
        raw["calibration"] = "walk_forward_conformal"
    else:
        raw["calibration"] = "raw_component_mc"
    raw["uncertainty"] = pd.cut(
        raw["role_uncertainty"], bins=[-0.01, 0.33, 0.66, 1.0], labels=["low", "medium", "high"]
    ).astype(str)
    raw["range_50"] = raw["p25"].map("{:.1f}".format) + "–" + raw["p75"].map("{:.1f}".format)
    raw["range_80"] = raw["p10"].map("{:.1f}".format) + "–" + raw["p90"].map("{:.1f}".format)
    columns = [
        "player_id", "player_name", "position", "team", "p10", "p25", "p50", "p75", "p90", "mean",
        "p_10_plus", "p_15_plus", "p_20_plus", "range_50", "range_80", "role_uncertainty", "uncertainty",
        "games_sample", "same_team", "team_change", "rookie", "prior_target_share", "prior_rush_share",
        "expected_team_pass_attempts", "expected_team_rush_attempts",
        "calibration",
    ]
    return raw[[column for column in columns if column in raw]].sort_values(
        ["position", "p50"], ascending=[True, False]
    ).reset_index(drop=True)
