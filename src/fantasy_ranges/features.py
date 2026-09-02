"""As-of feature construction. This is the leakage-control boundary."""
from __future__ import annotations

import numpy as np
import pandas as pd

RECENCY_DECAY = 0.60


def _weighted_mean(
    values: pd.Series,
    seasons: pd.Series,
    target_season: int,
    decay: float = RECENCY_DECAY,
) -> float:
    if len(values) == 0:
        return np.nan
    weights = decay ** (target_season - seasons.to_numpy())
    return float(np.average(values.to_numpy(dtype=float), weights=weights))


def _player_prior(player_history: pd.DataFrame, target_season: int) -> dict[str, float | str]:
    latest = player_history.sort_values(["season", "week"]).iloc[-1]
    n = len(player_history)
    recent = player_history.sort_values(["season", "week"]).tail(min(6, n))
    metrics = {
        "prior_target_share": "target_share_game",
        "prior_rush_share": "rush_share_game",
        "prior_targets": "targets",
        "prior_carries": "carries",
        "prior_catch_rate": "catch_rate",
        "prior_yards_per_reception": "yards_per_reception",
        "prior_rush_yards_per_carry": "rush_yards_per_carry",
        "prior_receiving_td_rate": "receiving_td_rate",
        "prior_rushing_td_rate": "rushing_td_rate",
    }
    out: dict[str, float | str] = {
        "prior_team": str(latest["team"]), "games_sample": n,
        "prior_season": int(latest["season"]), "late_season_share": float(recent["target_share_game"].mean()),
        "usage_volatility": float(player_history["target_share_game"].std(ddof=0) if n > 1 else 0.25),
    }
    for name, column in metrics.items():
        out[name] = _weighted_mean(player_history[column], player_history["season"], target_season)
    team_history = player_history.loc[player_history["season"] == latest["season"]]
    out["expected_team_pass_attempts"] = float(team_history["team_pass_attempts"].mean())
    out["expected_team_rush_attempts"] = float(team_history["team_rush_attempts"].mean())
    return out


def build_preseason_features(history: pd.DataFrame, candidates: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Build player priors using only rows with `season < target_season`.

    `candidates` supplies the known Week-1 player universe and any curated
    offseason fields (team, rookie, team_change, role overrides). In a historical
    Week-1 backtest its rows are labels only; their in-game stat columns are never
    read here.
    """
    historical = history.loc[history["season"] < target_season].copy()
    if historical.empty:
        raise ValueError("Need at least one completed season before target_season")
    for col, denominator in (("catch_rate", "targets"), ("yards_per_reception", "receptions"),
                             ("rush_yards_per_carry", "carries"), ("receiving_td_rate", "targets"),
                             ("rushing_td_rate", "carries")):
        numerator = {"catch_rate": "receptions", "yards_per_reception": "receiving_yards",
                     "rush_yards_per_carry": "rushing_yards", "receiving_td_rate": "receiving_tds",
                     "rushing_td_rate": "rushing_tds"}[col]
        historical[col] = historical[numerator] / historical[denominator].clip(lower=1)
    pos_defaults = historical.groupby("position").agg(
        default_target_share=("target_share_game", "mean"), default_rush_share=("rush_share_game", "mean"),
        default_targets=("targets", "mean"), default_carries=("carries", "mean"),
        default_catch_rate=("catch_rate", "mean"), default_ypr=("yards_per_reception", "mean"),
        default_ypc=("rush_yards_per_carry", "mean"), default_rec_td=("receiving_td_rate", "mean"),
        default_rush_td=("rushing_td_rate", "mean"), default_pass=("team_pass_attempts", "mean"),
        default_rush=("team_rush_attempts", "mean"),
    )
    # When a player changes teams (or is a rookie), team volume should follow
    # the destination team, not the player's old club or a league-wide average.
    # Each team-volume field is repeated on every player row. Deduplicate to
    # one team-game before averaging so a week with more rostered RB/WR/TEs
    # cannot receive extra weight.
    latest_team_games = (
        historical.loc[historical["season"] == historical["season"].max()]
        .drop_duplicates(["season", "week", "team"])
    )
    team_defaults = latest_team_games.groupby("team").agg(
        team_pass=("team_pass_attempts", "mean"), team_rush=("team_rush_attempts", "mean")
    )
    fallback_team_volume = team_defaults.mean(numeric_only=True)
    rows = []
    for _, candidate in candidates.drop_duplicates(["player_id", "team"]).iterrows():
        position = str(candidate["position"]).upper()
        defaults = pos_defaults.loc[position] if position in pos_defaults.index else pos_defaults.mean(numeric_only=True)
        ph = historical.loc[historical["player_id"] == candidate["player_id"]]
        prior = _player_prior(ph, target_season) if len(ph) else {}
        row = candidate.to_dict() | prior
        row["position"] = position
        # Player evidence is shrunk later in the simulator; these defaults give rookies a principled prior.
        mapping = {
            "prior_target_share": "default_target_share", "prior_rush_share": "default_rush_share",
            "prior_targets": "default_targets", "prior_carries": "default_carries",
            "prior_catch_rate": "default_catch_rate", "prior_yards_per_reception": "default_ypr",
            "prior_rush_yards_per_carry": "default_ypc", "prior_receiving_td_rate": "default_rec_td",
            "prior_rushing_td_rate": "default_rush_td", "expected_team_pass_attempts": "default_pass",
            "expected_team_rush_attempts": "default_rush",
        }
        for field, default in mapping.items():
            row[field] = float(row.get(field, np.nan)) if pd.notna(row.get(field, np.nan)) else float(defaults[default])
        team_volume = team_defaults.loc[row["team"]] if row["team"] in team_defaults.index else None
        # Explicit manual inputs take precedence. Otherwise, use destination
        # team volume from the last completed season for all player types.
        for field, team_field in (("expected_team_pass_attempts", "team_pass"),
                                  ("expected_team_rush_attempts", "team_rush")):
            provided = candidate.get(field, np.nan)
            if pd.notna(provided):
                row[field] = float(provided)
            elif team_volume is not None:
                row[field] = float(team_volume[team_field])
            else:
                row[field] = float(fallback_team_volume[team_field])
        row["games_sample"] = int(row.get("games_sample", 0))
        is_rookie = bool(row.get("rookie", False)) or row["games_sample"] == 0
        same_team = int(str(row.get("team", "")) == str(row.get("prior_team", ""))) if row["games_sample"] else 0
        team_change = bool(row.get("team_change", False)) or (row["games_sample"] > 0 and not same_team)
        row["rookie"] = int(is_rookie)
        row["same_team"] = same_team
        row["team_change"] = int(team_change)
        # 0 is stable; 1 is highly uncertain. Curated role_uncertainty can only widen this prior.
        sample_penalty = max(0.0, 1 - min(row["games_sample"], 32) / 32)
        instability = min(float(row.get("usage_volatility", 0.20)) / 0.25, 1.0)
        automatic = 0.08 + 0.32 * sample_penalty + 0.22 * int(team_change) + 0.22 * int(is_rookie) + 0.12 * instability
        curated_uncertainty = pd.to_numeric(pd.Series([row.get("role_uncertainty", 0)]), errors="coerce").iloc[0]
        curated_uncertainty = 0.0 if pd.isna(curated_uncertainty) else float(curated_uncertainty)
        row["role_uncertainty"] = min(1.0, max(curated_uncertainty, automatic))
        rows.append(row)
    return pd.DataFrame(rows)
