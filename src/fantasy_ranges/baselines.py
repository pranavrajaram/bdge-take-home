"""Transparent range baselines used to keep the simulator honest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulation import QUANTILES


def expected_points(features: pd.DataFrame) -> pd.Series:
    """A deterministic PPR center from prior component means, with no TD overfit."""
    team_targets = features["expected_team_pass_attempts"] * 0.64
    targets = team_targets * features["prior_target_share"]
    carries = features["expected_team_rush_attempts"] * features["prior_rush_share"]
    receptions = targets * features["prior_catch_rate"]
    points = receptions + 0.1 * (receptions * features["prior_yards_per_reception"] + carries * features["prior_rush_yards_per_carry"])
    points += 6 * (targets * features["prior_receiving_td_rate"] + carries * features["prior_rushing_td_rate"])
    return points.clip(lower=0)


class PositionNormalBaseline:
    """Player center + position residual width; a useful fixed-shape benchmark."""
    def fit(self, training: pd.DataFrame) -> "PositionNormalBaseline":
        centers = expected_points(training)
        residual = training["fantasy_points_ppr"].to_numpy() - centers.to_numpy()
        temp = training[["position"]].copy()
        temp["residual"] = residual
        self.scale = temp.groupby("position")["residual"].std().fillna(7.0).to_dict()
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        centers = expected_points(features)
        output = features[["player_id", "player_name", "position", "team"]].copy()
        output["mean"] = centers
        for q in QUANTILES:
            z = {0.1: -1.2816, 0.25: -0.6745, 0.5: 0, 0.75: 0.6745, 0.9: 1.2816}[q]
            output[f"p{int(q * 100):02d}"] = (centers + z * features["position"].map(self.scale).fillna(7.0)).clip(lower=0)
        return output


class UsageNearestNeighbors:
    """Weighted empirical outcome distributions for similar preseason profiles."""
    columns = ["prior_target_share", "prior_rush_share", "prior_targets", "prior_carries", "role_uncertainty"]

    def __init__(self, neighbors: int = 100):
        self.neighbors = neighbors

    def fit(self, training: pd.DataFrame) -> "UsageNearestNeighbors":
        self.training = training.dropna(subset=["fantasy_points_ppr"]).copy()
        numeric = self.training[self.columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.training.loc[:, self.columns] = numeric
        self.mean = numeric.mean()
        self.std = numeric.std().replace(0, 1).fillna(1)
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in features.iterrows():
            pool = self.training.loc[self.training["position"] == row["position"]].copy()
            if pool.empty:
                pool = self.training.copy()
            pool_values = pool[self.columns].apply(pd.to_numeric, errors="coerce").fillna(self.mean)
            row_values = pd.to_numeric(row[self.columns], errors="coerce").fillna(self.mean)
            delta = pool_values.sub(row_values, axis="columns").div(self.std, axis="columns")
            distance = np.sqrt((delta.to_numpy(dtype=float) ** 2).sum(axis=1))
            chosen = pool.assign(_distance=distance).nsmallest(min(self.neighbors, len(pool)), "_distance")
            weights = 1 / (chosen["_distance"].to_numpy() + 0.08)
            # Weighted resampling is robust to extreme historical fantasy games.
            rng = np.random.default_rng(abs(hash(str(row["player_id"]))) % (2**32))
            samples = rng.choice(chosen["fantasy_points_ppr"], size=5000, p=weights / weights.sum())
            values = {f"p{int(q * 100):02d}": float(np.quantile(samples, q)) for q in QUANTILES}
            values["mean"] = float(np.mean(samples))
            rows.append({"player_id": row["player_id"], "player_name": row["player_name"], "position": row["position"], "team": row["team"]} | values)
        return pd.DataFrame(rows)
