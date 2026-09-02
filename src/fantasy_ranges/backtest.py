"""Preseason-style Week 1 backtests and distributional scoring."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import PositionNormalBaseline, UsageNearestNeighbors
from .features import build_preseason_features
from .simulation import ComponentSimulator


QUANTILE_COLUMNS = {0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"}


class QuantileConformalizer:
    """Distribution-free, additive quantile calibration from prior forecasts.

    It calibrates each nominal quantile using **only previous held-out Week 1
    outcomes**. Unlike tuning on the current test set, this remains valid in a
    walk-forward evaluation and can be refit once after each completed week in
    production.
    """
    def fit(self, previous: pd.DataFrame) -> "QuantileConformalizer":
        self.adjustments: dict[str, dict[str, float]] = {}
        for position, group in previous.groupby("position"):
            self.adjustments[position] = {
                column: float(np.quantile(group["fantasy_points_ppr"] - group[column], q))
                for q, column in QUANTILE_COLUMNS.items()
            }
        self.global_adjustments = {
            column: float(np.quantile(previous["fantasy_points_ppr"] - previous[column], q))
            for q, column in QUANTILE_COLUMNS.items()
        }
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        out = predictions.copy()
        for index, row in out.iterrows():
            adjustment = self.adjustments.get(row["position"], self.global_adjustments)
            for column in QUANTILE_COLUMNS.values():
                out.at[index, column] = max(0.0, row[column] + adjustment[column])
            ordered = np.maximum.accumulate([out.at[index, c] for c in QUANTILE_COLUMNS.values()])
            out.loc[index, list(QUANTILE_COLUMNS.values())] = ordered
        return out


def build_week1_panel(games: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """Make a labeled historical panel where each feature uses completed seasons only."""
    panels = []
    for season in sorted(set(seasons)):
        candidates = games.loc[(games["season"] == season) & (games["week"] == 1)].copy()
        if candidates.empty or not (games["season"] < season).any():
            continue
        features = build_preseason_features(games, candidates, season)
        labels = candidates[["player_id", "team", "fantasy_points_ppr"]].drop_duplicates(["player_id", "team"])
        features = features.merge(labels, on=["player_id", "team"], suffixes=("", "_label"))
        if "fantasy_points_ppr_label" in features:
            features["fantasy_points_ppr"] = features["fantasy_points_ppr_label"]
            features = features.drop(columns="fantasy_points_ppr_label")
        features["target_season"] = season
        panels.append(features)
    if not panels:
        raise ValueError("No target weeks with a completed prior season were found")
    return pd.concat(panels, ignore_index=True)


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return coverage, interval width, pinball, and median error by model."""
    metrics = []
    for model, group in predictions.groupby("model"):
        y = group["fantasy_points_ppr"].to_numpy()
        item: dict[str, float | str] = {"model": model, "n": len(group)}
        item["median_ae"] = float(np.median(np.abs(y - group["p50"].to_numpy())))
        for q, col in QUANTILE_COLUMNS.items():
            error = y - group[col].to_numpy()
            item[f"pinball_p{int(q * 100):02d}"] = float(np.mean(np.maximum(q * error, (q - 1) * error)))
        for level, low, high in ((50, "p25", "p75"), (80, "p10", "p90")):
            item[f"coverage_{level}"] = float(np.mean((y >= group[low]) & (y <= group[high])))
            item[f"width_{level}"] = float(np.mean(group[high] - group[low]))
        metrics.append(item)
    return pd.DataFrame(metrics).sort_values("model").reset_index(drop=True)


def calibration_plot(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """Plot nominal vs realized interval coverage, stratified by model."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="0.45", label="perfect calibration")
    for model, group in predictions.groupby("model"):
        observed = [
            np.mean((group["fantasy_points_ppr"] >= group["p25"]) & (group["fantasy_points_ppr"] <= group["p75"])),
            np.mean((group["fantasy_points_ppr"] >= group["p10"]) & (group["fantasy_points_ppr"] <= group["p90"])),
        ]
        ax.plot([0.5, 0.8], observed, marker="o", label=model)
    ax.set(xlim=(0.4, 0.9), ylim=(0.0, 1.0), xlabel="Nominal interval coverage", ylabel="Observed coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_week1_backtest(games: pd.DataFrame, seasons: Iterable[int], simulations: int = 8_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk forward through Week 1s; no model sees its held-out season."""
    panel = build_week1_panel(games, seasons)
    predictions = []
    component_history: list[pd.DataFrame] = []
    for target_season in sorted(panel["target_season"].unique()):
        test = panel.loc[panel["target_season"] == target_season].copy()
        train = panel.loc[panel["target_season"] < target_season].copy()
        history = games.loc[games["season"] < target_season].copy()
        simulator = ComponentSimulator(simulations=simulations, seed=int(target_season)).fit(history)
        simulated, _ = simulator.predict(test)
        simulated["model"] = "component_mc"
        simulated["target_season"] = target_season
        simulated["fantasy_points_ppr"] = test["fantasy_points_ppr"].to_numpy()
        predictions.append(simulated)
        if component_history:
            calibrated = QuantileConformalizer().fit(pd.concat(component_history, ignore_index=True)).transform(simulated)
            calibrated["model"] = "component_mc_conformal"
            predictions.append(calibrated)
        component_history.append(simulated)
        # The 1st evaluable season has no older labeled Week-1 panel for learned baselines.
        if not train.empty:
            normal = PositionNormalBaseline().fit(train).predict(test)
            neighbors = UsageNearestNeighbors().fit(train).predict(test)
            for forecast, name in ((normal, "position_normal"), (neighbors, "usage_knn")):
                forecast["model"] = name
                forecast["target_season"] = target_season
                forecast["fantasy_points_ppr"] = test["fantasy_points_ppr"].to_numpy()
                predictions.append(forecast)
    all_predictions = pd.concat(predictions, ignore_index=True)
    return all_predictions, evaluate(all_predictions)
