"""Hierarchical component Monte Carlo model for RB/WR/TE PPR scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
FULL_PLAYER_EVIDENCE_GAMES = 48


def _safe_mean(x: pd.Series, fallback: float) -> float:
    value = float(pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(x) else np.nan
    return fallback if not np.isfinite(value) else value


def _beta_binomial(rng: np.random.Generator, n: np.ndarray, mean: float, concentration: float) -> np.ndarray:
    mean = float(np.clip(mean, 0.001, 0.999))
    concentration = max(float(concentration), 0.4)
    p = rng.beta(mean * concentration, (1 - mean) * concentration, size=len(n))
    return rng.binomial(np.maximum(n, 0).astype(int), p)


def _gamma_draw(rng: np.random.Generator, mean: float, cv: float, size: int) -> np.ndarray:
    mean = max(float(mean), 0.05)
    cv = max(float(cv), 0.08)
    shape = 1 / cv**2
    return rng.gamma(shape=shape, scale=mean / shape, size=size)


@dataclass
class Distribution:
    """Simulation output for a single player."""
    samples: np.ndarray
    role_uncertainty: float
    components: dict[str, np.ndarray] | None = None

    def summary(self, thresholds: Iterable[float] = (10, 15, 20)) -> dict[str, float]:
        values = {f"p{int(q * 100):02d}": float(np.quantile(self.samples, q)) for q in QUANTILES}
        values["mean"] = float(np.mean(self.samples))
        values["role_uncertainty"] = float(self.role_uncertainty)
        values |= {f"p_{int(t)}_plus": float(np.mean(self.samples >= t)) for t in thresholds}
        return values

    def component_summary(self) -> dict[str, float]:
        """Mean simulated stat line and its raw-PPR scoring decomposition.

        This diagnostic is intentionally pre-conformal: conformal calibration
        adjusts output quantiles, not individual football stat components.
        """
        if self.components is None:
            raise ValueError("Component draws were not retained for this distribution.")
        mean = {name: float(np.mean(values)) for name, values in self.components.items()}
        mean["reception_points"] = mean["receptions"]
        mean["yardage_points"] = 0.1 * (mean["receiving_yards"] + mean["rushing_yards"])
        mean["touchdown_points"] = 6 * (mean["receiving_tds"] + mean["rushing_tds"])
        mean["raw_ppr_mean"] = mean["reception_points"] + mean["yardage_points"] + mean["touchdown_points"]
        return mean


class ComponentSimulator:
    """Partially pooled simulation of volume, usage, efficiency and TDs.

    The model's pooling happens through a position-level prior plus a player
    evidence weight. `role_uncertainty` lowers beta concentration rather than
    mechanically adding fantasy-point noise, so uncertainty propagates through
    targets/carries and can produce asymmetric final distributions.
    """

    def __init__(self, simulations: int = 20_000, seed: int = 7):
        self.simulations = simulations
        self.seed = seed
        self.position_prior: dict[str, dict[str, float]] = {}

    def fit(self, history: pd.DataFrame) -> "ComponentSimulator":
        data = history.copy()
        data["catch_rate"] = data["receptions"] / data["targets"].clip(lower=1)
        data["ypr"] = data["receiving_yards"] / data["receptions"].clip(lower=1)
        data["ypc"] = data["rushing_yards"] / data["carries"].clip(lower=1)
        data["rec_td_rate"] = data["receiving_tds"] / data["targets"].clip(lower=1)
        data["rush_td_rate"] = data["rushing_tds"] / data["carries"].clip(lower=1)
        data["targetable_rate"] = data["team_targets"] / data["team_pass_attempts"].clip(lower=1)
        for position, group in data.groupby("position"):
            self.position_prior[position] = {
                "target_share": _safe_mean(group["target_share_game"], 0.15),
                "rush_share": _safe_mean(group["rush_share_game"], 0.10),
                "catch_rate": _safe_mean(group["catch_rate"], 0.63),
                "ypr": _safe_mean(group["ypr"], 10.5),
                "ypc": _safe_mean(group["ypc"], 4.2),
                "rec_td": _safe_mean(group["rec_td_rate"], 0.045),
                "rush_td": _safe_mean(group["rush_td_rate"], 0.025),
                "targetable_rate": float(np.clip(_safe_mean(group["targetable_rate"], 0.64), 0.45, 0.82)),
            }
        return self

    @staticmethod
    def _shrink(value: float, prior: float, games: int, strength: float = 14) -> float:
        if not np.isfinite(value):
            return prior
        # Positional pooling protects sparse histories. It should not continue
        # to pull an established starter toward an average that includes backups:
        # the pseudo-game strength tapers from its listed value at zero games to
        # zero at 48 prior games (roughly three full seasons).
        taper = max(0.0, 1 - max(games, 0) / FULL_PLAYER_EVIDENCE_GAMES)
        effective_strength = strength * taper
        return (games * value + effective_strength * prior) / (games + effective_strength)

    def simulate(self, feature: pd.Series | dict, simulations: int | None = None, seed_offset: int = 0) -> Distribution:
        if not self.position_prior:
            raise RuntimeError("Call fit(history) before simulate().")
        x = dict(feature)
        position = str(x["position"]).upper()
        prior = self.position_prior.get(position)
        if prior is None:
            raise ValueError(f"No fitted prior for position {position}")
        n_sims = simulations or self.simulations
        rng = np.random.default_rng(self.seed + seed_offset)
        games = int(x.get("games_sample", 0))
        uncertainty = float(np.clip(x.get("role_uncertainty", 0.5), 0.02, 1.0))
        # Stable roles are concentrated; uncertain roles retain the same mean but much wider shares.
        role_concentration = 70 * (1 - uncertainty) + 5
        target_share = self._shrink(float(x.get("prior_target_share", np.nan)), prior["target_share"], games)
        rush_share = self._shrink(float(x.get("prior_rush_share", np.nan)), prior["rush_share"], games)
        catch_rate = self._shrink(float(x.get("prior_catch_rate", np.nan)), prior["catch_rate"], games, 20)
        ypr = self._shrink(float(x.get("prior_yards_per_reception", np.nan)), prior["ypr"], games, 20)
        ypc = self._shrink(float(x.get("prior_rush_yards_per_carry", np.nan)), prior["ypc"], games, 20)
        rec_td = self._shrink(float(x.get("prior_receiving_td_rate", np.nan)), prior["rec_td"], games, 30)
        rush_td = self._shrink(float(x.get("prior_rushing_td_rate", np.nan)), prior["rush_td"], games, 30)
        pass_mean = float(x.get("expected_team_pass_attempts", 34))
        rush_mean = float(x.get("expected_team_rush_attempts", 25))
        # Environment uncertainty is intentionally smaller than role uncertainty unless role inputs request otherwise.
        pass_attempts = rng.poisson(_gamma_draw(rng, pass_mean, 0.18, n_sims))
        rush_attempts = rng.poisson(_gamma_draw(rng, rush_mean, 0.20, n_sims))
        targetable = rng.binomial(pass_attempts, prior["targetable_rate"])
        targets = _beta_binomial(rng, targetable, target_share, role_concentration)
        carries = _beta_binomial(rng, rush_attempts, rush_share, role_concentration)
        receptions = _beta_binomial(rng, targets, catch_rate, 45 * (1 - uncertainty / 2))
        # Gamma draws retain positive, right-skewed yardage and do not create impossible negative values.
        receiving_yards = np.where(receptions > 0, receptions * _gamma_draw(rng, ypr, 0.42, n_sims), 0)
        rushing_yards = np.where(carries > 0, carries * _gamma_draw(rng, ypc, 0.38, n_sims), 0)
        rec_tds = rng.poisson(np.clip(targets * rec_td, 0, 1.2))
        rush_tds = rng.poisson(np.clip(carries * rush_td, 0, 1.0))
        points = receptions + 0.1 * (receiving_yards + rushing_yards) + 6 * (rec_tds + rush_tds)
        # An optional consensus/curated center is a center prior, not a replacement distribution.
        anchor = x.get("expected_fantasy_points", x.get("consensus_fp", np.nan))
        if pd.notna(anchor) and float(anchor) > 0:
            median = max(float(np.median(points)), 0.5)
            points = points * (float(anchor) / median)
        components = {
            "pass_attempts": pass_attempts,
            "rush_attempts": rush_attempts,
            "targetable_passes": targetable,
            "targets": targets,
            "carries": carries,
            "receptions": receptions,
            "receiving_yards": receiving_yards,
            "rushing_yards": rushing_yards,
            "receiving_tds": rec_tds,
            "rushing_tds": rush_tds,
        }
        return Distribution(samples=points.astype(float), role_uncertainty=uncertainty, components=components)

    def predict(self, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        summaries = []
        draws: dict[str, np.ndarray] = {}
        for index, (_, row) in enumerate(features.iterrows()):
            distribution = self.simulate(row, seed_offset=index)
            player_key = f"{row['player_id']}::{row.get('team', '')}"
            draws[player_key] = distribution.samples
            summaries.append(dict(row) | distribution.summary())
        return pd.DataFrame(summaries), draws
