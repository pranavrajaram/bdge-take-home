"""Create the data bundle used by the interactive scoring-calculation widget."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fantasy_ranges.data import (
    load_player_games, week1_candidates_from_roster, week1_matchups_from_schedule,
)
from fantasy_ranges.features import build_preseason_features
from fantasy_ranges.simulation import ComponentSimulator


ROOT = Path(__file__).resolve().parents[1]
PLAYER_NAMES = ("Christian McCaffrey", "Puka Nacua")


def main() -> None:
    games = load_player_games(ROOT / "data/raw/player_games.csv")
    schedule = pd.read_csv(ROOT / "data/inputs/games.csv", low_memory=False)
    roster = week1_candidates_from_roster(
        pd.read_csv(ROOT / "data/inputs/roster_2026.csv", low_memory=False),
        season=2026,
        matchups=week1_matchups_from_schedule(schedule, 2026),
    )
    features = build_preseason_features(games, roster, target_season=2026)
    production = pd.read_csv(ROOT / "outputs/week1_2026/projections.csv")
    simulator = ComponentSimulator(seed=2026).fit(games.loc[games["season"] < 2026])
    diagnostics = []

    for index, player_name in enumerate(PLAYER_NAMES):
        feature = features.loc[features["player_name"].eq(player_name)].iloc[0]
        output = production.loc[production["player_name"].eq(player_name)].iloc[0]
        # Use the same number of draws and player seed offset as the production
        # run. This keeps the widget's matchup-adjusted mean identical to the
        # Mean column rather than showing Monte Carlo variation from a separate
        # 100,000-draw diagnostic sample.
        feature_offset = int(features.index.get_loc(feature.name))
        distribution = simulator.simulate(feature, simulations=20_000, seed_offset=feature_offset)
        components = distribution.component_summary()
        diagnostics.append(
            {
                "player": player_name,
                "position": str(feature["position"]),
                "team": str(feature["team"]),
                "opponent": str(feature["opponent"]),
                "matchup_multiplier": float(feature["matchup_multiplier"]),
                "targets": components["targets"],
                "carries": components["carries"],
                "receptions": components["receptions"],
                "receiving_yards": components["receiving_yards"],
                "rushing_yards": components["rushing_yards"],
                "receiving_tds": components["receiving_tds"],
                "rushing_tds": components["rushing_tds"],
                "reception_points": components["reception_points"],
                "yardage_points": components["yardage_points"],
                "touchdown_points": components["touchdown_points"],
                "raw_ppr_mean": components["raw_ppr_mean"],
                "matchup_ppr_mean": float(output["mean"]),
                "raw_p50": distribution.summary()["p50"],
                "reported_p50": float(output["p50"]),
            }
        )

    target = ROOT / "docs/diagnostic_widget_data.js"
    target.write_text(
        "window.COMPONENT_DIAGNOSTICS = " + json.dumps(diagnostics, indent=2) + ";\n"
    )
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
