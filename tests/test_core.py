import numpy as np
import pandas as pd
import pytest

from fantasy_ranges.data import load_player_games, normalize_player_games
from fantasy_ranges.demo import synthetic_games
from fantasy_ranges.features import RECENCY_DECAY, _weighted_mean, build_preseason_features
from fantasy_ranges.simulation import ComponentSimulator


def test_normalizer_keeps_qb_attempts_in_team_context():
    raw = pd.DataFrame([
        {"player_id": "qb", "player_display_name": "QB", "position": "QB", "team": "AAA", "season": 2025, "week": 1, "attempts": 31},
        {"player_id": "wr", "player_display_name": "WR", "position": "WR", "team": "AAA", "season": 2025, "week": 1, "targets": 8},
    ])
    games = normalize_player_games(raw)
    assert len(games) == 1
    assert games.iloc[0]["team_pass_attempts"] == 31


def test_load_player_games_preserves_existing_team_volume(tmp_path):
    raw = pd.DataFrame([
        {"player_id": "qb", "player_display_name": "QB", "position": "QB", "team": "AAA", "season": 2025, "week": 1, "attempts": 31},
        {"player_id": "wr", "player_display_name": "WR", "position": "WR", "team": "AAA", "season": 2025, "week": 1, "targets": 8},
    ])
    path = tmp_path / "player_games.csv"
    normalize_player_games(raw).to_csv(path, index=False)

    loaded = load_player_games(path)

    assert loaded.iloc[0]["team_pass_attempts"] == 31


def test_recency_decay_puts_about_half_weight_on_latest_of_three_full_seasons():
    values = pd.Series([0.0, 0.0, 1.0])
    seasons = pd.Series([2023, 2024, 2025])

    latest_weight = _weighted_mean(values, seasons, 2026)

    assert RECENCY_DECAY == 0.60
    assert latest_weight == pytest.approx(0.5102, abs=0.001)


def test_positional_pooling_tapers_away_for_established_player():
    assert ComponentSimulator._shrink(0.50, 0.10, games=48, strength=14) == pytest.approx(0.50)


def test_preseason_features_ignore_target_season_outcomes():
    games = synthetic_games()
    candidate = games.loc[(games.season == 2025) & (games.week == 1)].iloc[[0]][["player_id", "player_name", "position", "team"]]
    baseline = build_preseason_features(games, candidate, 2025)
    changed = games.copy()
    changed.loc[(changed.season == 2025) & (changed.player_id == candidate.iloc[0].player_id), "targets"] = 99
    actual_changed = build_preseason_features(changed, candidate, 2025)
    columns = ["prior_target_share", "prior_rush_share", "games_sample", "role_uncertainty"]
    assert np.allclose(baseline[columns], actual_changed[columns])


def test_destination_team_volume_averages_one_row_per_game():
    # Week 1 has two fantasy-position rows and Week 2 has one; the 20 and 40
    # pass attempts must receive equal game weights rather than row weights.
    raw = pd.DataFrame([
        {"player_id": "qb1", "player_display_name": "QB", "position": "QB", "team": "AAA", "season": 2025, "week": 1, "attempts": 20},
        {"player_id": "rb", "player_display_name": "RB", "position": "RB", "team": "AAA", "season": 2025, "week": 1, "targets": 2, "carries": 8},
        {"player_id": "wr", "player_display_name": "WR", "position": "WR", "team": "AAA", "season": 2025, "week": 1, "targets": 5},
        {"player_id": "qb2", "player_display_name": "QB", "position": "QB", "team": "AAA", "season": 2025, "week": 2, "attempts": 40},
        {"player_id": "rb", "player_display_name": "RB", "position": "RB", "team": "AAA", "season": 2025, "week": 2, "targets": 1, "carries": 9},
    ])
    games = normalize_player_games(raw)
    candidate = pd.DataFrame([{"player_id": "rb", "player_name": "RB", "position": "RB", "team": "AAA"}])
    feature = build_preseason_features(games, candidate, 2026).iloc[0]
    assert feature["expected_team_pass_attempts"] == 30


def test_role_uncertainty_propagates_to_wider_outcomes():
    games = synthetic_games()
    stable = pd.DataFrame([{"player_id": "wr_alpha", "player_name": "Veteran WR", "position": "WR", "team": "BBB", "role_uncertainty": .10}])
    uncertain = pd.DataFrame([{"player_id": "new_wr", "player_name": "Rookie WR", "position": "WR", "team": "BBB", "rookie": 1, "role_uncertainty": .95}])
    features = build_preseason_features(games, pd.concat([stable, uncertain]), 2026)
    features["expected_fantasy_points"] = 16.0
    model = ComponentSimulator(simulations=8_000, seed=123).fit(games)
    stable_draws = model.simulate(features.iloc[0]).samples
    uncertain_draws = model.simulate(features.iloc[1]).samples
    stable_width = np.quantile(stable_draws, .9) - np.quantile(stable_draws, .1)
    uncertain_width = np.quantile(uncertain_draws, .9) - np.quantile(uncertain_draws, .1)
    assert uncertain_width > stable_width
