"""Command line interface for fetch, backtest and an offline demonstration."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import calibration_plot, run_week1_backtest
from .data import (
    fetch_player_stats, fetch_roster, fetch_schedule, load_player_games,
    week1_candidates_from_roster, week1_matchups_from_schedule,
)
from .demo import run_demo
from .projections import project_week1


def main() -> None:
    parser = argparse.ArgumentParser(description="Preseason fantasy range projections")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download public nflverse weekly player stats")
    fetch.add_argument("--seasons", type=int, nargs="+", required=True)
    fetch.add_argument("--data-dir", default="data/raw")
    backtest = sub.add_parser("backtest", help="walk-forward historical Week 1 backtest")
    backtest.add_argument("--input", required=True)
    backtest.add_argument("--seasons", type=int, nargs="+", required=True)
    backtest.add_argument("--out-dir", default="outputs/backtest")
    backtest.add_argument("--simulations", type=int, default=8_000)
    demo = sub.add_parser("demo", help="run reproducible offline synthetic demonstration")
    demo.add_argument("--out-dir", default="outputs/demo")
    project = sub.add_parser("project-week1", help="generate real Week 1 projections from a roster snapshot")
    project.add_argument("--input", required=True, help="normalized historical player-game CSV")
    project.add_argument("--season", type=int, default=2026)
    project.add_argument("--roster", help="optional local nflverse roster CSV; fetched/cached when omitted")
    project.add_argument("--schedule", help="optional local nflverse games CSV; fetched/cached when omitted")
    project.add_argument("--data-dir", default="data/inputs", help="cache location when roster is fetched")
    project.add_argument("--out-dir", default="outputs/week1_2026")
    project.add_argument("--simulations", type=int, default=20_000)
    project.add_argument("--no-calibrate", action="store_true", help="skip historical walk-forward conformal calibration")
    args = parser.parse_args()
    if args.command == "fetch":
        frame = fetch_player_stats(args.seasons, args.data_dir)
        print(f"Saved {len(frame):,} normalized RB/WR/TE player games to {Path(args.data_dir) / 'player_games.csv'}")
    elif args.command == "backtest":
        games = load_player_games(args.input)
        predictions, metrics = run_week1_backtest(games, args.seasons, args.simulations)
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out / "predictions.csv", index=False)
        metrics.to_csv(out / "metrics.csv", index=False)
        calibration_plot(predictions, out / "calibration.png")
        print(metrics.to_string(index=False))
    elif args.command == "project-week1":
        games = load_player_games(args.input)
        roster = pd.read_csv(args.roster, low_memory=False) if args.roster else fetch_roster(args.season, args.data_dir)
        schedule = pd.read_csv(args.schedule, low_memory=False) if args.schedule else fetch_schedule(args.data_dir)
        matchups = week1_matchups_from_schedule(schedule, args.season)
        candidates = week1_candidates_from_roster(roster, args.season, matchups=matchups)
        projection = project_week1(games, candidates, args.season, args.simulations, calibrate=not args.no_calibrate)
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        projection.to_csv(out / "projections.csv", index=False)
        candidates.to_csv(out / "candidate_roster.csv", index=False)
        print(f"Saved {len(projection):,} active RB/WR/TE Week {1} projections to {out / 'projections.csv'}")
        print(projection[["player_name", "position", "team", "p10", "p50", "p90", "uncertainty"]].head(20).to_string(index=False))
    else:
        metrics, projection = run_demo(args.out_dir)
        print(metrics.to_string(index=False))
        print("\nExample equal-center, different-uncertainty projections:\n", projection[["player_name", "p10", "p50", "p90", "role_uncertainty"]].to_string(index=False))


if __name__ == "__main__":
    main()
