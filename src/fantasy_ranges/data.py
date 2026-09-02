"""Public-data ingestion and a defensive normalization boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

NFLVERSE_WEEKLY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)
NFLVERSE_ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"
)

# nflverse names are stable, but aliases keep this project compatible with older exports.
ALIASES = {
    "player_id": ("player_id", "gsis_id"),
    "player_name": ("player_display_name", "player_name", "display_name"),
    "position": ("position", "pos"),
    "team": ("team", "recent_team", "posteam"),
    "opponent": ("opponent_team", "opponent", "defteam"),
    "fantasy_points_ppr": ("fantasy_points_ppr", "fantasy_points"),
    "receiving_tds": ("receiving_tds", "receiving_touchdowns"),
    "rushing_tds": ("rushing_tds", "rushing_touchdowns"),
    "passing_attempts": ("attempts", "passing_attempts"),
}
NUMERIC = (
    "week", "targets", "receptions", "receiving_yards", "receiving_tds", "carries",
    "rushing_yards", "rushing_tds", "passing_attempts", "target_share",
    "fantasy_points_ppr", "routes", "route_participation", "snap_share",
)
REQUIRED = ("player_id", "player_name", "position", "season", "week", "team")
NORMALIZED_COLUMNS = {
    "team_targets", "team_rush_attempts", "team_pass_attempts",
    "target_share_game", "rush_share_game",
}


def _rename_known_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    rename: dict[str, str] = {}
    for canonical, candidates in ALIASES.items():
        present = next((name for name in candidates if name in frame.columns), None)
        if present and present != canonical:
            rename[present] = canonical
    return frame.rename(columns=rename)


def normalize_player_games(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a model table with consistent names and derived team opportunities.

    This function does not create any prior features. Every non-label column in
    a downstream prediction table must be computed by `build_preseason_features`.
    """
    df = _rename_known_columns(frame)
    missing = set(REQUIRED).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required player-game columns: {sorted(missing)}")
    if "season_type" in df:
        df = df.loc[df["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
    for column in NUMERIC:
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df["position"] = df["position"].astype(str).str.upper()
    # nflverse fantasy_points_ppr is preferred, but calculating is transparent if absent.
    if "fantasy_points_ppr" not in frame.columns and "fantasy_points" not in frame.columns:
        df["fantasy_points_ppr"] = (
            df["receptions"] + 0.1 * (df["receiving_yards"] + df["rushing_yards"])
            + 6 * (df["receiving_tds"] + df["rushing_tds"])
        )
    group = ["season", "week", "team"]
    # Aggregate before retaining fantasy positions: QB pass attempts and QB rushes
    # belong in team opportunity, even though QB is not a modeled output position.
    df["team_targets"] = df.groupby(group)["targets"].transform("sum")
    df["team_rush_attempts"] = df.groupby(group)["carries"].transform("sum")
    df["team_pass_attempts"] = df.groupby(group)["passing_attempts"].transform("sum")
    # Fall back to targetable passes when passing attempts are not present in an export.
    df["team_pass_attempts"] = df["team_pass_attempts"].where(
        df["team_pass_attempts"] > 0, df["team_targets"] / 0.64
    )
    df = df.loc[df["position"].isin(["RB", "WR", "TE"])].copy()
    df["target_share_game"] = df["targets"] / df["team_targets"].clip(lower=1)
    df["rush_share_game"] = df["carries"] / df["team_rush_attempts"].clip(lower=1)
    df["touchdowns"] = df["receiving_tds"] + df["rushing_tds"]
    df["game_key"] = df.get("game_id", df["season"].astype(str) + "_" + df["week"].astype(str) + "_" + df["team"])
    return df.sort_values(["season", "week", "player_id", "team"]).reset_index(drop=True)


def load_player_games(path: str | Path) -> pd.DataFrame:
    """Load raw nflverse rows or preserve an already-normalized model table.

    The fetch_player_stats output has already used QB rows to calculate team
    pass attempts before filtering output positions. Re-running the normalizer
    on that file would remove the QB evidence and overwrite that field with an
    incomplete total. Raw nflverse exports still pass through the normalizer.
    """
    frame = pd.read_csv(path, low_memory=False)
    if NORMALIZED_COLUMNS.issubset(frame.columns):
        return frame
    return normalize_player_games(frame)


def fetch_player_stats(seasons: Iterable[int], data_dir: str | Path) -> pd.DataFrame:
    """Download nflverse weekly box scores, cache individual CSVs, and combine them."""
    output = Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    manifest = []
    for season in seasons:
        path = output / f"stats_player_week_{season}.csv"
        url = NFLVERSE_WEEKLY_URL.format(season=season)
        if not path.exists():
            pd.read_csv(url).to_csv(path, index=False)
        raw = pd.read_csv(path, low_memory=False)
        frames.append(raw)
        manifest.append({"season": season, "url": url, "file": path.name, "rows": len(raw)})
    combined = normalize_player_games(pd.concat(frames, ignore_index=True))
    combined.to_csv(output / "player_games.csv", index=False)
    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)
    return combined


def fetch_roster(season: int, data_dir: str | Path) -> pd.DataFrame:
    """Download and cache the season roster release used as the Week 1 universe."""
    output = Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"roster_{season}.csv"
    if not path.exists():
        pd.read_csv(NFLVERSE_ROSTER_URL.format(season=season)).to_csv(path, index=False)
    return pd.read_csv(path, low_memory=False)


def week1_candidates_from_roster(roster: pd.DataFrame, season: int, active_only: bool = True) -> pd.DataFrame:
    """Turn nflverse's official Week 1 roster snapshot into projection candidates.

    A roster answers who can plausibly play, not how much. Role uncertainty and
    historical usage are intentionally resolved in the feature builder.
    """
    required = {"gsis_id", "full_name", "position", "team"}
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"Roster is missing required columns: {sorted(missing)}")
    candidates = roster.copy()
    if "week" in candidates:
        candidates = candidates.loc[candidates["week"].eq(1)]
    if active_only and "status" in candidates:
        candidates = candidates.loc[candidates["status"].eq("ACT")]
    candidates = candidates.loc[candidates["position"].isin(["RB", "WR", "TE"])].copy()
    candidates = candidates.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    rookie_year = pd.to_numeric(candidates.get("rookie_year"), errors="coerce")
    years_exp = pd.to_numeric(candidates.get("years_exp"), errors="coerce")
    candidates["rookie"] = ((rookie_year == season) | (years_exp == 0)).fillna(False).astype(int)
    keep = ["player_id", "player_name", "position", "team", "rookie"]
    for column in ("depth_chart_position", "years_exp", "status", "headshot_url"):
        if column in candidates:
            keep.append(column)
    return candidates[keep].dropna(subset=["player_id", "team"]).drop_duplicates(["player_id", "team"])
