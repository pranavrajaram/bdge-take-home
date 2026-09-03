"""Render the complete projection CSV as a navigable demo page."""
from __future__ import annotations

from html import escape
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "week1_2026" / "projections.csv"
OUTPUT = ROOT / "docs" / "projections.html"
PUBLIC_CSV = ROOT / "docs" / "projections.csv"
CALIBRATION_SOURCE = ROOT / "outputs" / "backtest" / "calibration.png"
PUBLIC_CALIBRATION = ROOT / "docs" / "assets" / "calibration.png"


def main() -> None:
    projections = pd.read_csv(INPUT).sort_values("mean", ascending=False).reset_index(drop=True)
    projections.to_csv(PUBLIC_CSV, index=False)
    PUBLIC_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CALIBRATION_SOURCE, PUBLIC_CALIBRATION)
    columns = [
        "player_name", "position", "team", "opponent", "matchup_multiplier", "mean", "p10", "p25", "p50", "p75", "p90",
        "p_10_plus", "p_15_plus", "p_20_plus", "role_uncertainty", "games_sample",
    ]
    display = projections[columns].rename(
        columns={
            "player_name": "player",
            "position": "pos",
            "opponent": "opp",
            "matchup_multiplier": "matchup",
            "p_10_plus": "P(10+)",
            "p_15_plus": "P(15+)",
            "p_20_plus": "P(20+)",
            "role_uncertainty": "role uncertainty",
            "games_sample": "prior games",
        }
    )
    formatters = {
        "mean": "{:.2f}".format, "p10": "{:.2f}".format, "p25": "{:.2f}".format,
        "p50": "{:.2f}".format, "p75": "{:.2f}".format, "p90": "{:.2f}".format,
        "matchup": "{:.3f}".format,
        "P(10+)": "{:.3f}".format, "P(15+)": "{:.3f}".format, "P(20+)": "{:.3f}".format,
        "role uncertainty": "{:.3f}".format,
    }
    table = display.to_html(index=False, classes="dataframe", border=0, formatters=formatters)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>2026 Week 1 Fantasy Projections</title>
  <link rel="stylesheet" href="demo.css">
</head>
<body>
<main>
  <header>
    <div class="kicker">2026 Week 1 · Full PPR</div>
    <h1>Complete Player Projections</h1>
    <p class="subtitle">All active RB / WR / TE candidates, sorted by simulated mean fantasy points across positions.</p>
    <p class="meta">{len(display):,} players · PPR: 1 reception, 0.1 yards, 6 TD</p>
    <nav>
      <a href="week1_projection_demo.html">Demo walkthrough</a>
      <a href="methodology.html">Methodology</a>
      <a href="hyperparameters.html">Hyperparameters</a>
      <a href="projections.csv">Download CSV</a>
    </nav>
  </header>
  {table}
  <p class="footer">Generated from <code>{escape(str(INPUT.relative_to(ROOT)))}</code>.</p>
</main>
</body>
</html>
"""
    OUTPUT.write_text(page)
    print(f"Wrote {OUTPUT}, {PUBLIC_CSV}, and {PUBLIC_CALIBRATION}")


if __name__ == "__main__":
    main()
