# Data source decision record

| Source | Use | Fields available / caveat | Decision |
| --- | --- | --- | --- |
| nflverse `stats_player` | Core historical player-game table | IDs, player/team/opponent, box-score rushing/receiving, targets, target share, fantasy points | Required |
| nflverse play-by-play / `stats_team` | Team volume and game environment | offensive attempts/plays, score context, game IDs; calculate with clear play filters | Recommended join |
| nflverse FTN charting / participation | Routes and route participation | Historical 2022+ is available after season completion; source/timing changed around 2023 | Optional, never an in-season dependency |
| nflverse rosters / players / draft picks | Age, experience, draft capital, roster status | Reliable identifiers and demographic/draft joins | Recommended prior enrichment |
| nflverse `games.csv` schedule | Known Week 1 opponent | `season`, `week`, `home_team`, `away_team`; stable before kickoff | Required when applying matchup adjustment |
| Public consensus projections | Center / benchmark | Current sources may be accessible; reproducible historical archives are uneven | Optional input, not required |

Direct weekly stats URL pattern used by the loader:

`https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv`

Data retrieval should be pinned to downloaded files plus a timestamp/URL manifest before submitting a take-home. Do not scrape a web page as the authoritative historical source. nflverse documents the releases and update schedule at https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html.
