# Data Sources and Collection Plan

This project needs two categories of data.

## Required External Data

| Dataset | Why it matters | Practical source options |
| --- | --- | --- |
| Historical international results | Train team strength, form, attack, defense, and expected-goals model | Kaggle international results datasets, football-data APIs, public GitHub CSVs |
| World Cup 2026 group fixtures | App match list and context | FIFA schedule, reputable fixture feeds, curated CSV |
| FIFA rankings and Elo ratings | Core team-strength features | FIFA rankings, World Football Elo snapshots |
| Player attributes | Squad quality and scenario modelling | Legal CSV exports, Kaggle datasets, manually curated selected-player table |
| Player performance | Current form and role strength | FBref/StatsBomb/club data where licensing allows |
| Squads and expected lineups | Starting XI and bench impact | FIFA squad announcements, reputable preview feeds, manual curation |
| Injuries and suspensions | Scenario analysis and RAG explanation | News feeds, official team updates, injury reports |
| Tactical previews and news | RAG context | Articles, previews, official team pages, analyst reports |
| Weather and venue context | Host/venue adjustment | Weather API, stadium altitude/city data |
| Betting odds, optional | Market benchmark and calibration | Odds APIs where allowed |

## What We Build Internally

- Data validation and normalized CSV schema.
- Feature engineering for form, rankings, squad quality, and match context.
- ML expected-goals model.
- Poisson scoreline probability model.
- FastAPI prediction service.
- Streamlit user interface.
- RAG ingestion, retrieval, and natural-language explanation.
- Scenario engine for missing key players and lineup changes.

## Legal and Practical Note on EA FC Data

EA FC player ratings can be useful for a student/demo project if you use a legally available
dataset or manually curated selected-player data. Avoid scraping protected or unauthorized
sources. The current app uses curated EAFC-style selected-player attributes for all 48 teams.
If you obtain a legal official or third-party CSV, replace `players.csv` using the same schema:
`player_name, team, position, overall, pace, shooting, passing, dribbling, defending, physical,
gk, minutes_projection, is_key_player`.
