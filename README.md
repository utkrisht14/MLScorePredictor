# ML Score Predictor

Hybrid machine learning, probabilistic scoreline, and RAG application for FIFA World Cup
2026 group-stage match predictions.

The core prediction pipeline is deliberately split into two responsibilities:

- **Machine learning predicts expected goals** for both teams.
- **Poisson score modelling converts expected goals** into scoreline probabilities,
  top 5 scorelines, and win/draw/loss probabilities.

The RAG layer then explains the prediction using match context from Pinecone and OpenAI.

## What Works Now

- FastAPI backend with `/predict`, `/teams`, `/fixtures`, and `/health`.
- Streamlit frontend for selecting fixtures and running scenario analysis.
- Group-stage simulator with Monte Carlo qualification probabilities.
- Fixture prediction table with CSV export.
- Demo backtesting metrics on the included historical seed data.
- Seed real-world-style data tables for fixtures, rankings, players, historical results,
  recent team form, selected players, and RAG documents.
- ML expected-goals model with a robust fallback if the training sample is small.
- Poisson probability engine for top scorelines and match outcome probabilities.
- Pinecone/OpenAI hooks that activate when keys are available.

## Setup

Create a `.env` file from the template:

```powershell
Copy-Item .env.example .env
```

Install dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Start the API:

```powershell
uvicorn mlscorepredictor.api.main:app --reload
```

Start the Streamlit app:

```powershell
streamlit run app/streamlit_app.py
```

## Secrets

Do not commit real API keys. Use `.env`:

```text
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=football-rag
```

## Data Strategy

The seed files in `data/seed` are intentionally small so the app runs immediately.
For a realistic final project, replace or augment them with:

- full international results history
- current FIFA ranking and Elo rating snapshots
- World Cup 2026 fixtures
- selected player attributes from legal/available datasets
- expected lineups, injuries, suspensions, and tactical previews

Use `docs/data_sources.md` as the checklist for what should come from the outside world.

## Current Seed Coverage

- 72 World Cup 2026 group-stage fixtures.
- 48 team ranking/form profiles.
- 144 selected player profiles, three per team.
- What-if comparison, player impact table, feature comparison, and probability charts.

The selected-player table uses EAFC-style attribute columns. It is intentionally marked as
curated seed data, not an official complete EA database export. If you obtain a legal EAFC CSV,
replace `data/seed/players.csv` with the same column schema.
