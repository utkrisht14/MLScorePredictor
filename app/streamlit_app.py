from __future__ import annotations

import streamlit as st

from mlscorepredictor.data.loaders import DataCatalog
from mlscorepredictor.modeling.predictor import FootballPredictor
from mlscorepredictor.rag.service import RagService
from mlscorepredictor.schemas import PlayerScenario, PredictionRequest
from mlscorepredictor.utils import parse_bool


@st.cache_resource
def load_services() -> tuple[DataCatalog, FootballPredictor, RagService]:
    catalog = DataCatalog()
    return catalog, FootballPredictor(catalog=catalog), RagService(catalog=catalog)


catalog, predictor, rag_service = load_services()
fixtures = catalog.fixtures()
teams = sorted(catalog.teams()["team"].tolist())
players = catalog.players()

st.set_page_config(page_title="World Cup 2026 Score Predictor", layout="wide")
st.title("World Cup 2026 Score Predictor")

left, right = st.columns([0.36, 0.64])

with left:
    fixture_labels = [
        f"{row.fixture_id} | Group {row.group}: {row.team_a} vs {row.team_b}"
        for row in fixtures.itertuples()
    ]
    selected_label = st.selectbox("Fixture", fixture_labels)
    selected_fixture = fixtures.iloc[fixture_labels.index(selected_label)]

    team_a = st.selectbox("Team A", teams, index=teams.index(selected_fixture["team_a"]))
    team_b = st.selectbox("Team B", teams, index=teams.index(selected_fixture["team_b"]))
    neutral = st.checkbox("Neutral venue", value=parse_bool(selected_fixture["neutral_venue"]))

    st.subheader("Scenario")
    scenario_notes = st.text_area(
        "Scenario notes",
        placeholder="Example: France rotate midfield, Senegal start a deeper defensive block.",
    )

    scenario_players = players[players["team"].isin([team_a, team_b])]["player_name"].tolist()
    player_name = st.selectbox("Player availability change", ["None"] + scenario_players)
    player_status = st.selectbox("Status", ["out", "doubtful", "limited", "starts"])

    run = st.button("Predict", type="primary", use_container_width=True)

with right:
    if run:
        scenarios = []
        if player_name != "None":
            player_team = players.loc[players["player_name"].eq(player_name), "team"].iloc[0]
            scenarios.append(
                PlayerScenario(
                    player_name=player_name,
                    team=player_team,
                    status=player_status,
                )
            )

        request = PredictionRequest(
            team_a=team_a,
            team_b=team_b,
            neutral_venue=neutral,
            fixture_id=str(selected_fixture["fixture_id"]),
            scenario_notes=scenario_notes or None,
            player_scenarios=scenarios,
        )
        prediction = rag_service.explain(request, predictor.predict(request))

        metric_cols = st.columns(3)
        metric_cols[0].metric(f"{team_a} xG", f"{prediction.expected_goals_a:.2f}")
        metric_cols[1].metric(f"{team_b} xG", f"{prediction.expected_goals_b:.2f}")
        metric_cols[2].metric("Top score", prediction.top_scorelines[0].scoreline)

        outcome = prediction.outcome_probabilities
        st.subheader("Outcome Probabilities")
        st.progress(outcome.team_a_win, text=f"{team_a} win: {outcome.team_a_win:.1%}")
        st.progress(outcome.draw, text=f"Draw: {outcome.draw:.1%}")
        st.progress(outcome.team_b_win, text=f"{team_b} win: {outcome.team_b_win:.1%}")

        st.subheader("Top 5 Scorelines")
        st.dataframe(
            [
                {
                    "Scoreline": item.scoreline,
                    "Probability": f"{item.probability:.1%}",
                }
                for item in prediction.top_scorelines
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Explanation")
        st.write(prediction.explanation)

        with st.expander("Retrieved context"):
            for context in prediction.retrieved_context:
                st.write(context)
    else:
        st.info("Choose a fixture and scenario, then run the prediction.")
