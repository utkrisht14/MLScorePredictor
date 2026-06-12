from __future__ import annotations

import pandas as pd
import streamlit as st

from mlscorepredictor.data.loaders import DataCatalog
from mlscorepredictor.modeling.evaluation import (
    evaluate_historical_predictions,
    fixture_prediction_frame,
)
from mlscorepredictor.modeling.predictor import FootballPredictor
from mlscorepredictor.modeling.simulation import expected_group_table, simulate_group_stage
from mlscorepredictor.rag.service import RagService
from mlscorepredictor.schemas import PlayerScenario, PredictionRequest
from mlscorepredictor.ui_helpers import player_impact_frame, round_frame
from mlscorepredictor.utils import parse_bool


@st.cache_resource
def load_services() -> tuple[DataCatalog, FootballPredictor, RagService]:
    catalog = DataCatalog()
    return catalog, FootballPredictor(catalog=catalog), RagService(catalog=catalog)


@st.cache_data(show_spinner=False)
def cached_fixture_predictions() -> pd.DataFrame:
    return fixture_prediction_frame(predictor, fixtures)


@st.cache_data(show_spinner=False)
def cached_group_simulation(iterations: int, seed: int) -> pd.DataFrame:
    return simulate_group_stage(predictor, fixtures, iterations=iterations, seed=seed)


catalog, predictor, rag_service = load_services()
fixtures = catalog.fixtures()
teams = sorted(catalog.teams()["team"].tolist())
players = catalog.players()
recent_form = catalog.recent_team_form()


st.set_page_config(page_title="World Cup 2026 Score Predictor", layout="wide")
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #064e3b;
        border-color: #064e3b;
        color: #ffffff;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #065f46;
        border-color: #065f46;
        color: #ffffff;
    }
    div.stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 0.2rem rgba(6, 95, 70, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("World Cup 2026 Score Predictor")
st.caption(
    f"{len(fixtures)} fixtures | {len(teams)} teams | "
    f"{len(players)} selected players | {len(recent_form)} recent-form profiles"
)

match_tab, simulator_tab, fixture_tab, evaluation_tab = st.tabs(
    ["Match Prediction", "Group Simulator", "Fixture Predictions", "Backtesting"]
)


with match_tab:
    left, right = st.columns([0.36, 0.64])

    with left:
        selected_group = st.selectbox(
            "Group",
            ["All"] + sorted(fixtures["group"].unique().tolist()),
            key="match_group",
        )
        visible_fixtures = (
            fixtures if selected_group == "All" else fixtures[fixtures["group"].eq(selected_group)]
        )

        fixture_labels = [
            f"{row.fixture_id} | Group {row.group}: {row.team_a} vs {row.team_b}"
            for row in visible_fixtures.itertuples()
        ]
        selected_label = st.selectbox("Fixture", fixture_labels)
        selected_fixture = visible_fixtures.iloc[fixture_labels.index(selected_label)]

        team_a = st.selectbox("Team A", teams, index=teams.index(selected_fixture["team_a"]))
        team_b = st.selectbox("Team B", teams, index=teams.index(selected_fixture["team_b"]))
        neutral = st.checkbox("Neutral venue", value=parse_bool(selected_fixture["neutral_venue"]))

        st.subheader("Scenario")
        scenario_notes = st.text_area(
            "Scenario notes",
            placeholder="Example: France rotate midfield, Senegal start a deeper defensive block.",
        )

        scenario_players = players[players["team"].isin([team_a, team_b])]["player_name"].tolist()
        st.caption(
            f"{team_a}/{team_b} selected-player coverage: "
            f"{len(players[players['team'].isin([team_a, team_b])])} players"
        )
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

            baseline_request = PredictionRequest(
                team_a=team_a,
                team_b=team_b,
                neutral_venue=neutral,
                fixture_id=str(selected_fixture["fixture_id"]),
            )
            request = PredictionRequest(
                team_a=team_a,
                team_b=team_b,
                neutral_venue=neutral,
                fixture_id=str(selected_fixture["fixture_id"]),
                scenario_notes=scenario_notes or None,
                player_scenarios=scenarios,
            )
            baseline = predictor.predict(baseline_request)
            prediction = rag_service.explain(request, predictor.predict(request))

            metric_cols = st.columns(3)
            metric_cols[0].metric(f"{team_a} xG", f"{prediction.expected_goals_a:.2f}")
            metric_cols[1].metric(f"{team_b} xG", f"{prediction.expected_goals_b:.2f}")
            metric_cols[2].metric("Top score", prediction.top_scorelines[0].scoreline)

            outcome = prediction.outcome_probabilities
            st.subheader("Outcome Probabilities")
            outcome_frame = pd.DataFrame(
                [
                    {"Outcome": f"{team_a} win", "Probability": outcome.team_a_win},
                    {"Outcome": "Draw", "Probability": outcome.draw},
                    {"Outcome": f"{team_b} win", "Probability": outcome.team_b_win},
                ]
            )
            st.bar_chart(outcome_frame, x="Outcome", y="Probability")

            st.subheader("Top 5 Scorelines")
            scoreline_frame = pd.DataFrame(
                [
                    {"Scoreline": item.scoreline, "Probability": item.probability}
                    for item in prediction.top_scorelines
                ]
            )
            st.bar_chart(scoreline_frame, x="Scoreline", y="Probability")
            st.dataframe(
                scoreline_frame.assign(Probability=lambda frame: frame["Probability"].map("{:.1%}".format)),
                hide_index=True,
                use_container_width=True,
            )

            if scenarios:
                st.subheader("What-If Change")
                comparison = pd.DataFrame(
                    [
                        {
                            "Metric": f"{team_a} xG",
                            "Baseline": baseline.expected_goals_a,
                            "Scenario": prediction.expected_goals_a,
                            "Change": prediction.expected_goals_a - baseline.expected_goals_a,
                        },
                        {
                            "Metric": f"{team_b} xG",
                            "Baseline": baseline.expected_goals_b,
                            "Scenario": prediction.expected_goals_b,
                            "Change": prediction.expected_goals_b - baseline.expected_goals_b,
                        },
                        {
                            "Metric": f"{team_a} win",
                            "Baseline": baseline.outcome_probabilities.team_a_win,
                            "Scenario": outcome.team_a_win,
                            "Change": outcome.team_a_win - baseline.outcome_probabilities.team_a_win,
                        },
                        {
                            "Metric": "Draw",
                            "Baseline": baseline.outcome_probabilities.draw,
                            "Scenario": outcome.draw,
                            "Change": outcome.draw - baseline.outcome_probabilities.draw,
                        },
                        {
                            "Metric": f"{team_b} win",
                            "Baseline": baseline.outcome_probabilities.team_b_win,
                            "Scenario": outcome.team_b_win,
                            "Change": outcome.team_b_win - baseline.outcome_probabilities.team_b_win,
                        },
                    ]
                )
                st.dataframe(comparison, hide_index=True, use_container_width=True)

            with st.expander("Feature Comparison"):
                features = predictor.feature_builder.match_features(team_a, team_b, neutral)
                feature_frame = pd.DataFrame(
                    [{"Feature": key, "Value": value} for key, value in features.items()]
                )
                st.dataframe(feature_frame, hide_index=True, use_container_width=True)

            with st.expander("Player Impact"):
                st.dataframe(
                    player_impact_frame(players[players["team"].isin([team_a, team_b])]),
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


with simulator_tab:
    sim_left, sim_right = st.columns([0.28, 0.72])
    with sim_left:
        simulator_group = st.selectbox(
            "View group",
            ["All"] + sorted(fixtures["group"].unique().tolist()),
            key="sim_group",
        )
        iterations = st.slider("Monte Carlo simulations", 250, 5000, 1000, step=250)
        seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
        simulate = st.button("Run Simulation", type="primary", use_container_width=True)

    with sim_right:
        selected_sim_fixtures = (
            fixtures if simulator_group == "All" else fixtures[fixtures["group"].eq(simulator_group)]
        )
        if simulator_group != "All":
            st.subheader(f"Expected Group {simulator_group} Table")
            expected_table = expected_group_table(predictor, selected_sim_fixtures)
            st.dataframe(round_frame(expected_table), hide_index=True, use_container_width=True)

        if simulate:
            simulation = cached_group_simulation(int(iterations), int(seed))
            if simulator_group != "All":
                simulation = simulation[simulation["group"].eq(simulator_group)]
            st.subheader("Monte Carlo Qualification Probabilities")
            st.bar_chart(simulation, x="team", y="advance")
            st.dataframe(round_frame(simulation), hide_index=True, use_container_width=True)
            st.download_button(
                "Download simulation CSV",
                simulation.to_csv(index=False),
                file_name="world_cup_2026_group_simulation.csv",
                mime="text/csv",
            )
        else:
            st.info("Run the simulation to estimate group finish and qualification probabilities.")


with fixture_tab:
    fixture_predictions = cached_fixture_predictions()
    group_filter = st.selectbox(
        "Group filter",
        ["All"] + sorted(fixture_predictions["group"].unique().tolist()),
        key="fixture_group",
    )
    if group_filter != "All":
        fixture_predictions = fixture_predictions[fixture_predictions["group"].eq(group_filter)]
    st.dataframe(round_frame(fixture_predictions), hide_index=True, use_container_width=True)
    st.download_button(
        "Download fixture predictions CSV",
        fixture_predictions.to_csv(index=False),
        file_name="world_cup_2026_fixture_predictions.csv",
        mime="text/csv",
    )


with evaluation_tab:
    metrics = evaluate_historical_predictions(predictor)
    cols = st.columns(4)
    cols[0].metric("Seed matches", f"{int(metrics['matches'])}")
    cols[1].metric("Outcome accuracy", f"{metrics['accuracy']:.1%}")
    cols[2].metric("Log loss", f"{metrics['log_loss']:.3f}")
    cols[3].metric("Brier score", f"{metrics['brier_score']:.3f}")
    st.info(
        "This is a demo backtest on the included seed history. "
        "A larger free historical dataset will make these metrics more meaningful."
    )

