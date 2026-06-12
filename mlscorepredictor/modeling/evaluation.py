from __future__ import annotations

import math

import pandas as pd

from mlscorepredictor.modeling.predictor import FootballPredictor
from mlscorepredictor.schemas import PredictionRequest
from mlscorepredictor.utils import parse_bool


def evaluate_historical_predictions(predictor: FootballPredictor) -> dict[str, float]:
    """Evaluates outcome probabilities on the available historical seed data."""

    rows = predictor.historical_matches
    if rows.empty:
        return {"matches": 0, "accuracy": 0.0, "log_loss": 0.0, "brier_score": 0.0}

    correct = 0
    log_loss = 0.0
    brier_score = 0.0
    for match in rows.to_dict("records"):
        prediction = predictor.predict(
            PredictionRequest(
                team_a=match["team_a"],
                team_b=match["team_b"],
                neutral_venue=parse_bool(match["neutral_venue"]),
            )
        )
        probabilities = {
            "team_a_win": prediction.outcome_probabilities.team_a_win,
            "draw": prediction.outcome_probabilities.draw,
            "team_b_win": prediction.outcome_probabilities.team_b_win,
        }
        actual = _actual_outcome(int(match["team_a_goals"]), int(match["team_b_goals"]))
        predicted = max(probabilities, key=probabilities.get)
        correct += int(predicted == actual)
        log_loss -= math.log(max(probabilities[actual], 1e-9))
        brier_score += sum((probabilities[key] - float(key == actual)) ** 2 for key in probabilities)

    total = len(rows)
    return {
        "matches": float(total),
        "accuracy": correct / total,
        "log_loss": log_loss / total,
        "brier_score": brier_score / total,
    }


def fixture_prediction_frame(predictor: FootballPredictor, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fixture in fixtures.to_dict("records"):
        prediction = predictor.predict(
            PredictionRequest(
                team_a=fixture["team_a"],
                team_b=fixture["team_b"],
                neutral_venue=parse_bool(fixture["neutral_venue"]),
                fixture_id=str(fixture["fixture_id"]),
            )
        )
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "group": fixture["group"],
                "team_a": fixture["team_a"],
                "team_b": fixture["team_b"],
                "expected_goals_a": prediction.expected_goals_a,
                "expected_goals_b": prediction.expected_goals_b,
                "team_a_win": prediction.outcome_probabilities.team_a_win,
                "draw": prediction.outcome_probabilities.draw,
                "team_b_win": prediction.outcome_probabilities.team_b_win,
                "top_scoreline": prediction.top_scorelines[0].scoreline,
            }
        )
    return pd.DataFrame(rows)


def _actual_outcome(team_a_goals: int, team_b_goals: int) -> str:
    if team_a_goals > team_b_goals:
        return "team_a_win"
    if team_a_goals < team_b_goals:
        return "team_b_win"
    return "draw"
