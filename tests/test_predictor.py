from mlscorepredictor.modeling.predictor import FootballPredictor
from mlscorepredictor.schemas import PredictionRequest


def test_predictor_returns_top_five_scorelines() -> None:
    predictor = FootballPredictor()
    prediction = predictor.predict(PredictionRequest(team_a="France", team_b="Senegal"))

    assert prediction.expected_goals_a > 0
    assert prediction.expected_goals_b > 0
    assert len(prediction.top_scorelines) == 5
