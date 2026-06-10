from __future__ import annotations

from mlscorepredictor.config import get_settings
from mlscorepredictor.data.loaders import DataCatalog
from mlscorepredictor.features.team_features import TeamFeatureBuilder
from mlscorepredictor.modeling.expected_goals import ExpectedGoalsModel
from mlscorepredictor.modeling.poisson import scoreline_distribution
from mlscorepredictor.schemas import PredictionRequest, PredictionResponse


class FootballPredictor:
    """Coordinates feature building, expected-goals prediction, and score probabilities."""

    def __init__(self, catalog: DataCatalog | None = None) -> None:
        self.catalog = catalog or DataCatalog()
        self.settings = get_settings()
        self.historical_matches = self.catalog.historical_matches()
        self.teams = self.catalog.teams()
        self.players = self.catalog.players()
        self.fixtures = self.catalog.fixtures()

        self.feature_builder = TeamFeatureBuilder(
            historical_matches=self.historical_matches,
            teams=self.teams,
            players=self.players,
        )
        self.expected_goals_model = ExpectedGoalsModel(self.feature_builder, self.players)
        self.expected_goals_model.train(self.historical_matches)

    def predict(self, request: PredictionRequest, explanation: str = "") -> PredictionResponse:
        expected_a, expected_b = self.expected_goals_model.predict(
            request.team_a,
            request.team_b,
            request.neutral_venue,
            request.player_scenarios,
        )
        scorelines, outcomes = scoreline_distribution(
            expected_a,
            expected_b,
            max_goals=self.settings.max_score_goals,
        )

        return PredictionResponse(
            team_a=request.team_a,
            team_b=request.team_b,
            expected_goals_a=expected_a,
            expected_goals_b=expected_b,
            top_scorelines=scorelines[:5],
            outcome_probabilities=outcomes,
            explanation=explanation,
        )
