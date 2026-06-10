from __future__ import annotations

import numpy as np
import pandas as pd

from mlscorepredictor.features.team_features import TeamFeatureBuilder
from mlscorepredictor.schemas import PlayerScenario
from mlscorepredictor.utils import parse_bool

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    RandomForestRegressor = None
    Pipeline = None
    StandardScaler = None


FEATURE_COLUMNS = [
    "fifa_rank_diff",
    "elo_diff",
    "attack_diff",
    "defense_diff",
    "recent_goal_diff_delta",
    "squad_attack_diff",
    "squad_midfield_diff",
    "squad_defense_diff",
    "squad_goalkeeper_diff",
    "neutral_venue",
]


class ExpectedGoalsModel:
    """Hybrid xG model: ML when training data exists, formula fallback otherwise."""

    def __init__(self, feature_builder: TeamFeatureBuilder, players: pd.DataFrame) -> None:
        self.feature_builder = feature_builder
        self.players = players.copy()
        self.model_a = None
        self.model_b = None
        self.training_rows = 0

    def train(self, historical_matches: pd.DataFrame) -> None:
        if RandomForestRegressor is None or Pipeline is None or StandardScaler is None:
            return

        rows = []
        targets_a = []
        targets_b = []
        for match in historical_matches.to_dict("records"):
            features = self.feature_builder.match_features(
                match["team_a"],
                match["team_b"],
                parse_bool(match["neutral_venue"]),
            )
            rows.append(features)
            targets_a.append(float(match["team_a_goals"]))
            targets_b.append(float(match["team_b_goals"]))

        if len(rows) < 20:
            self.training_rows = len(rows)
            return

        x = pd.DataFrame(rows)[FEATURE_COLUMNS]
        self.model_a = self._pipeline()
        self.model_b = self._pipeline()
        self.model_a.fit(x, np.array(targets_a))
        self.model_b.fit(x, np.array(targets_b))
        self.training_rows = len(rows)

    def predict(
        self,
        team_a: str,
        team_b: str,
        neutral_venue: bool = True,
        player_scenarios: list[PlayerScenario] | None = None,
    ) -> tuple[float, float]:
        features = self.feature_builder.match_features(team_a, team_b, neutral_venue)
        frame = pd.DataFrame([features])[FEATURE_COLUMNS]

        if self.model_a and self.model_b:
            goals_a = float(self.model_a.predict(frame)[0])
            goals_b = float(self.model_b.predict(frame)[0])
        else:
            goals_a, goals_b = self._fallback_expected_goals(features)

        goals_a, goals_b = self._apply_player_scenarios(
            team_a,
            team_b,
            goals_a,
            goals_b,
            player_scenarios or [],
        )
        return self._clip(goals_a), self._clip(goals_b)

    @staticmethod
    def _pipeline():
        if RandomForestRegressor is None or Pipeline is None or StandardScaler is None:
            raise RuntimeError("scikit-learn is required for ML training.")
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        )

    @staticmethod
    def _fallback_expected_goals(features: dict[str, float]) -> tuple[float, float]:
        base = 1.28
        strength = (
            0.0020 * features["elo_diff"]
            + 0.018 * features["fifa_rank_diff"]
            + 0.22 * features["attack_diff"]
            - 0.12 * features["defense_diff"]
            + 0.10 * features["recent_goal_diff_delta"]
            + 0.010 * features["squad_attack_diff"]
            + 0.006 * features["squad_midfield_diff"]
            - 0.006 * features["squad_defense_diff"]
            - 0.004 * features["squad_goalkeeper_diff"]
        )
        team_a_goals = base + strength
        team_b_goals = base - strength * 0.85
        return team_a_goals, team_b_goals

    def _apply_player_scenarios(
        self,
        team_a: str,
        team_b: str,
        goals_a: float,
        goals_b: float,
        scenarios: list[PlayerScenario],
    ) -> tuple[float, float]:
        for scenario in scenarios:
            player_rows = self.players[
                self.players["player_name"].str.casefold().eq(scenario.player_name.casefold())
                & self.players["team"].str.casefold().eq(scenario.team.casefold())
            ]
            if player_rows.empty:
                continue

            impact = self._player_goal_impact(player_rows.iloc[0])
            status = scenario.status.lower()
            multiplier = {"out": 1.0, "doubtful": 0.55, "limited": 0.35, "starts": -0.15}.get(
                status,
                0.0,
            )

            if scenario.team.casefold() == team_a.casefold():
                goals_a -= impact * multiplier
                goals_b += max(impact * multiplier * 0.20, 0.0)
            elif scenario.team.casefold() == team_b.casefold():
                goals_b -= impact * multiplier
                goals_a += max(impact * multiplier * 0.20, 0.0)

        return goals_a, goals_b

    @staticmethod
    def _player_goal_impact(player: pd.Series) -> float:
        position = str(player["position"])
        overall_delta = max(float(player["overall"]) - 75.0, 0.0)
        if position == "FW":
            return 0.035 * overall_delta + 0.004 * float(player["shooting"])
        if position == "MF":
            return 0.022 * overall_delta + 0.003 * float(player["passing"])
        if position == "DF":
            return 0.018 * overall_delta + 0.002 * float(player["defending"])
        if position == "GK":
            return 0.018 * max(float(player["gk"]) - 75.0, 0.0)
        return 0.05

    @staticmethod
    def _clip(value: float) -> float:
        return round(float(np.clip(value, 0.15, 4.75)), 3)
