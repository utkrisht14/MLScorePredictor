from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from mlscorepredictor.modeling.poisson import scoreline_distribution
from mlscorepredictor.modeling.predictor import FootballPredictor
from mlscorepredictor.schemas import PredictionRequest
from mlscorepredictor.utils import parse_bool


def simulate_group_stage(
    predictor: FootballPredictor,
    fixtures: pd.DataFrame,
    iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Runs a Monte Carlo simulation for all group-stage fixtures."""

    rng = np.random.default_rng(seed)
    fixture_distributions = [_fixture_distribution(predictor, row) for row in fixtures.to_dict("records")]
    teams_by_group = {
        group: sorted(set(group_rows["team_a"]) | set(group_rows["team_b"]))
        for group, group_rows in fixtures.groupby("group")
    }
    groups_by_team = {
        team: group for group, teams in teams_by_group.items() for team in teams
    }

    counters = {
        team: {
            "group": groups_by_team[team],
            "first": 0,
            "second": 0,
            "third": 0,
            "fourth": 0,
            "top_two": 0,
            "advance": 0,
            "points_sum": 0.0,
            "gd_sum": 0.0,
        }
        for team in groups_by_team
    }

    for _ in range(iterations):
        stats = _fresh_stats(groups_by_team)
        for fixture, goals_a, goals_b, _ in _sample_results(fixture_distributions, rng):
            _apply_result(stats, fixture["team_a"], fixture["team_b"], goals_a, goals_b)

        third_place_rows = []
        for group, group_teams in teams_by_group.items():
            table = rank_table({team: stats[team] for team in group_teams})
            for index, row in enumerate(table, start=1):
                key = ["first", "second", "third", "fourth"][index - 1]
                counters[row["team"]][key] += 1
                counters[row["team"]]["points_sum"] += row["points"]
                counters[row["team"]]["gd_sum"] += row["goal_difference"]
                if index <= 2:
                    counters[row["team"]]["top_two"] += 1
                    counters[row["team"]]["advance"] += 1
                elif index == 3:
                    third_place_rows.append(row | {"group": group})

        third_place_rows = sorted(
            third_place_rows,
            key=lambda row: (
                row["points"],
                row["goal_difference"],
                row["goals_for"],
                -row["goals_against"],
            ),
            reverse=True,
        )
        for row in third_place_rows[:8]:
            counters[row["team"]]["advance"] += 1

    rows = []
    for team, values in counters.items():
        rows.append(
            {
                "team": team,
                "group": values["group"],
                "1st": values["first"] / iterations,
                "2nd": values["second"] / iterations,
                "3rd": values["third"] / iterations,
                "4th": values["fourth"] / iterations,
                "top_two": values["top_two"] / iterations,
                "advance": values["advance"] / iterations,
                "avg_points": values["points_sum"] / iterations,
                "avg_goal_difference": values["gd_sum"] / iterations,
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "advance", "top_two"], ascending=[True, False, False])


def expected_group_table(predictor: FootballPredictor, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Builds a deterministic group table from expected match points and goals."""

    teams = sorted(set(fixtures["team_a"]) | set(fixtures["team_b"]))
    stats = {
        team: {"team": team, "points": 0.0, "goals_for": 0.0, "goals_against": 0.0}
        for team in teams
    }
    for fixture in fixtures.to_dict("records"):
        request = PredictionRequest(
            team_a=fixture["team_a"],
            team_b=fixture["team_b"],
            neutral_venue=parse_bool(fixture["neutral_venue"]),
            fixture_id=str(fixture["fixture_id"]),
        )
        expected_a, expected_b = predictor.expected_goals_model.predict(
            request.team_a,
            request.team_b,
            request.neutral_venue,
            request.player_scenarios,
        )
        _, outcomes = scoreline_distribution(
            expected_a,
            expected_b,
            max_goals=predictor.settings.max_score_goals,
        )
        stats[request.team_a]["points"] += 3 * outcomes.team_a_win + outcomes.draw
        stats[request.team_b]["points"] += 3 * outcomes.team_b_win + outcomes.draw
        stats[request.team_a]["goals_for"] += expected_a
        stats[request.team_a]["goals_against"] += expected_b
        stats[request.team_b]["goals_for"] += expected_b
        stats[request.team_b]["goals_against"] += expected_a

    rows = []
    for values in stats.values():
        goal_difference = values["goals_for"] - values["goals_against"]
        rows.append(
            {
                "team": values["team"],
                "expected_points": values["points"],
                "expected_goals_for": values["goals_for"],
                "expected_goals_against": values["goals_against"],
                "expected_goal_difference": goal_difference,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["expected_points", "expected_goal_difference", "expected_goals_for"],
        ascending=False,
    )


def rank_table(stats: dict[str, dict[str, float]]) -> list[dict[str, float]]:
    rows = []
    for team, values in stats.items():
        rows.append(
            {
                "team": team,
                "points": values["points"],
                "goals_for": values["goals_for"],
                "goals_against": values["goals_against"],
                "goal_difference": values["goals_for"] - values["goals_against"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["points"], row["goal_difference"], row["goals_for"], row["team"]),
        reverse=True,
    )


def _fixture_distribution(
    predictor: FootballPredictor,
    fixture: dict[str, object],
) -> dict[str, object]:
    expected_a, expected_b = predictor.expected_goals_model.predict(
        str(fixture["team_a"]),
        str(fixture["team_b"]),
        parse_bool(fixture["neutral_venue"]),
        [],
    )
    distribution, _ = scoreline_distribution(
        expected_a,
        expected_b,
        max_goals=predictor.settings.max_score_goals,
    )
    probabilities = np.array([item.probability for item in distribution], dtype=float)
    probabilities = probabilities / probabilities.sum()
    return {
        "fixture": fixture,
        "scores": [(item.team_a_goals, item.team_b_goals) for item in distribution],
        "probabilities": probabilities,
    }


def _fresh_stats(groups_by_team: dict[str, str]) -> dict[str, dict[str, int]]:
    return {
        team: {"points": 0, "goals_for": 0, "goals_against": 0}
        for team in groups_by_team
    }


def _sample_results(distributions: list[dict[str, object]], rng: np.random.Generator):
    for item in distributions:
        scores = item["scores"]
        probabilities = item["probabilities"]
        index = int(rng.choice(len(scores), p=probabilities))
        goals_a, goals_b = scores[index]
        yield item["fixture"], goals_a, goals_b, probabilities[index]


def _apply_result(
    stats: defaultdict[str, dict[str, int]] | dict[str, dict[str, int]],
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
) -> None:
    stats[team_a]["goals_for"] += goals_a
    stats[team_a]["goals_against"] += goals_b
    stats[team_b]["goals_for"] += goals_b
    stats[team_b]["goals_against"] += goals_a
    if goals_a > goals_b:
        stats[team_a]["points"] += 3
    elif goals_a < goals_b:
        stats[team_b]["points"] += 3
    else:
        stats[team_a]["points"] += 1
        stats[team_b]["points"] += 1
