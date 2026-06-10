from __future__ import annotations

import math

from mlscorepredictor.schemas import OutcomeProbabilities, ScorelineProbability


def poisson_probability(lam: float, goals: int) -> float:
    lam = max(lam, 0.01)
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def scoreline_distribution(
    expected_goals_a: float,
    expected_goals_b: float,
    max_goals: int = 7,
) -> tuple[list[ScorelineProbability], OutcomeProbabilities]:
    rows: list[ScorelineProbability] = []
    team_a_win = 0.0
    draw = 0.0
    team_b_win = 0.0

    for goals_a in range(max_goals + 1):
        p_a = poisson_probability(expected_goals_a, goals_a)
        for goals_b in range(max_goals + 1):
            probability = p_a * poisson_probability(expected_goals_b, goals_b)
            rows.append(
                ScorelineProbability(
                    scoreline=f"{goals_a}-{goals_b}",
                    team_a_goals=goals_a,
                    team_b_goals=goals_b,
                    probability=probability,
                )
            )
            if goals_a > goals_b:
                team_a_win += probability
            elif goals_a == goals_b:
                draw += probability
            else:
                team_b_win += probability

    total = team_a_win + draw + team_b_win
    if total > 0:
        team_a_win /= total
        draw /= total
        team_b_win /= total

    rows = sorted(rows, key=lambda item: item.probability, reverse=True)
    return rows, OutcomeProbabilities(team_a_win=team_a_win, draw=draw, team_b_win=team_b_win)
