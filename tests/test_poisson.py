from mlscorepredictor.modeling.poisson import scoreline_distribution


def test_scoreline_distribution_returns_normalized_outcomes() -> None:
    scorelines, outcomes = scoreline_distribution(1.5, 1.1)

    assert len(scorelines) == 64
    assert scorelines[0].probability > 0
    assert 0.99 <= outcomes.team_a_win + outcomes.draw + outcomes.team_b_win <= 1.01
