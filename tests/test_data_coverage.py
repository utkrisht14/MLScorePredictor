import pandas as pd


def test_seed_data_covers_all_world_cup_group_stage_teams() -> None:
    fixtures = pd.read_csv("data/seed/fixtures_2026.csv")
    teams = pd.read_csv("data/seed/teams.csv")
    players = pd.read_csv("data/seed/players.csv")
    recent_form = pd.read_csv("data/seed/recent_team_form.csv")

    fixture_teams = set(fixtures["team_a"]) | set(fixtures["team_b"])

    assert len(fixtures) == 72
    assert len(fixture_teams) == 48
    assert fixture_teams <= set(teams["team"])
    assert fixture_teams <= set(recent_form["team"])
    assert fixture_teams <= set(players["team"])
    assert players.groupby("team").size().min() >= 3
