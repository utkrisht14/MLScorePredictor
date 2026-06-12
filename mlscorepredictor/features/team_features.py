from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TeamProfile:
    team: str
    fifa_rank: float
    elo_rating: float
    attack_strength: float
    defense_strength: float
    recent_goal_diff: float
    squad_attack: float
    squad_midfield: float
    squad_defense: float
    squad_goalkeeper: float


class TeamFeatureBuilder:
    """Builds compact team profiles from match, team, and player datasets."""

    def __init__(
        self,
        historical_matches: pd.DataFrame,
        teams: pd.DataFrame,
        players: pd.DataFrame,
        recent_form: pd.DataFrame | None = None,
    ) -> None:
        self.historical_matches = historical_matches.copy()
        self.teams = teams.copy()
        self.players = players.copy()
        self.recent_form = recent_form.copy() if recent_form is not None else pd.DataFrame()
        self._profiles = self._build_profiles()

    def profile(self, team: str) -> TeamProfile:
        if team in self._profiles:
            return self._profiles[team]

        # Unknown teams still receive a neutral profile so ingestion gaps do not break the app.
        median_rank = float(self.teams["fifa_rank"].median())
        median_elo = float(self.teams["elo_rating"].median())
        return TeamProfile(
            team=team,
            fifa_rank=median_rank,
            elo_rating=median_elo,
            attack_strength=1.0,
            defense_strength=1.0,
            recent_goal_diff=0.0,
            squad_attack=75.0,
            squad_midfield=75.0,
            squad_defense=75.0,
            squad_goalkeeper=75.0,
        )

    def match_features(self, team_a: str, team_b: str, neutral_venue: bool = True) -> dict[str, float]:
        a = self.profile(team_a)
        b = self.profile(team_b)
        return {
            "fifa_rank_diff": b.fifa_rank - a.fifa_rank,
            "elo_diff": a.elo_rating - b.elo_rating,
            "attack_diff": a.attack_strength - b.attack_strength,
            "defense_diff": a.defense_strength - b.defense_strength,
            "recent_goal_diff_delta": a.recent_goal_diff - b.recent_goal_diff,
            "squad_attack_diff": a.squad_attack - b.squad_attack,
            "squad_midfield_diff": a.squad_midfield - b.squad_midfield,
            "squad_defense_diff": a.squad_defense - b.squad_defense,
            "squad_goalkeeper_diff": a.squad_goalkeeper - b.squad_goalkeeper,
            "neutral_venue": float(neutral_venue),
        }

    def _build_profiles(self) -> dict[str, TeamProfile]:
        match_profiles = self._match_profiles()
        recent_profiles = self._recent_form_profiles()
        squad_profiles = self._squad_profiles()
        profiles: dict[str, TeamProfile] = {}

        for row in self.teams.to_dict("records"):
            team = row["team"]
            match_profile = match_profiles.get(team, {})
            recent_profile = recent_profiles.get(team, {})
            squad_profile = squad_profiles.get(team, {})
            profiles[team] = TeamProfile(
                team=team,
                fifa_rank=float(row["fifa_rank"]),
                elo_rating=float(row["elo_rating"]),
                attack_strength=float(
                    recent_profile.get("attack_strength", match_profile.get("attack_strength", 1.0))
                ),
                defense_strength=float(
                    recent_profile.get("defense_strength", match_profile.get("defense_strength", 1.0))
                ),
                recent_goal_diff=float(
                    recent_profile.get("recent_goal_diff", match_profile.get("recent_goal_diff", 0.0))
                ),
                squad_attack=float(squad_profile.get("squad_attack", 75.0)),
                squad_midfield=float(squad_profile.get("squad_midfield", 75.0)),
                squad_defense=float(squad_profile.get("squad_defense", 75.0)),
                squad_goalkeeper=float(squad_profile.get("squad_goalkeeper", 75.0)),
            )
        return profiles

    def _recent_form_profiles(self) -> dict[str, dict[str, float]]:
        if self.recent_form.empty:
            return {}

        profiles: dict[str, dict[str, float]] = {}
        for row in self.recent_form.to_dict("records"):
            goals_for = max(float(row["goals_for_10"]) / 10.0, 0.1)
            goals_against = max(float(row["goals_against_10"]) / 10.0, 0.1)
            profiles[row["team"]] = {
                "attack_strength": goals_for / 1.35,
                "defense_strength": goals_against / 1.20,
                "recent_goal_diff": (float(row["goals_for_10"]) - float(row["goals_against_10"]))
                / 10.0,
            }
        return profiles

    def _match_profiles(self) -> dict[str, dict[str, float]]:
        long_rows = []
        for match in self.historical_matches.to_dict("records"):
            long_rows.append(
                {
                    "team": match["team_a"],
                    "goals_for": match["team_a_goals"],
                    "goals_against": match["team_b_goals"],
                }
            )
            long_rows.append(
                {
                    "team": match["team_b"],
                    "goals_for": match["team_b_goals"],
                    "goals_against": match["team_a_goals"],
                }
            )

        if not long_rows:
            return {}

        long_df = pd.DataFrame(long_rows)
        league_avg_for = max(float(long_df["goals_for"].mean()), 0.25)
        league_avg_against = max(float(long_df["goals_against"].mean()), 0.25)

        profiles: dict[str, dict[str, float]] = {}
        for team, group in long_df.groupby("team"):
            recent = group.tail(10)
            profiles[team] = {
                "attack_strength": float(recent["goals_for"].mean() / league_avg_for),
                "defense_strength": float(recent["goals_against"].mean() / league_avg_against),
                "recent_goal_diff": float(
                    (recent["goals_for"] - recent["goals_against"]).mean()
                ),
            }
        return profiles

    def _squad_profiles(self) -> dict[str, dict[str, float]]:
        if self.players.empty:
            return {}

        profiles: dict[str, dict[str, float]] = {}
        for team, group in self.players.groupby("team"):
            forwards = group[group["position"].eq("FW")]
            midfielders = group[group["position"].eq("MF")]
            defenders = group[group["position"].eq("DF")]
            keepers = group[group["position"].eq("GK")]
            profiles[team] = {
                "squad_attack": self._rating_mean(forwards, ["overall", "shooting", "pace"]),
                "squad_midfield": self._rating_mean(midfielders, ["overall", "passing", "dribbling"]),
                "squad_defense": self._rating_mean(defenders, ["overall", "defending", "physical"]),
                "squad_goalkeeper": self._rating_mean(keepers, ["overall", "gk"]),
            }
        return profiles

    @staticmethod
    def _rating_mean(frame: pd.DataFrame, columns: list[str]) -> float:
        if frame.empty:
            return 75.0
        return float(frame[columns].mean(axis=1).mean())
