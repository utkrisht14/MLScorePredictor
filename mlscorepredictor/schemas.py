from pydantic import BaseModel, Field


class PlayerScenario(BaseModel):
    player_name: str
    team: str
    status: str = Field(default="out", description="out, doubtful, limited, or starts")
    minutes_expected: int | None = Field(default=None, ge=0, le=120)


class PredictionRequest(BaseModel):
    team_a: str
    team_b: str
    neutral_venue: bool = True
    fixture_id: str | None = None
    scenario_notes: str | None = None
    player_scenarios: list[PlayerScenario] = Field(default_factory=list)


class ScorelineProbability(BaseModel):
    scoreline: str
    team_a_goals: int
    team_b_goals: int
    probability: float


class OutcomeProbabilities(BaseModel):
    team_a_win: float
    draw: float
    team_b_win: float


class PredictionResponse(BaseModel):
    team_a: str
    team_b: str
    expected_goals_a: float
    expected_goals_b: float
    top_scorelines: list[ScorelineProbability]
    outcome_probabilities: OutcomeProbabilities
    explanation: str
    retrieved_context: list[str] = Field(default_factory=list)
