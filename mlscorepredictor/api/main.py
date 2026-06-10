from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from mlscorepredictor.data.loaders import DataCatalog
from mlscorepredictor.modeling.predictor import FootballPredictor
from mlscorepredictor.rag.service import RagService
from mlscorepredictor.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="ML Score Predictor",
    description="Hybrid ML + Poisson + RAG football scoreline prediction API.",
    version="0.1.0",
)


@lru_cache
def get_catalog() -> DataCatalog:
    return DataCatalog()


@lru_cache
def get_predictor() -> FootballPredictor:
    return FootballPredictor(catalog=get_catalog())


@lru_cache
def get_rag_service() -> RagService:
    return RagService(catalog=get_catalog())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/teams")
def teams() -> list[dict[str, object]]:
    return get_catalog().teams().sort_values("team").to_dict("records")


@app.get("/fixtures")
def fixtures() -> list[dict[str, object]]:
    return get_catalog().fixtures().to_dict("records")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    known_teams = set(get_catalog().teams()["team"].tolist())
    if request.team_a not in known_teams:
        raise HTTPException(status_code=400, detail=f"Unknown team_a: {request.team_a}")
    if request.team_b not in known_teams:
        raise HTTPException(status_code=400, detail=f"Unknown team_b: {request.team_b}")
    if request.team_a == request.team_b:
        raise HTTPException(status_code=400, detail="Teams must be different.")

    raw_prediction = get_predictor().predict(request)
    return get_rag_service().explain(request, raw_prediction)
