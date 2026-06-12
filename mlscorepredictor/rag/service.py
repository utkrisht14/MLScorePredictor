from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlscorepredictor.config import Settings, get_settings
from mlscorepredictor.data.loaders import DataCatalog
from mlscorepredictor.schemas import PredictionRequest, PredictionResponse


@dataclass(frozen=True)
class RetrievedDocument:
    title: str
    source: str
    text: str


class RagService:
    """Retrieves football context and generates prediction explanations."""

    def __init__(self, catalog: DataCatalog | None = None, settings: Settings | None = None) -> None:
        self.catalog = catalog or DataCatalog()
        self.settings = settings or get_settings()
        self.documents = self.catalog.rag_documents()
        self.players = self.catalog.players()
        self.recent_form = self.catalog.recent_team_form()
        self._openai_client = None
        self._pinecone_index = None

    def explain(
        self,
        request: PredictionRequest,
        prediction: PredictionResponse,
    ) -> PredictionResponse:
        docs = self.retrieve(request)
        context = [f"{doc.title}: {doc.text}" for doc in docs]
        prediction.retrieved_context = context
        prediction.explanation = self._generate_explanation(request, prediction, docs)
        return prediction

    def retrieve(self, request: PredictionRequest) -> list[RetrievedDocument]:
        query = self._query_text(request)
        pinecone_docs = self._retrieve_from_pinecone(query, request)
        if pinecone_docs:
            return pinecone_docs + self._structured_context(request)
        return self._retrieve_from_local_seed(request) + self._structured_context(request)

    def _retrieve_from_local_seed(self, request: PredictionRequest) -> list[RetrievedDocument]:
        team_a = request.team_a.casefold()
        team_b = request.team_b.casefold()

        def score(row: pd.Series) -> int:
            text = " ".join(
                str(row[column]) for column in ["teams", "players", "title", "text"] if column in row
            ).casefold()
            return int(team_a in text) + int(team_b in text)

        rows = []
        for _, row in self.documents.iterrows():
            row_score = score(row)
            if row_score > 0:
                rows.append((row_score, row))

        rows = sorted(rows, key=lambda item: item[0], reverse=True)[: self.settings.rag_top_k]
        return [
            RetrievedDocument(
                title=str(row["title"]),
                source=str(row["source"]),
                text=str(row["text"]),
            )
            for _, row in rows
        ]

    def _structured_context(self, request: PredictionRequest) -> list[RetrievedDocument]:
        contexts = []
        for team in [request.team_a, request.team_b]:
            player_rows = self.players[self.players["team"].eq(team)].sort_values(
                ["is_key_player", "overall"],
                ascending=[False, False],
            )
            form_rows = self.recent_form[self.recent_form["team"].eq(team)]
            player_text = ", ".join(
                f"{row.player_name} ({row.position}, {row.overall})"
                for row in player_rows.head(5).itertuples()
            )
            if form_rows.empty:
                form_text = "Recent form profile unavailable."
            else:
                form = form_rows.iloc[0]
                form_text = (
                    f"Last 10: {form.last_10_w}W-{form.last_10_d}D-{form.last_10_l}L, "
                    f"{form.goals_for_10} scored, {form.goals_against_10} conceded, "
                    f"{form.clean_sheets_10} clean sheets."
                )
            contexts.append(
                RetrievedDocument(
                    title=f"{team} structured model context",
                    source="seed_structured_data",
                    text=f"{form_text} Selected player attributes: {player_text}.",
                )
            )
        return contexts

    def _retrieve_from_pinecone(
        self,
        query: str,
        request: PredictionRequest,
    ) -> list[RetrievedDocument]:
        if not self.settings.openai_api_key or not self.settings.pinecone_api_key:
            return []

        try:
            embedding = self._embed(query)
            index = self._get_pinecone_index()
            result = index.query(
                vector=embedding,
                top_k=self.settings.rag_top_k,
                include_metadata=True,
                filter={"teams": {"$in": [request.team_a, request.team_b]}},
            )
        except Exception:
            return []

        documents: list[RetrievedDocument] = []
        for match in getattr(result, "matches", []) or []:
            metadata = getattr(match, "metadata", {}) or {}
            documents.append(
                RetrievedDocument(
                    title=str(metadata.get("title", "Retrieved document")),
                    source=str(metadata.get("source", "pinecone")),
                    text=str(metadata.get("text", "")),
                )
            )
        return [doc for doc in documents if doc.text]

    def _generate_explanation(
        self,
        request: PredictionRequest,
        prediction: PredictionResponse,
        docs: list[RetrievedDocument],
    ) -> str:
        if self.settings.openai_api_key:
            try:
                return self._generate_openai_explanation(request, prediction, docs)
            except Exception:
                pass
        return self._template_explanation(request, prediction, docs)

    def _generate_openai_explanation(
        self,
        request: PredictionRequest,
        prediction: PredictionResponse,
        docs: list[RetrievedDocument],
    ) -> str:
        client = self._get_openai_client()
        context = "\n\n".join(f"{doc.title} ({doc.source}): {doc.text}" for doc in docs)
        top_scores = ", ".join(
            f"{item.scoreline} ({item.probability:.1%})" for item in prediction.top_scorelines
        )
        scenario = request.scenario_notes or "No extra scenario notes."
        player_scenarios = "; ".join(
            f"{item.player_name} for {item.team}: {item.status}" for item in request.player_scenarios
        ) or "No player availability changes."

        prompt = f"""
You are explaining a football score prediction. Do not invent facts.
Use only the model outputs and retrieved context below.

Match: {request.team_a} vs {request.team_b}
Expected goals: {request.team_a} {prediction.expected_goals_a:.2f}, {request.team_b} {prediction.expected_goals_b:.2f}
Top scorelines: {top_scores}
Win/draw/loss: {request.team_a} {prediction.outcome_probabilities.team_a_win:.1%}, draw {prediction.outcome_probabilities.draw:.1%}, {request.team_b} {prediction.outcome_probabilities.team_b_win:.1%}
Scenario notes: {scenario}
Player scenarios: {player_scenarios}

Retrieved context:
{context or "No retrieved context."}

Write a concise explanation with:
1. why the top scoreline is plausible
2. which team/player factors matter
3. how the scenario changes the prediction, if applicable
"""
        response = client.responses.create(
            model=self.settings.openai_model,
            input=prompt,
            temperature=0.2,
        )
        return response.output_text

    def _template_explanation(
        self,
        request: PredictionRequest,
        prediction: PredictionResponse,
        docs: list[RetrievedDocument],
    ) -> str:
        top = prediction.top_scorelines[0]
        outcome = prediction.outcome_probabilities
        context_note = (
            f" Retrieved context highlights: {docs[0].text}" if docs else " No RAG context was found."
        )
        scenario_note = (
            f" Scenario considered: {request.scenario_notes}" if request.scenario_notes else ""
        )
        return (
            f"The model makes {top.scoreline} the most likely scoreline at {top.probability:.1%}. "
            f"Expected goals are {prediction.expected_goals_a:.2f} for {request.team_a} and "
            f"{prediction.expected_goals_b:.2f} for {request.team_b}. Outcome probabilities are "
            f"{request.team_a} win {outcome.team_a_win:.1%}, draw {outcome.draw:.1%}, and "
            f"{request.team_b} win {outcome.team_b_win:.1%}.{scenario_note}{context_note}"
        )

    def _embed(self, text: str) -> list[float]:
        client = self._get_openai_client()
        response = client.embeddings.create(model=self.settings.openai_embedding_model, input=text)
        return response.data[0].embedding

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    def _get_pinecone_index(self):
        if self._pinecone_index is None:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            self._pinecone_index = pc.Index(self.settings.pinecone_index_name)
        return self._pinecone_index

    @staticmethod
    def _query_text(request: PredictionRequest) -> str:
        players = " ".join(item.player_name for item in request.player_scenarios)
        return f"{request.team_a} {request.team_b} preview injuries lineups tactics {players}"
