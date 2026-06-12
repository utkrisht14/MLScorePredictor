from pathlib import Path

import pandas as pd

from mlscorepredictor.config import get_settings


class DataCatalog:
    """Loads normalized project datasets from a configurable data directory."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir or get_settings().data_dir)

    def read_csv(self, name: str) -> pd.DataFrame:
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")
        return pd.read_csv(path)

    def historical_matches(self) -> pd.DataFrame:
        return self.read_csv("historical_matches.csv")

    def teams(self) -> pd.DataFrame:
        return self.read_csv("teams.csv")

    def fixtures(self) -> pd.DataFrame:
        return self.read_csv("fixtures_2026.csv")

    def players(self) -> pd.DataFrame:
        return self.read_csv("players.csv")

    def recent_team_form(self) -> pd.DataFrame:
        return self.read_csv("recent_team_form.csv")

    def rag_documents(self) -> pd.DataFrame:
        return self.read_csv("rag_documents.csv")
