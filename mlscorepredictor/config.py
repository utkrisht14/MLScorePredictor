from functools import lru_cache
import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    """Runtime settings loaded from environment variables or `.env`."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "football-rag"

    data_dir: Path = Path("data/seed")
    max_score_goals: int = 7
    rag_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    env = _read_dotenv(Path(".env"))
    return Settings(
        openai_api_key=_get("OPENAI_API_KEY", env),
        openai_model=_get("OPENAI_MODEL", env, "gpt-4.1-mini") or "gpt-4.1-mini",
        openai_embedding_model=_get(
            "OPENAI_EMBEDDING_MODEL",
            env,
            "text-embedding-3-small",
        )
        or "text-embedding-3-small",
        pinecone_api_key=_get("PINECONE_API_KEY", env),
        pinecone_index_name=_get("PINECONE_INDEX_NAME", env, "football-rag") or "football-rag",
        data_dir=Path(_get("DATA_DIR", env, "data/seed") or "data/seed"),
        max_score_goals=int(_get("MAX_SCORE_GOALS", env, "7") or "7"),
        rag_top_k=int(_get("RAG_TOP_K", env, "5") or "5"),
    )


def _get(name: str, env: dict[str, str], default: str | None = None) -> str | None:
    return os.getenv(name) or env.get(name) or default


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
