from __future__ import annotations

from mlscorepredictor.config import get_settings
from mlscorepredictor.data.loaders import DataCatalog
from mlscorepredictor.rag.service import RagService


def ingest_seed_documents() -> int:
    """Embeds and upserts local RAG documents into Pinecone."""

    settings = get_settings()
    if not settings.openai_api_key or not settings.pinecone_api_key:
        raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY are required for ingestion.")

    catalog = DataCatalog()
    service = RagService(catalog=catalog, settings=settings)
    index = service._get_pinecone_index()

    vectors = []
    for row in catalog.rag_documents().to_dict("records"):
        embedding = service._embed(row["text"])
        vectors.append(
            {
                "id": row["doc_id"],
                "values": embedding,
                "metadata": {
                    "title": row["title"],
                    "source": row["source"],
                    "published_date": row["published_date"],
                    "teams": str(row["teams"]).split(";"),
                    "players": str(row["players"]).split(";"),
                    "document_type": row["document_type"],
                    "match_id": row["match_id"],
                    "text": row["text"],
                },
            }
        )

    if vectors:
        index.upsert(vectors=vectors)
    return len(vectors)


if __name__ == "__main__":
    count = ingest_seed_documents()
    print(f"Upserted {count} football RAG documents.")
