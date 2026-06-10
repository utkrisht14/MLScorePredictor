from mlscorepredictor.rag.ingest import ingest_seed_documents


if __name__ == "__main__":
    count = ingest_seed_documents()
    print(f"Upserted {count} documents into Pinecone.")
