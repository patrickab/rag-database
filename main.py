import json
import os

import polars as pl

from rag_database import EmbeddingModel, RagDatabase, RAGQuery


def main() -> None:
    """Main function to demonstrate RAG Database functionality."""

    ### Simple embedding demo
    test_text = "This is a test document for embedding."
    embedding_model = EmbeddingModel()
    embedding = embedding_model.single_embed(test_text)
    print("Embedding shape:", embedding.shape)
    print("Embedding vector:", embedding)

    ### RAG Database demo
    # 1. Initialize Empty RAG Database
    rag_db = RagDatabase()
    documents = os.listdir("example_docs/")

    # 2. Load all documents into memory
    texts = []
    titles = []
    for doc in documents:
        with open(f"example_docs/{doc}", "r") as f:
            text = f.read()
            texts.append(text)
            titles.append(doc)

    # 3. Add documents to RAG Database
    rag_db.add_documents(titles=titles, texts=texts)

    # 4. Process a RAG Query
    rag_query = RAGQuery(query="What is the memory wall & how does it relate to Moores law?", k_documents=5)
    rag_response = rag_db.rag_process_query(rag_query)
    print("RAG Response JSON:")
    print(json.dumps(json.loads(rag_response.to_json()), indent=2))


    # 5. Store Vector DB to disk
    rag_db.vector_db.database.write_parquet("rag_vector_db.parquet")

    # 6. Load Vector DB from disk
    loaded_db = pl.read_parquet("rag_vector_db.parquet")
    rag_db_loaded = RagDatabase(database=loaded_db)
    print("Loaded RAG Database from disk with", rag_db_loaded.vector_db.database.height, "documents.")


if __name__ == "__main__":
    main()