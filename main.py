import json
import os

import polars as pl

from rag_database import EmbeddingModel, RagDatabase, RAGQuery


def main() -> None:
    """Main function to demonstrate RAG Database functionality."""

    ### Simple embedding demo
    # Uses a configurable default model - expects installation of Ollama - modify rag_config.py to use eg OpenAI or Gemini
    embedding_model = EmbeddingModel()
    test_document = "This is a test document for embedding."
    test_query = "What is the meaning of life?"

    # Gemini / Gemma models offer different task types that can improve performance - not needed for OpenAI / Ollama
    document_embedding = embedding_model.single_embed(test_document, task_type="RETRIEVAL_DOCUMENT")
    query_embedding = embedding_model.single_embed(test_query, task_type="RETRIEVAL_QUERY")
    print(f"\n\nEmbedding shapes  -  Doc: {document_embedding.shape}  Query: {query_embedding.shape}")

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
    print(f"\n\nRAG Query: {rag_query.query}\nRAG Response JSON:")
    print(json.dumps(json.loads(rag_response.to_json()), indent=2))

    # 5. Store Vector DB to disk
    rag_db.vector_db.database.write_parquet("rag_vector_db.parquet")

    # 6. Load Vector DB from disk
    loaded_db = pl.read_parquet("rag_vector_db.parquet")
    rag_db_loaded = RagDatabase(database=loaded_db)
    print("Loaded RAG Database from disk with", rag_db_loaded.vector_db.database.height, "documents.")


if __name__ == "__main__":
    main()