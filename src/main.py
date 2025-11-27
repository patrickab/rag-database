import json
import os

import polars as pl

from rag_database._logger import get_logger
from rag_database.rag_config import DEFAULT_EMBEDDING_MODEL, MODEL_CONFIG
from rag_database.rag_database import EmbeddingModel, RagDatabase, RAGIngestionPayload, RAGQuery


def main() -> None:
    """Demonstrate RAG Database functionality."""

    # ---------------------------------------------------------
    # 1. Simple Embedding Demo (Manual Usage)
    # ---------------------------------------------------------
    logger = get_logger()
    logger.info("--- Starting Manual Embedding Demo ---")

    # Initialize model
    embedding_model = EmbeddingModel(model=DEFAULT_EMBEDDING_MODEL)

    test_document = "This is a test document for embedding."
    test_query = "What is the meaning of life?"

    logger.info("Embedding Demo: Embedding Document (Default)...")
    document_embedding = embedding_model.embed(texts=test_document)
    query_embedding = embedding_model.embed(texts=test_query)

    # Gemini/Gemma models support task-specific embeddings
    # Any other model specific kwargs can be passed as needed
    # -> leave empty for default behavior or customize to your models capabilities
    kwargs_doc = {"task_type": "RETRIEVAL_DOCUMENT"}
    kwargs_query = {"task_type": "RETRIEVAL_QUERY"}

    logger.info("Embedding Demo: Embedding Document (Task: RETRIEVAL_DOCUMENT)...")
    document_embedding = embedding_model.embed(texts=test_document,**kwargs_doc) # noqa

    logger.info("Embedding Demo: Embedding Query (Task: RETRIEVAL_QUERY)...")
    query_embedding = embedding_model.embed(texts=test_query, **kwargs_query) # noqa
    logger.info("Embedding Demo: Completed.\n")

    # ---------------------------------------------------------
    # 2. RAG Database Demo (Automated usage)
    # ---------------------------------------------------------
    logger.info("\n--- Starting RAG Database Demo ---")

    # Initialize RAG Database - you will need to set embedding dimensions for Database allocation - this is done for memory efficiency
    rag_db = RagDatabase(model=DEFAULT_EMBEDDING_MODEL, embedding_dimensions=MODEL_CONFIG[DEFAULT_EMBEDDING_MODEL]["dimensions"])

    # Load documents from disk
    docs_dir = "example_docs/"
    texts = []
    titles = []

    if os.path.exists(docs_dir):
        documents = os.listdir(docs_dir)
        for doc in documents:
            path = os.path.join(docs_dir, doc)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    titles.append(doc)

    # create a dictionary of metadata with entries for each documents
    metadata = []
    for i in range(len(titles)):
        meta = {
            "source": titles[i],
            "length": len(texts[i]),
            "index": i
        }
        metadata.append(meta)

    ### Create ingestion payloads
    # 1. Using a single text source for both embedding and retrieval
    # 2. In practice it can be beneficial to separate texts for embedding vs retrieval
    payload_single_texts = RAGIngestionPayload.from_lists(titles=titles, texts=texts, metadata=metadata)
    payload_separate_texts = RAGIngestionPayload.from_lists(titles=titles, texts_embedding=texts, texts_retrieval=texts, metadata=metadata) # noqa

    # (Gemini/Gemma specific parameter RETRIEVAL_DOCUMENT improves performance)
    rag_db.add_documents(payload=payload_single_texts)
    rag_db.add_documents(payload=payload_separate_texts, task_type="RETRIEVAL_DOCUMENT")

    # Process a RAG Query
    rag_query = RAGQuery(
        query="What is the memory wall & how does it relate to Moores law?", 
        k_documents=5
    )

    # Query documents (default) vs (Gemini/Gemma specific)
    rag_response = rag_db.rag_process_query(rag_query)
    rag_response = rag_db.rag_process_query(rag_query, task_type="RETRIEVAL_QUERY")

    logger.info(f"RAG Demo: Query  -  {rag_query.query}")
    logger.info("RAG Demo: Response:")
    response_data = json.loads(rag_response.to_json())
    logger.info(f"RAG Demo: Titles {json.dumps(response_data["title"], indent=4)}")
    logger.info(f"RAG Demo: Similarities {json.dumps(response_data["similarities"], indent=4)}")

    # ---------------------------------------------------------
    # 3. Store/load Demo
    # ---------------------------------------------------------
    parquet_file = "rag_vector_db.parquet"

    rag_db.vector_db.database.write_parquet(parquet_file)                           # Store Vector DB to disk
    loaded_db = pl.read_parquet(parquet_file)                                       # Load Vector DB from disk
    rag_db_loaded = RagDatabase(model=DEFAULT_EMBEDDING_MODEL, database=loaded_db)  # Re-initialize RAG with the loaded dataframe

    count = rag_db_loaded.vector_db.database.height
    logger.info(f"Successfully loaded RAG Database from disk with {count} documents.")

if __name__ == "__main__":
    main()
