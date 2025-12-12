import json
import os
from pathlib import Path
import random
import time
from typing import Any, List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_baseclient.client import LLMClient
import numpy as np
import polars as pl
from transformers import AutoTokenizer

from ._logger import get_logger
from .dataclasses import RAGIngestionPayload, RAGQuery, RAGResponse
from .rag_config import (
    BATCH_SIZE,
    CHUNKING_OVERLAP,
    HF_TOKEN,
    MODEL_CONFIG,
    TIMEOUT,
    DatabaseKeys,
    empty_rag_schema,
)

logger = get_logger()


class EmbeddingModel:
    """Wrapper using litellm's client for unified embedding calls."""

    def __init__(self, model: str) -> None:
        """Initialize the tokenizer and the unified LLM client."""

        # Handle restricted access for specific models (e.g., Gemma)
        if model == "embeddinggemma:300m": 
            from huggingface_hub import login
            hf_token = HF_TOKEN
            login(token=hf_token)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG[self.model]["tokenizer"])
        self.llm_client = LLMClient()

    def chunk_texts_tokenaware(self, texts: list[str]) -> list[list[str]]:
        """Splits text using a token-aware, hierarchical, general-purpose strategy with contextual overlap."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MODEL_CONFIG[self.model]["max_tokens"],
            chunk_overlap=CHUNKING_OVERLAP,
            length_function=lambda text: len(self.tokenizer.encode(text)), # token-aware splitting
            separators=["\n\n", "\n", " "], # hierarchical splitting - adjust as needed eg # ## ### for markdown articles
        )

        all_chunked_texts = []
        for text in texts:
            # The splitter returns a list of strings (chunks) for the document.
            chunks = text_splitter.split_text(text)
            if len(chunks) > 1:
                logger.debug(f"RAG Database: Text split into {len(chunks)} chunks.")
            all_chunked_texts.append(chunks)

        return all_chunked_texts

    def chunk_texts_naive(self, texts: list[str]) -> list[list[str]]:
        """Split chunks (naive general purpose approach - split by max tokens, ignores semantics & overlap)."""

        chunked_texts = []
        for text in texts:
            text_chunks = []
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= MODEL_CONFIG[self.model]["max_tokens"]:
                chunked_texts.append([text])
            else:
                for i in range(0, len(tokens), MODEL_CONFIG[self.model]["max_tokens"]):
                    chunk = tokens[i : i + MODEL_CONFIG[self.model]["max_tokens"]]
                    chunk_text = self.tokenizer.decode(chunk)
                    text_chunks.append(chunk_text)

                chunked_texts.append(text_chunks)

        return chunked_texts

    def embed(self,
        texts: Union[str, List[str]],
        **kwargs: dict[str, Any],
    )-> Union[np.ndarray, List[np.ndarray]]:
        """
        Unified embedding method for single strings or batches.
        Handles chunking, API calls via LLMClient, and aggregating (mean pooling) chunks back to documents.

        API Models:
            - Assumes API key in system environment variable.
        Local Models:
            - CPU inference: Assumes Ollama is installed configured & model is downloaded.
            - GPU inference: Assumes vLLM compatible local model is downloaded and configured.
        """
        # 1. Standardize Input
        is_single_input = isinstance(texts, str)
        input_texts = [texts] if is_single_input else texts

        if not input_texts or input_texts == [""]:
            empty_vec = np.zeros(MODEL_CONFIG[self.model]["dimensions"])
            return empty_vec if is_single_input else [empty_vec]

        # 2. Chunking
        # Returns list of lists: [[doc1_chunk1, doc1_chunk2], [doc2_chunk1]]
        chunked_texts = self.chunk_texts_tokenaware(input_texts)

        # 3. Flatten for API efficiency
        lengths = [len(chunks) for chunks in chunked_texts]
        flattened_chunks = [chunk for chunks in chunked_texts for chunk in chunks]

        # 4. Call Unified Client in batches
        all_embeddings = []

        # Gemini free tier has a hard limit of 20 inputs per request
        batch_size = BATCH_SIZE if "gemini" not in self.model else 19

        for i in range(0, len(flattened_chunks), batch_size):
            batch = flattened_chunks[i:i + batch_size]
            max_retries = 4
            base_delay = 15 # seconds

            for attempt in range(max_retries): # Retry up to max_retries
                try:
                    response = self.llm_client.get_embedding(
                        model=self.model,
                        input_text=batch,
                        timeout=TIMEOUT,
                        **kwargs,
                    )
                    batch_embeddings = [item['embedding'] for item in response['data']]
                    all_embeddings.extend(batch_embeddings)
                    break  # Success: exit the retry loop for batch

                except Exception as e:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 5s, 10s, 20s, 40s
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Embedding failed on attempt {attempt + 1}/{max_retries}. "
                            f"Retrying in {delay:.2f}s... Error: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Embedding failed for batch {i//BATCH_SIZE + 1} "
                            f"after {max_retries} attempts. Error: {e}"
                        )
                        raise e

        # 5. Extract Embeddings
        raw_embeddings = np.array(all_embeddings)

        # 6. Reconstruct Structure (Split flattened list back into document groups)
        grouped_embeddings = np.split(raw_embeddings, np.cumsum(lengths)[:-1]) if len(lengths) > 1 else [raw_embeddings]

        # 7. Aggregate (Mean Pooling) per document
        aggregated_embeddings = [
            np.mean(group, axis=0) for group in grouped_embeddings
        ]

        if is_single_input:
            return aggregated_embeddings[0]

        return np.array(aggregated_embeddings)


class VectorDB:
    """
    A simple in-memory vector database using Polars and NumPy.
    Supports initialization with a stored Polars DataFrame to avoid rebuilding the DB. 
    """

    def __init__(self, database: pl.DataFrame) -> None:
        self.database = database

    def similarity_search(self, query_embedding: np.ndarray, k_documents: int = 20) -> RAGResponse:
        """
        Search for similar vectors in the database using cosine similarity.

        Parameters:
            query_embedding (np.ndarray): Query vector of shape (embedding_dim,)
            k_documents (int): Number of top similarity results to return.

        Returns:
            RAGResponse
        """

        db_vectors = np.stack(self.database[DatabaseKeys.KEY_EMBEDDINGS].to_list())

        # Compute cosine similarity
        dot_product = np.dot(db_vectors, query_embedding)
        db_norms = np.linalg.norm(db_vectors, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        cosine_similarities = dot_product / (db_norms * query_norm)

        # Get top-k indices
        top_indices = np.argsort(cosine_similarities)[-k_documents:][::-1]
        df_top_k = self.database[top_indices]

        return RAGResponse(
            titles=df_top_k[DatabaseKeys.KEY_TITLE].to_list(),
            texts=df_top_k[DatabaseKeys.KEY_TXT_RETRIEVAL].to_list(),
            similarities=cosine_similarities[top_indices].tolist(),
            metadata=[json.loads(meta) for meta in df_top_k[DatabaseKeys.KEY_METADATA].to_list()],
        )


class RagDatabase:
    """Database for Retrieval Augmented Generation (RAG)"""

    def __init__(self, model:str, database: Optional[pl.DataFrame]=None, embedding_dimensions: Optional[int]=None) -> None:
        """Initialize RAG Database with embedding model and vector DB."""
        try:

            if (database is None and embedding_dimensions is None):
                logger.error("RAG Database: Either 'database' or 'embedding_dimensions' must be provided.")
                raise ValueError("Either 'database' or 'embedding_dimensions' must be provided.")

            if database is None:
                logger.info("RAG Database: Initializing empty RAG database.")
                database = empty_rag_schema(dimensions=embedding_dimensions)
            else:
                logger.info("RAG Database: Initializing RAG database from DataFrame.")

            self.embedding_model: EmbeddingModel = EmbeddingModel(model=model)
            self.vector_db: VectorDB = VectorDB(database=database)

        except Exception:
            logger.error("RAG Database: Error initializing RAG Database:")
            raise

    @classmethod
    def from_parquet(cls, parquet_path: Union[str, Path], model: str) -> "RagDatabase":
        """Initialize RAG Database from stored parquet file."""
        try:
            logger.info(f"RAG Database: Loading RAG database from {parquet_path}")
            database = pl.read_parquet(parquet_path)
            return cls(model=model, database=database)
        except Exception:
            logger.error("RAG Database: Error loading RAG Database from parquet:")
            raise

    def is_document_in_database(self, title: str) -> bool:
        """Check if a document with the given title exists in the database."""
        return title in self.vector_db.database[DatabaseKeys.KEY_TITLE].to_list()

    def rag_process_query(self, rag_query: RAGQuery, **kwargs: dict[str, Any]) -> RAGResponse:
        """Process RAG query and return relevant results"""
        try:
            # Query embedding might also benefit from a prefix, e.g., "query: "
            query_text_for_embedding = f"query: {rag_query.query}"
            query_embedding = self.embedding_model.embed(query_text_for_embedding, **kwargs)
        except Exception:
            logger.error("RAG Database: Error embedding query:")
            raise

        return self.vector_db.similarity_search(query_embedding, rag_query.k_documents)

    def add_documents(self, payload: RAGIngestionPayload, **kwargs: dict[str, Any]) -> None:
        """Add documents to the RAG database from an ingestion payload."""
        # Extract data from payload
        _titles = payload.df[DatabaseKeys.KEY_TITLE].to_list()
        active_texts_embedding = payload.df[DatabaseKeys.KEY_TXT_EMBEDDING].to_list()
        active_texts_retrieval = payload.df[DatabaseKeys.KEY_TXT_RETRIEVAL].to_list()
        metadata_json = payload.df[DatabaseKeys.KEY_METADATA].to_list()

        logger.info(f"RAG Database: Using embedding model {self.embedding_model.model}")
        logger.info(f"RAG Database: Probing {len(_titles)} documents from payload")

        try:
            texts_to_embed_filtered = []
            titles_to_embed = []
            texts_to_retrieve_filtered = []
            metadata_to_add_json = []

            # Filter duplicates or updates
            for text_embedding_item, title, text_retrieval_item, meta_json_item in zip(
                active_texts_embedding, _titles, active_texts_retrieval, metadata_json, strict=True
            ):
                if not self.is_document_in_database(title):
                    texts_to_embed_filtered.append(text_embedding_item)
                    texts_to_retrieve_filtered.append(text_retrieval_item)
                    titles_to_embed.append(title)
                    metadata_to_add_json.append(meta_json_item)
                else:
                    # Check if text used for embedding has changed
                    existing_text = self.vector_db.database.filter(
                        pl.col(DatabaseKeys.KEY_TITLE) == title
                    )[DatabaseKeys.KEY_TXT_EMBEDDING][0]
                    if existing_text != text_embedding_item:
                        texts_to_embed_filtered.append(text_embedding_item)
                        texts_to_retrieve_filtered.append(text_retrieval_item)
                        titles_to_embed.append(title)
                        metadata_to_add_json.append(meta_json_item)
                        # Remove old entry if text has changed to update it
                        self.vector_db.database = self.vector_db.database.filter(pl.col(DatabaseKeys.KEY_TITLE) != title)

            if not texts_to_embed_filtered:
                logger.info("RAG Database: No new documents to add.")
                return

            logger.info(f"RAG Database: Found {len(texts_to_embed_filtered)} new documents. Proceeding to embed...")

            embeddings = self.embedding_model.embed(texts_to_embed_filtered, **kwargs)
            new_entries = pl.DataFrame(
                {
                    DatabaseKeys.KEY_TITLE: titles_to_embed,
                    DatabaseKeys.KEY_METADATA: metadata_to_add_json,
                    DatabaseKeys.KEY_TXT_RETRIEVAL: texts_to_retrieve_filtered, # Store text used for retrieval
                    DatabaseKeys.KEY_TXT_EMBEDDING: texts_to_embed_filtered, # Store text used for embedding
                    DatabaseKeys.KEY_EMBEDDINGS: embeddings,
                }
            )

            self.vector_db.database = pl.concat([self.vector_db.database, new_entries])

        except Exception:
            logger.error("RAG Database: Error adding documents:")
            raise

        logger.info("RAG Database: Documents added successfully.")
