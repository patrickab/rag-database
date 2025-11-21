from dataclasses import dataclass
import json
import os
from typing import Optional

from google.genai import Client as GeminiClient
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from ollama import Client as OllamaClient
from openai import OpenAI as OpenAI
import polars as pl
import tqdm
from transformers import AutoTokenizer

from rag_database.rag_config import (
    CHUNKING_OVERLAP,
    DEFAULT_EMBEDDING_MODEL,
    EMPTY_RAG_SCHEMA,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODELS,
    MODEL_CONFIG,
    OLLAMA_EMBEDDING_MODELS,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODELS,
    TIMEOUT,
    DatabaseKeys,
)


@dataclass
class RAGQuery:
    """RAG Query Structure"""

    query: str
    k_documents: int


@dataclass
class RAGResponse:
    """Response from a RAG Database Query"""

    titles: list[str]
    texts: list[str]
    similarities: list[float]

    def to_json(self) -> str:
        """Convert RAGResponse to a JSON string."""
        return json.dumps(
            {
                DatabaseKeys.KEY_TITLE: self.titles,
                DatabaseKeys.KEY_SIMILARITIES: self.similarities,
                DatabaseKeys.KEY_TXT: self.texts,
            },
            ensure_ascii=False,
            indent=2,
        )

    def to_polars(self) -> pl.DataFrame:
        """Convert RAGResponse to a Polars DataFrame."""
        return pl.DataFrame(
            {
                DatabaseKeys.KEY_SIMILARITIES: self.similarities,
                DatabaseKeys.KEY_TITLE: self.titles,
                DatabaseKeys.KEY_TXT: self.texts,
            }
        )

class EmbeddingModel:
    """Wrapper for the OpenAI Embedding Model"""

    def __init__(self, model:str=DEFAULT_EMBEDDING_MODEL) -> None:
        """Initialize the async embedding client and tokenizer."""

        if model == "embeddinggemma:300m": # manage restricted access
            from huggingface_hub import login

            hf_token = os.getenv("HUGGINGFACE_API_KEY")
            login(token=hf_token)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG[self.model]["tokenizer"])

        if self.model in OLLAMA_EMBEDDING_MODELS:
            self.ollama_client = OllamaClient(host="http://localhost:11434")

        if self.model in OPENAI_EMBEDDING_MODELS:
            self.openai_sync_client = OpenAI(api_key=OPENAI_API_KEY)

        if self.model in GEMINI_EMBEDDING_MODELS:
            self.gemini_sync_client = GeminiClient(api_key=GEMINI_API_KEY)

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

    def single_embed(self, document: str, task_type: Optional[str]=None) -> np.ndarray:
        """Embed a single text."""

        if document == "":
            return np.zeros(MODEL_CONFIG[self.model]["dimensions"])

        chunked_texts = self.chunk_texts_tokenaware([document])

        embeddings_list: list[np.ndarray] = []  

        if self.model in GEMINI_EMBEDDING_MODELS:
            config = types.EmbedContentConfig(
                output_dimensionality=MODEL_CONFIG[self.model]["dimensions"],
                task_type=task_type,
            )

            response = self.gemini_sync_client.models.embed_content(
                model=self.model,
                contents=chunked_texts,
                config=config
            )
            embeddings = [np.array(e.values) for e in response.embeddings]
            embeddings_list.extend(embeddings)

        if self.model in OPENAI_EMBEDDING_MODELS:
            for chunk in chunked_texts:
                response = self.openai_sync_client.embeddings.create(
                    input=chunk,
                    model=self.model,
                    dimensions=MODEL_CONFIG[self.model]["dimensions"],
                    timeout=TIMEOUT,
                )
                embedding = np.array(response.data[0].embedding)
                embeddings_list.append(embedding)

        if self.model in OLLAMA_EMBEDDING_MODELS:
            for chunk in chunked_texts:
                response = self.ollama_client.embed(
                    model=self.model,
                    input=chunk,
                    dimensions=MODEL_CONFIG[self.model]["dimensions"],
                )
                embedding = np.array(response.embeddings[0])
                embeddings_list.append(embedding)

        aggregated_embedding = np.mean(embeddings_list, axis=0)

        return aggregated_embedding

    def embed_batch(self, texts: list[str], task_type: Optional[str]) -> np.ndarray:
        """
        Embed many texts in one API call.
        Calculates document-level embeddings. Considers chunking.
        Chapter-wise embeddings can be considered for pdf's articles etc.
        """
        chunked_texts = self.chunk_texts_tokenaware(texts)

        # 1. Flatten Chunks & Store Mapping
        lengths = [len(chunks) for chunks in chunked_texts]
        flattend_chunked_texts = [chunk for chunks in chunked_texts for chunk in chunks]

        if self.model in OPENAI_EMBEDDING_MODELS:

            # 2. Batch Embed Flattened Chunks
            response = self.openai_sync_client.embeddings.create(
                input=flattend_chunked_texts,
                model=self.model,
                dimensions=MODEL_CONFIG[self.model]["dimensions"],
                timeout=TIMEOUT,
            )

            embeddings = np.array([d.embedding for d in response.data])

        if self.model in GEMINI_EMBEDDING_MODELS:

            config = types.EmbedContentConfig(
                output_dimensionality=MODEL_CONFIG[self.model]["dimensions"],
                task_type=task_type)

            # 2. Batch Embed Flattened Chunks
            response = self.gemini_sync_client.models.embed_content(
                model=self.model,
                contents=flattend_chunked_texts,
                config=config
            )

            embeddings = np.array([e.values for e in response.embeddings])

        # NOTE: Ollama batch embedding not supported yet.

        # 3. Reconstruct Structure
        embeddings = np.split(embeddings, np.cumsum(lengths)[:-1])

        # 4. Aggregate Embeddings per Text
        aggregated_embeddings = [
            np.mean(emb_list, axis=0) for emb_list in embeddings
        ]
        return aggregated_embeddings


class VectorDB:
    """
    A simple in-memory vector database using Polars and NumPy.
    Supports initialization with a stored Polars DataFrame to avoid rebuilding the DB. 
    """

    def __init__(self, database: pl.DataFrame=EMPTY_RAG_SCHEMA) -> None:
        """Defaults to empty EMPTY_RAG_SCHEMA but can be initialized with existing dataframe."""
        if database.schema == EMPTY_RAG_SCHEMA.schema:
            self.database = database
        else:
            error_msg = (
                f"Dataframe does not match the required schema: {EMPTY_RAG_SCHEMA.schema}"
            )
            raise ValueError(error_msg)

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
            texts=df_top_k[DatabaseKeys.KEY_TXT].to_list(),
            similarities=cosine_similarities[top_indices].tolist(),
        )


class RagDatabase:
    """Database for Retrieval Augmented Generation (RAG)"""

    def __init__(self, database: pl.DataFrame=EMPTY_RAG_SCHEMA, model:str=DEFAULT_EMBEDDING_MODEL) -> None:
        """
        Initialize RAG Database with embedding model and vector DB.
        Can be initialized with existing dataframe to avoid rebuilding the DB.
        Defaults to empty EMPTY_RAG_SCHEMA if no dataframe is provided.
        """
        self.embedding_model = EmbeddingModel(model=model)
        self.vector_db = VectorDB(database=database)

    def is_document_in_database(self, title: str) -> bool:
        """Check if a document with the given title exists in the database."""
        return title in self.vector_db.database[DatabaseKeys.KEY_TITLE].to_list()

    def rag_process_query(self, rag_query: RAGQuery) -> RAGResponse:
        """Process RAG query and return relevant results"""
        query_embedding = self.embedding_model.single_embed(rag_query.query, task_type="RETRIEVAL_QUERY")
        return self.vector_db.similarity_search(query_embedding, rag_query.k_documents)

    def add_documents(self, titles: str|list[str], texts: str|list[str]) -> None:
        """Add documents to the RAG database."""

        try:
            texts_to_embed = []
            for text,title in zip(texts, titles, strict=True):
                if not self.is_document_in_database(title):
                    texts_to_embed.append(text)
                else:
                    # Check if text has changed - if so, re-embed and update
                    existing_text = self.vector_db.database.filter(pl.col(DatabaseKeys.KEY_TITLE) == title)[DatabaseKeys.KEY_TXT][0]
                    if existing_text != text:
                        texts_to_embed.append(text)
                        self.vector_db.database = self.vector_db.database.filter(pl.col(DatabaseKeys.KEY_TITLE) != title)

            if self.embedding_model.model in OPENAI_EMBEDDING_MODELS:
                embeddings = self.embedding_model.embed_batch(texts_to_embed)

                new_entries = pl.DataFrame(
                    {
                        DatabaseKeys.KEY_TITLE: titles,
                        DatabaseKeys.KEY_TXT: texts,
                        DatabaseKeys.KEY_EMBEDDINGS: embeddings,
                    }
                )

            if self.embedding_model.model in GEMINI_EMBEDDING_MODELS:
                embeddings = self.embedding_model.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

                new_entries = pl.DataFrame(
                    {
                        DatabaseKeys.KEY_TITLE: titles,
                        DatabaseKeys.KEY_TXT: texts,
                        DatabaseKeys.KEY_EMBEDDINGS: embeddings,
                    }
                )

            if self.embedding_model.model in OLLAMA_EMBEDDING_MODELS:

                embeddings = [
                    self.embedding_model.single_embed(text) for text in tqdm.tqdm(texts)
                ]

                new_entries = pl.DataFrame(
                    {
                        DatabaseKeys.KEY_TITLE: titles,
                        DatabaseKeys.KEY_TXT: texts,
                        DatabaseKeys.KEY_EMBEDDINGS: embeddings,
                    }
                )

            self.vector_db.database = pl.concat([self.vector_db.database, new_entries])

        except Exception as e:
            print(f"Error adding documents: {e}")
            raise e
