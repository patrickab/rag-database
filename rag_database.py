from dataclasses import dataclass
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from openai import AsyncOpenAI, OpenAI
import polars as pl
from transformers import AutoTokenizer

from rag_config import (
    DEFAULT_EMBEDDING_MODEL,
    MODEL_CONFIG,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODELS,
    RAG_SCHEMA,
    TIMEOUT,
    DatabaseKeys,
)


@dataclass
class RAGQuery:
    """RAG Query Structure"""

    query: str
    k_queries: int


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
                DatabaseKeys.KEY_TXT: self.texts,
                DatabaseKeys.KEY_SIMILARITIES: self.similarities,
            },
            ensure_ascii=False,
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

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG[self.model]["tokenizer"])

        def length_function(text:str) -> int: return len(self.tokenizer.encode(text))
        self.length_function = length_function

        if self.model in OPENAI_EMBEDDING_MODELS:
            self.openai_sync_client = OpenAI(api_key=OPENAI_API_KEY)
            self.openai_async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    def chunk_text_tokenaware(self, texts: list[str]) -> list[list[str]]:
        """Splits text using a token-aware, hierarchical, general-purpose strategy with contextual overlap."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MODEL_CONFIG[self.model]["max_tokens"],
            chunk_overlap=200, # Common value, can be tuned.
            length_function=self.length_function, # Tokenizer-based length function.
            separators=["\n\n", "\n", " ", ""], # Hierarchical separators. Will try to split "\n\n" first, then "\n", then " ", etc.
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

    def sync_embed(self, document: str) -> np.ndarray:
        """Embed a single text."""

        if document == "":
            return np.zeros(MODEL_CONFIG[self.model]["dimensions"])

        chunked_texts = self.chunk_text([document])

        embeddings_list: list[np.ndarray] = []
        for chunk in chunked_texts:
            if self.model in OPENAI_EMBEDDING_MODELS:
                response = self.openai_sync_client.embeddings.create(
                    input=chunk,
                    model=self.model,
                    dimensions=MODEL_CONFIG[self.model]["dimensions"],
                    timeout=TIMEOUT,
                )
                embedding = np.array(response.data[0].embedding)
                embeddings_list.append(embedding)

        aggregated_embedding = np.mean(embeddings_list, axis=0)

        return aggregated_embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed many texts in one API call.
        Calculates document-level embeddings. Considers chunking.
        Chapter-wise embeddings can be considered for pdf's articles etc.
        """

        chunked_texts = self.chunk_texts(texts)

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

    def __init__(self, dataframe: pl.DataFrame) -> None:
        if dataframe.schema == RAG_SCHEMA.schema:
            self.database = dataframe
        else:
            error_msg = (
                f"Dataframe does not match the required schema: {RAG_SCHEMA.schema}"
            )
            raise ValueError(error_msg)

    def similarity_search(self, query_vector: np.ndarray, k_queries: int = 20) -> RAGResponse:
        """
        Search for similar vectors in the database using cosine similarity.

        Parameters:
            query_vector (np.ndarray): Query vector of shape (embedding_dim,)
            k_queries (int): Number of top similarity results to return.

        Returns:
            RAGResponse
        """

        db_vectors = np.stack(self.database[DatabaseKeys.KEY_EMBEDDINGS].to_list())

        # Compute cosine similarity
        dot_product = np.dot(db_vectors, query_vector)
        db_norms = np.linalg.norm(db_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        cosine_similarities = dot_product / (db_norms * query_norm)

        # Get top-k indices
        top_indices = np.argsort(cosine_similarities)[-k_queries:][::-1]
        df_top_k = self.database[top_indices]

        return RAGResponse(
            titles=df_top_k[DatabaseKeys.KEY_TITLE].to_list(),
            texts=df_top_k[DatabaseKeys.KEY_TXT].to_list(),
            similarities=cosine_similarities[top_indices].tolist(),
        )


class RagDatabase:
    """Database for Retrieval Augmented Generation (RAG)"""

    def __init__(self, vector_db: VectorDB, model:str=DEFAULT_EMBEDDING_MODEL) -> None:
        self.embedding_model = EmbeddingModel(model=model)
        self.vector_db = vector_db

    async def rag_process_query(self, rag_query: RAGQuery) -> RAGResponse:
        """Process RAG query and return relevant results"""
        query_embedding = await self.embedding_model.embed(rag_query.query)
        return self.vector_db.similarity_search(query_embedding, rag_query.k_queries)
