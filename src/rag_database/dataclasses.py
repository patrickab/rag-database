from dataclasses import dataclass
import json
import pathlib
from typing import Any, Optional, Self

import polars as pl

from ._logger import get_logger
from .rag_config import DatabaseKeys

logger = get_logger()

@dataclass
class RAGQuery:
    """RAG Query Structure"""

    query: str
    k_documents: int

@dataclass
class RAGIngestionPayload:
    """Payload for RAG Document Ingestion"""
    """
    A batch of documents intended for RAG database ingestion.
    Encapsulates titles, texts for embedding, texts for retrieval, and metadata
    within a Polars DataFrame for efficient processing and serialization.
    """
    df: pl.DataFrame

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df
        self._validate_schema()

    @classmethod
    def from_parquet(cls, path: pathlib.Path) -> Self:
        """Reads a document batch from a Parquet file."""
        df = pl.read_parquet(path)
        return cls(df)

    def to_parquet(self, path: pathlib.Path) -> None:
        """Writes the document batch to a Parquet file."""
        self.df.write_parquet(path)

    @classmethod
    def from_lists(
        cls,
        titles: list[str],
        metadata: list[dict[str, Any]],
        texts: Optional[list[str]] = None,
        texts_embedding: Optional[list[str]] = None,
        texts_retrieval: Optional[list[str]] = None,
    ) -> Self:
        """
        Constructs a RAGIngestionPayload from lists of components.
        Requires either 'texts' (for both embedding and retrieval)
        OR 'texts_embedding' and 'texts_retrieval' together.
        """
        is_texts_only = texts is not None and texts_embedding is None and texts_retrieval is None
        is_embedding_retrieval_together = texts is None and texts_embedding is not None and texts_retrieval is not None

        if not (is_texts_only or is_embedding_retrieval_together):
            raise ValueError(
                "Invalid document input. Provide either 'texts' alone (for both embedding and retrieval), "
                "or 'texts_embedding' and 'texts_retrieval' together."
            )

        active_texts_embedding = texts if is_texts_only else texts_embedding
        active_texts_retrieval = texts if is_texts_only else texts_retrieval

        # Basic length checks
        if not (len(titles) == len(active_texts_embedding) == len(active_texts_retrieval) == len(metadata)):
            raise ValueError(
                f"All input lists (titles, metadata, embedding texts, retrieval texts) must have the same length. "
                f"Titles: {len(titles)}, Metadata: {len(metadata)}, "
                f"Embedding texts: {len(active_texts_embedding)}, Retrieval texts: {len(active_texts_retrieval)}"
            )

        # Convert metadata dicts to JSON strings for efficient storage as pl.String
        serialized_metadata = [json.dumps(m) for m in metadata]

        df = pl.DataFrame({
            DatabaseKeys.KEY_TITLE: titles,
            DatabaseKeys.KEY_TXT_EMBEDDING: active_texts_embedding,
            DatabaseKeys.KEY_TXT_RETRIEVAL: active_texts_retrieval,
            DatabaseKeys.KEY_METADATA: serialized_metadata,
        })
        return cls(df)

    def _validate_schema(self) -> None:
        """Ensures the DataFrame conforms to the expected minimal schema for a batch."""
        expected_cols_and_types = {
            DatabaseKeys.KEY_TITLE: pl.Utf8,
            DatabaseKeys.KEY_TXT_EMBEDDING: pl.Utf8,
            DatabaseKeys.KEY_TXT_RETRIEVAL: pl.Utf8,
            DatabaseKeys.KEY_METADATA: pl.String,
        }
        for col, dtype in expected_cols_and_types.items():
            if col not in self.df.columns:
                raise ValueError(f"RAGIngestionPayload missing required column: '{col}'")
            if self.df[col].dtype != dtype:
                raise TypeError(f"Column '{col}' in RAGIngestionPayload - incorrect dtype: expected {dtype}, got {self.df[col].dtype}")

    @property
    def dataframe(self) -> pl.DataFrame:
        """Returns a clone of the internal Polars DataFrame to prevent external modification."""
        return self.df.clone()

    @property
    def titles(self) -> list[str]:
        return self.df[DatabaseKeys.KEY_TITLE].to_list()

    @property
    def texts_embedding(self) -> list[str]:
        return self.df[DatabaseKeys.KEY_TXT_EMBEDDING].to_list()

    @property
    def texts_retrieval(self) -> list[str]:
        return self.df[DatabaseKeys.KEY_TXT_RETRIEVAL].to_list()

    @property
    def metadata(self) -> list[dict[str, Any]]:
        """
        Returns metadata as a list of parsed dictionaries.
        Assumes metadata in the DataFrame is stored as JSON strings.
        """
        return [json.loads(s) for s in self.df[DatabaseKeys.KEY_METADATA].to_list()]

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"DocumentBatch(num_documents={len(self)}, titles={self.titles}, metadata={self.metadata})"

@dataclass
class RAGResponse:
    """Response from a RAG Database Query"""

    titles: list[str]
    texts: list[str]
    similarities: list[float]
    metadata: list[dict[str, Any]]

    def __str__(self) -> str:
        """Provide a user-friendly string representation with all results in columns."""

        output_lines = [
            f"RAGResponse: {len(self.titles)} results\n",
            f"{'Title':<50} {'Similarity':>10}"
        ]
        output_lines.append("-" * 61)

        for title, similarity in zip(self.titles, self.similarities, strict=False):
            truncated_title = title[:47] + "..." if len(title) > 50 else title
            output_lines.append(f"{truncated_title:<50} {similarity:>10.4f}")

        result_str = "\n".join(output_lines)
        logger.info(result_str)
        return f"RAGResponse ({len(self.titles)} results)"

    def to_json(self) -> str:
        """Convert RAGResponse to a JSON string."""
        return json.dumps(
            {
                DatabaseKeys.KEY_SIMILARITIES: self.similarities,
                DatabaseKeys.KEY_TITLE: self.titles,
                DatabaseKeys.KEY_METADATA: self.metadata,
                DatabaseKeys.KEY_TXT_RETRIEVAL: self.texts,
            },
            ensure_ascii=False,
            indent=2,
        )

    def to_polars(self) -> pl.DataFrame:
        """Convert RAGResponse to a Polars DataFrame, serializing metadata to JSON strings."""
        metadata_json = [json.dumps(m) for m in self.metadata]
        return pl.DataFrame(
            {
                DatabaseKeys.KEY_SIMILARITIES: self.similarities,
                DatabaseKeys.KEY_TITLE: self.titles,
                DatabaseKeys.KEY_TXT_RETRIEVAL: self.texts,
                DatabaseKeys.KEY_METADATA: metadata_json,
            }
        )
