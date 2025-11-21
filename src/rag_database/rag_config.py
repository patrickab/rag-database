import os

import polars as pl

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TIMEOUT = 20

MODEL_CONFIG = {
    # All currently tested/supported embedding models - can be easily extended with any Ollama models
    "embeddinggemma:300m": {
        "max_tokens": 2048,
        "dimensions": 768, # max value, also allows smaller sizes
        "tokenizer": "google/embeddinggemma-300m" # used to select tokenizer
    },
    "text-embedding-3-large": {
        "max_tokens": 8191,
        "dimensions": 3072,
        "tokenizer": "DWDMaiMai/tiktoken_cl100k_base"
    },
    "text-embedding-3-small": {
        "max_tokens": 4096,
        "dimensions": 1536,
        "tokenizer": "DWDMaiMai/tiktoken_cl100k_base"
    },
    "gemini-embedding-001": {
        "max_tokens": 2048,
        "dimensions": 3072,
        "tokenizer": "DWDMaiMai/tiktoken_cl100k_base" # use OpenAI tokenizer as proxy for splitting chunks
    }
}

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

OPENAI_EMBEDDING_MODELS = ["text-embedding-3-large", "text-embedding-3-small"]
GEMINI_EMBEDDING_MODELS = ["gemini-embedding-001"]
OLLAMA_EMBEDDING_MODELS = ["embeddinggemma:300m"]

# Define RAG database schema
class DatabaseKeys:
    """Keys for the RAG DataFrame"""

    KEY_TITLE = "title"
    KEY_TXT = "text"
    KEY_EMBEDDINGS = "embeddings"
    KEY_SIMILARITIES = "similarities"

EMPTY_RAG_SCHEMA = pl.DataFrame(
    schema={
        DatabaseKeys.KEY_TITLE: pl.Utf8,
        DatabaseKeys.KEY_TXT: pl.Utf8,
        DatabaseKeys.KEY_EMBEDDINGS: pl.Array(pl.Float64, width=MODEL_CONFIG[DEFAULT_EMBEDDING_MODEL]["dimensions"]),
    }
)