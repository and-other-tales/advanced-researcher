"""Local embedding models using Ollama."""
import os
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings


def get_embeddings_model() -> Embeddings:
    """Get the embeddings model to use."""
    # Use Ollama for local embedding if specified
    if os.environ.get("USE_OLLAMA") == "true":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=base_url,
        )
    
    # Default to OpenAI embeddings
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )