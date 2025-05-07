"""Local embedding models using various providers."""
import os
import logging
from typing import List, Optional

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

def get_embeddings_model() -> Embeddings:
    """Get the embeddings model to use.
    
    Returns:
        An embeddings model based on available API keys and environment settings.
    """
    from backend.utils import get_bool_env, get_env

    # Use Ollama for local embedding if specified
    if get_bool_env("USE_OLLAMA", False):
        base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=base_url,
        )
    
    # Try different embedding providers based on available API keys
    provider_embeddings = {
        "openai": _get_openai_embeddings,
        "cohere": _get_cohere_embeddings,
        "voyageai": _get_voyageai_embeddings,
        "google": _get_google_embeddings,
    }
    
    # Try each provider based on available API keys
    # Order of precedence: OpenAI, Cohere, VoyageAI, Google
    providers_to_try = ["openai", "cohere", "voyageai", "google"]
    for provider in providers_to_try:
        try:
            embeddings_func = provider_embeddings.get(provider)
            if embeddings_func:
                embeddings = embeddings_func()
                if embeddings:
                    logger.info(f"Using {provider} for embeddings")
                    return embeddings
        except Exception as e:
            logger.warning(f"Error initializing {provider} embeddings: {e}")
    
    # Fallback to dummy embeddings if no provider is available
    logger.warning("No embedding provider available. Using dummy embeddings.")
    return _get_dummy_embeddings()


def _get_openai_embeddings() -> Optional[Embeddings]:
    """Get OpenAI embeddings model."""
    from backend.utils import get_env
    
    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        return None
        
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_cohere_embeddings() -> Optional[Embeddings]:
    """Get Cohere embeddings model."""
    from backend.utils import get_env
    
    api_key = get_env("COHERE_API_KEY")
    if not api_key:
        return None
        
    try:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model="embed-english-v3.0")
    except ImportError:
        logger.warning("Cohere package not installed. Skipping.")
        return None


def _get_voyageai_embeddings() -> Optional[Embeddings]:
    """Get VoyageAI embeddings model."""
    from backend.utils import get_env
    
    api_key = get_env("VOYAGEAI_API_KEY")
    if not api_key:
        return None
    
    try:
        from voyageai import get_embeddings as voyage_get_embeddings
        
        # Create custom embeddings class for VoyageAI
        class VoyageAIEmbeddings(Embeddings):
            """Wrapper for VoyageAI embeddings."""
            
            def __init__(self, api_key: str, model: str = "voyage-2"):
                """Initialize VoyageAI embeddings.
                
                Args:
                    api_key: VoyageAI API key
                    model: VoyageAI model name
                """
                self.api_key = api_key
                self.model = model
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed documents using VoyageAI."""
                return voyage_get_embeddings(
                    texts,
                    model=self.model,
                    api_key=self.api_key,
                )
            
            def embed_query(self, text: str) -> List[float]:
                """Embed query using VoyageAI."""
                embeddings = voyage_get_embeddings(
                    [text],
                    model=self.model,
                    api_key=self.api_key,
                )
                return embeddings[0]
        
        return VoyageAIEmbeddings(api_key=api_key)
    except ImportError:
        logger.warning("VoyageAI package not installed. Skipping.")
        return None


def _get_google_embeddings() -> Optional[Embeddings]:
    """Get Google embeddings model."""
    from backend.utils import get_env
    
    api_key = get_env("GOOGLE_API_KEY")
    if not api_key:
        return None
        
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
    except ImportError:
        logger.warning("Google GenerativeAI package not installed. Skipping.")
        return None


def _get_dummy_embeddings() -> Embeddings:
    """Get dummy embeddings model for fallback."""
    class DummyEmbeddings(Embeddings):
        """Dummy embeddings that return fixed-size vectors of zeros."""
        
        def __init__(self, size: int = 768):
            """Initialize dummy embeddings.
            
            Args:
                size: Size of the embedding vectors
            """
            self.size = size
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed documents with zeros."""
            return [[0.0] * self.size for _ in texts]
        
        def embed_query(self, text: str) -> List[float]:
            """Embed query with zeros."""
            return [0.0] * self.size
    
    return DummyEmbeddings()