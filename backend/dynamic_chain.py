"""Dynamic chain utilities for supporting multiple knowledge bases."""
import logging
import os
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import BaseModel

from backend.dynamic_routes import get_knowledge_base_by_id, get_knowledge_bases
from backend.local_embeddings import get_embeddings_model
from backend.domain_detection import get_domain_info, DomainType

logger = logging.getLogger(__name__)


class ChatRequestWithKB(BaseModel):
    """Chat request with knowledge base specification."""
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None
    knowledge_base_id: Optional[str] = None


async def get_domain_retriever(query: str, llm: Optional[LanguageModelLike] = None) -> BaseRetriever:
    """Get a retriever based on domain detection from the query.
    
    Args:
        query: The user's query
        llm: Optional language model for domain detection
        
    Returns:
        BaseRetriever: A retriever for the detected domain
    """
    # Detect the domain from the query
    domain_info = await get_domain_info(query, llm)
    graph_name = domain_info["graph_name"]
    
    logger.info(f"Domain detection for query: '{query[:50]}...' -> {domain_info['domain']} "
                f"(confidence: {domain_info['confidence']}, graph: {graph_name})")
    
    # Use the domain-specific graph
    if os.environ.get("WEAVIATE_URL") and os.environ.get("WEAVIATE_API_KEY"):
        # Use Weaviate for cloud deployment
        import weaviate
        from langchain_community.vectorstores import Weaviate
        
        weaviate_client = weaviate.Client(
            url=os.environ["WEAVIATE_URL"],
            auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
        )
        
        try:
            weaviate_client = Weaviate(
                client=weaviate_client,
                index_name=graph_name,
                text_key="text",
                embedding=get_embeddings_model(),
                by_text=False,
                attributes=["source", "title"],
            )
            return weaviate_client.as_retriever(search_kwargs=dict(k=6))
        except Exception as e:
            logger.error(f"Error creating Weaviate retriever for {graph_name}: {e}")
            # Fall back to default if there's an error
            from backend.local_chain import get_retriever
            return get_retriever()
    else:
        # Use Chroma for local deployment
        from langchain_community.vectorstores import Chroma
        
        try:
            chroma_client = Chroma(
                collection_name=graph_name,
                embedding_function=get_embeddings_model(),
            )
            return chroma_client.as_retriever(search_kwargs=dict(k=6))
        except Exception as e:
            logger.error(f"Error creating Chroma retriever for {graph_name}: {e}")
            # Fall back to default if there's an error
            from backend.local_chain import get_retriever
            return get_retriever()


def get_dynamic_retriever(kb_id: Optional[str] = None) -> BaseRetriever:
    """Get a retriever for the specified knowledge base.
    
    Args:
        kb_id: Optional knowledge base ID
        
    Returns:
        BaseRetriever: A retriever for the knowledge base
    """
    # If no KB ID is provided, use the default
    if not kb_id:
        from backend.local_chain import get_retriever
        return get_retriever()
    
    # Get the knowledge base metadata
    kb = get_knowledge_base_by_id(kb_id)
    if not kb:
        logger.warning(f"Knowledge base {kb_id} not found, using default retriever")
        from backend.local_chain import get_retriever
        return get_retriever()
    
    # Create a retriever for the knowledge base
    if os.environ.get("WEAVIATE_URL") and os.environ.get("WEAVIATE_API_KEY"):
        # Use Weaviate for cloud deployment
        import weaviate
        from langchain_community.vectorstores import Weaviate
        
        weaviate_client = weaviate.Client(
            url=os.environ["WEAVIATE_URL"],
            auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
        )
        
        try:
            weaviate_client = Weaviate(
                client=weaviate_client,
                index_name=kb_id,
                text_key="text",
                embedding=get_embeddings_model(),
                by_text=False,
                attributes=["source", "title"],
            )
            return weaviate_client.as_retriever(search_kwargs=dict(k=6))
        except Exception as e:
            logger.error(f"Error creating Weaviate retriever for {kb_id}: {e}")
            # Fall back to default if there's an error
            from backend.local_chain import get_retriever
            return get_retriever()
    else:
        # Use Chroma for local deployment
        from langchain_community.vectorstores import Chroma
        
        try:
            chroma_client = Chroma(
                collection_name=kb_id,
                embedding_function=get_embeddings_model(),
            )
            return chroma_client.as_retriever(search_kwargs=dict(k=6))
        except Exception as e:
            logger.error(f"Error creating Chroma retriever for {kb_id}: {e}")
            # Fall back to default if there's an error
            from backend.local_chain import get_retriever
            return get_retriever()


def create_dynamic_chain(base_chain: Runnable, llm: LanguageModelLike) -> Runnable:
    """Create a chain that supports dynamic knowledge base selection and domain detection."""
    
    @chain
    async def select_retriever(inputs: dict) -> dict:
        """Select the appropriate retriever based on KB ID or domain detection."""
        # First check if a specific knowledge base ID is provided
        kb_id = inputs.get("knowledge_base_id")
        
        if kb_id:
            # Use the specific knowledge base if provided
            retriever = get_dynamic_retriever(kb_id)
            logger.info(f"Using explicit knowledge base: {kb_id}")
            return {"retriever": retriever, **inputs}
        else:
            # Use domain detection to determine the appropriate retriever
            question = inputs.get("question", "")
            
            try:
                # Get the domain-specific retriever
                domain_retriever = await get_domain_retriever(question, llm)
                logger.info(f"Using domain-specific retriever for query: '{question[:50]}...'")
                return {"retriever": domain_retriever, **inputs}
            except Exception as e:
                logger.error(f"Error getting domain retriever: {e}")
                # Fall back to default retriever
                from backend.local_chain import get_retriever
                return {"retriever": get_retriever(), **inputs}
    
    # Create a wrapper chain that selects the appropriate retriever
    dynamic_chain = (
        RunnablePassthrough.assign(
            selected_inputs=select_retriever
        )
        | {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "retriever": lambda x: x["selected_inputs"]["retriever"],
        }
        | base_chain
    )
    
    return dynamic_chain