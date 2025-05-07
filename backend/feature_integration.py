"""Integration of special features with the main chat chain."""
import asyncio
import logging
import re
from typing import Dict, Any, Optional, List, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableLambda

from backend.feature_detection import (
    detect_feature_from_query,
    detect_dynamic_ingestion_phrases,
    detect_dataset_creation_phrases
)
from backend.dynamic_ingest import IngestRequest, process_ingestion_request
from backend.dataset_creator import DatasetRequest, create_dataset
from backend.domain_detection import get_domain_info, DomainType, DOMAIN_GRAPH_MAPPING

logger = logging.getLogger(__name__)


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text.
    
    Args:
        text: The text to extract URLs from
        
    Returns:
        List of URLs found in the text
    """
    # Regular expression to match URLs
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+(?!\S)'
    return re.findall(url_pattern, text)


async def handle_dynamic_ingestion(query: str, llm: Optional[LanguageModelLike] = None, user_id: Optional[str] = None) -> str:
    """Handle a dynamic ingestion request from the user.
    
    Args:
        query: The user's query containing URLs or ingestion request
        llm: Optional language model for domain detection
        user_id: Optional user ID for multi-user support
        
    Returns:
        str: Response message about the ingestion status
    """
    # Extract URLs from the query
    urls = extract_urls(query)
    
    if not urls:
        return "I couldn't find any URLs to ingest in your message. Please provide a valid URL."
    
    # Use the first URL for now (could be enhanced to handle multiple URLs)
    url = urls[0]
    
    # Determine source type - default to recursive URL
    source_type = "recursive_url"
    
    # Detect the domain from the URL and query
    domain_info = await get_domain_info(query + " " + url, llm)
    detected_domain = domain_info["domain"]
    graph_name = domain_info["graph_name"]
    
    # Determine a name for the knowledge base (use domain from URL)
    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if domain_match:
        name = f"{detected_domain}_{domain_match.group(1).replace('.', '_')}"
    else:
        name = f"{detected_domain}_{uuid.uuid4().hex[:8]}"
    
    # Log the domain detection
    logger.info(f"Detected domain for ingestion: {detected_domain} (graph: {graph_name})")
    logger.info(f"Ingesting URL: {url} into knowledge base: {name}")
    
    # Create the ingest request
    ingest_request = IngestRequest(
        source_type=source_type,
        name=name,
        url=url,
        user_id=user_id,
        max_depth=3,  # Use a default depth of 3 to avoid too much crawling
        graph_name=graph_name  # Use the domain-specific graph name
    )
    
    try:
        # Process the ingestion request
        ingest_id = await process_ingestion_request(ingest_request)
        
        # Return a response to the user
        return (
            f"I've started ingesting content from {url} into the {detected_domain.value.title()} knowledge graph. "
            f"The knowledge base ID is `{ingest_id}`. "
            f"This process will run in the background. "
            f"You can use this knowledge base in future queries by referring to this ID or by "
            f"asking questions related to {detected_domain.value}."
        )
    except Exception as e:
        logger.error(f"Error processing ingestion request: {e}")
        return f"I encountered an error while trying to ingest from {url}: {str(e)}"


async def handle_dataset_creation(query: str, llm: Optional[LanguageModelLike] = None, user_id: Optional[str] = None) -> str:
    """Handle a dataset creation request from the user.
    
    Args:
        query: The user's query containing dataset creation request
        llm: Optional language model for domain detection
        user_id: Optional user ID for multi-user support
        
    Returns:
        str: Response message about the dataset creation status
    """
    # Extract URLs from the query
    urls = extract_urls(query)
    
    if not urls:
        return "I couldn't find any URLs to create a dataset from in your message. Please provide a valid URL."
    
    # Use the first URL for now (could be enhanced to handle multiple URLs)
    url = urls[0]
    
    # Determine source type - default to recursive URL
    source_type = "recursive_url"
    
    # Detect the domain from the URL and query
    domain_info = await get_domain_info(query + " " + url, llm)
    detected_domain = domain_info["domain"]
    
    # Determine a name for the dataset
    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if domain_match:
        name = f"dataset_{detected_domain}_{domain_match.group(1).replace('.', '_')}"
    else:
        name = f"dataset_{detected_domain}_{uuid.uuid4().hex[:8]}"
    
    # Log the domain detection
    logger.info(f"Detected domain for dataset creation: {detected_domain}")
    logger.info(f"Creating dataset from URL: {url} with name: {name}")
    
    # Create the dataset request with default parameters
    dataset_request = DatasetRequest(
        source_type=source_type,
        name=name,
        url=url,
        dataset_type="question_generation",
        user_id=user_id,
        max_depth=3,  # Use a default depth of 3 to avoid too much crawling
        max_questions=20,  # Generate a reasonable number of questions
        domain=detected_domain.value  # Tag the dataset with the detected domain
    )
    
    try:
        # Create the dataset
        dataset_id = await create_dataset(dataset_request)
        
        # Return a response to the user
        return (
            f"I've started creating a {detected_domain.value.title()} domain dataset from {url}. "
            f"The dataset ID is `{dataset_id}`. "
            f"This process will run in the background and may take some time. "
            f"The dataset will contain questions and answers specifically tailored for the {detected_domain.value} domain. "
            f"You can check the progress later using this ID."
        )
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return f"I encountered an error while trying to create a dataset from {url}: {str(e)}"


async def handle_special_feature(query: str, llm: Optional[LanguageModelLike] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Handle special feature requests from the user.
    
    Args:
        query: The user's query
        llm: Optional language model for domain detection and processing
        user_id: Optional user ID for multi-user support
        
    Returns:
        Dict: Contains 'used_feature' flag, 'feature_type' and 'response'
    """
    # Detect the feature type
    feature_info = detect_feature_from_query(query)
    feature_type = feature_info["feature_type"]
    
    # If no special feature is detected, return empty result
    if feature_type == "none":
        return {
            "used_feature": False,
            "feature_type": None,
            "response": None
        }
    
    # Handle dynamic ingestion
    if feature_type == "dynamic_ingestion":
        response = await handle_dynamic_ingestion(query, llm, user_id)
        return {
            "used_feature": True,
            "feature_type": "dynamic_ingestion",
            "response": response
        }
    
    # Handle dataset creation
    if feature_type == "dataset_creation":
        response = await handle_dataset_creation(query, llm, user_id)
        return {
            "used_feature": True,
            "feature_type": "dataset_creation",
            "response": response
        }
    
    # For deep_research and auto_learn, we'll let the existing handlers take care of them
    # Just indicate that it's a feature request
    return {
        "used_feature": False,
        "feature_type": feature_type,
        "response": None
    }


def create_feature_integration_runnable() -> RunnableLambda:
    """Create a runnable that can be used to process special features in a chain.
    
    Returns:
        RunnableLambda: A runnable that checks and handles special features
    """
    async def process_features(inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process special features based on inputs and configuration.
        
        Args:
            inputs: The inputs including the user's query
            config: Optional configuration including an LLM
            
        Returns:
            Dict: The processing results
        """
        query = inputs.get("question", "")
        user_id = config.get("configurable", {}).get("user_id") if config else None
        
        # Get the LLM from config if available
        llm = None
        if config and "llm" in config:
            llm = config["llm"]
            
        return await handle_special_feature(query, llm, user_id)
        
    return RunnableLambda(process_features)