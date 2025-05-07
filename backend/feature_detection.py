"""Detection utilities for advanced features based on user queries."""
import re
import logging
from typing import Literal, Optional, Dict, Any

logger = logging.getLogger(__name__)

FeatureType = Literal["deep_research", "dynamic_ingestion", "dataset_creation", "auto_learn", "none"]


def detect_dynamic_ingestion_phrases(query: str) -> bool:
    """Detect if a user query is requesting dynamic ingestion.
    
    Args:
        query: The user's question or request
        
    Returns:
        bool: True if the query contains dynamic ingestion phrases
    """
    dynamic_ingestion_phrases = [
        "ingest",
        "ingestion", 
        "add document",
        "add source",
        "add url",
        "add website",
        "add this url",
        "add this website",
        "add this source",
        "add the url",
        "add the website", 
        "add the source",
        "include document", 
        "include url",
        "include website",
        "import document",
        "import url",
        "import website",
        "load document",
        "load url", 
        "load website",
        "process url",
        "process website",
        "process document",
        "use this document",
        "use this url",
        "use this website",
        "upload document",
        "upload url",
        "learn from",
        "index document",
        "index url",
        "index website",
        "dynamic ingestion"
    ]
    
    query_lower = query.lower()
    
    # Check for explicit dynamic ingestion phrases
    for phrase in dynamic_ingestion_phrases:
        if phrase in query_lower:
            return True
    
    # Check for URL patterns - http(s)://... or www....
    url_pattern = r'https?://\S+|www\.\S+'
    if re.search(url_pattern, query_lower):
        return True
    
    return False


def detect_dataset_creation_phrases(query: str) -> bool:
    """Detect if a user query is requesting dataset creation.
    
    Args:
        query: The user's question or request
        
    Returns:
        bool: True if the query contains dataset creation phrases
    """
    dataset_creation_phrases = [
        "create dataset",
        "build dataset",
        "generate dataset",
        "make dataset",
        "create training data",
        "build training data", 
        "generate training data",
        "create test set",
        "build test set",
        "create eval",
        "create evaluation", 
        "create training",
        "create fine-tuning",
        "make training examples",
        "generate examples",
        "create examples",
        "build examples",
        "create question",
        "create questions and answers",
        "create q&a",
        "create qa pairs",
        "generate qa pairs",
        "generate questions and answers"
    ]
    
    query_lower = query.lower()
    
    # Check for explicit dataset creation phrases
    for phrase in dataset_creation_phrases:
        if phrase in query_lower:
            return True
            
    return False


def detect_auto_learn_phrases(query: str) -> bool:
    """Detect if a user query is explicitly requesting auto-learning.
    
    Args:
        query: The user's question or request
        
    Returns:
        bool: True if the query contains auto-learning phrases
    """
    auto_learn_phrases = [
        "auto learn",
        "auto-learn",
        "learn automatically",
        "learn and update",
        "update your knowledge",
        "enhance your knowledge",
        "expand your knowledge",
        "teach yourself",
        "learn more about",
        "learn on your own",
        "update your information",
        "keep learning"
    ]
    
    query_lower = query.lower()
    
    # Check for explicit auto-learning phrases
    for phrase in auto_learn_phrases:
        if phrase in query_lower:
            return True
            
    return False


def detect_deep_research_phrases(query: str) -> bool:
    """Detect if a user query is explicitly requesting deep research.
    
    Args:
        query: The user's question or request
        
    Returns:
        bool: True if the query contains deep research phrases
    """
    deep_research_phrases = [
        "deep research",
        "using deep research",
        "conduct deep research",
        "perform deep research",
        "thorough research",
        "comprehensive research",
        "extensive research",
        "detailed research",
        "in-depth analysis",
        "comprehensive analysis",
        "write a report",
        "research report",
        "detailed report",
        "academic report",
        "scholarly research",
        "scientific research"
    ]
    
    query_lower = query.lower()
    
    # Check for explicit deep research phrases
    for phrase in deep_research_phrases:
        if phrase in query_lower:
            return True
    
    return False


def detect_feature_from_query(query: str) -> Dict[str, Any]:
    """Detect which advanced feature is being requested in a query.
    
    Args:
        query: The user's question or request
        
    Returns:
        Dict: Contains 'feature_type' and other relevant information
    """
    result = {
        "feature_type": "none",
        "details": {}
    }
    
    # Check for dynamic ingestion first (highest priority)
    if detect_dynamic_ingestion_phrases(query):
        result["feature_type"] = "dynamic_ingestion"
        # Try to extract URLs
        urls = re.findall(r'https?://\S+|www\.\S+', query)
        if urls:
            result["details"]["urls"] = urls
        return result
        
    # Check for dataset creation second
    if detect_dataset_creation_phrases(query):
        result["feature_type"] = "dataset_creation"
        return result
        
    # Check for deep research third
    if detect_deep_research_phrases(query):
        result["feature_type"] = "deep_research"
        return result
        
    # Check for auto-learn last (lowest priority)
    if detect_auto_learn_phrases(query):
        result["feature_type"] = "auto_learn"
        return result
    
    return result