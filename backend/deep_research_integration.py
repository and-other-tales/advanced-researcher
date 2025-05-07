"""Integration between deep research and the main chat chain."""
import asyncio
import logging
from typing import Dict, Any, Optional

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableLambda

from backend.deep_research import (
    ResearchRequest, 
    ResearchDepthLevel, 
    deep_research_engine, 
    detect_deep_research_request,
    check_if_complex_query
)

logger = logging.getLogger(__name__)


async def perform_deep_research(question: str, config: Dict[str, Any] = None) -> str:
    """Perform deep research on a complex or explicitly requested topic.
    
    Args:
        question: The user's question
        config: Optional configuration for deep research
        
    Returns:
        str: The formatted research response
    """
    if not deep_research_engine:
        return "Deep research capabilities are not available. Using standard retrieval instead."
    
    # Extract deep research configuration
    configurable = config.get("configurable", {})
    
    # Set up research request with the provided configuration
    request = ResearchRequest(
        topic=question,
        max_depth=configurable.get("max_search_depth", 2),
        number_of_queries=configurable.get("number_of_queries", 2),
        depth_level=ResearchDepthLevel(configurable.get("research_depth_level", "general_reference")),
        min_documents=configurable.get("min_documents"),
        max_documents=configurable.get("max_documents"),
        min_sites=configurable.get("min_sites"),
        max_sites=configurable.get("max_sites"),
        search_api=configurable.get("search_api", "tavily")
    )
    
    try:
        # Start the research task
        research_id = await deep_research_engine.start_research(request)
        
        # Check research status (wait for completion with timeout)
        max_wait_time = 60  # seconds
        wait_interval = 2  # seconds
        total_waited = 0
        
        while total_waited < max_wait_time:
            await asyncio.sleep(wait_interval)
            total_waited += wait_interval
            
            research_status = deep_research_engine.get_research_status(research_id)
            if research_status.status == "completed":
                # Format and return the final report
                if research_status.final_report:
                    # Add a header to indicate deep research was used
                    research_depth = configurable.get("research_depth_level", "general_reference")
                    header = f"# Deep Research Report\n\n*Research depth: {research_depth.replace('_', ' ').title()}*\n\n"
                    return header + research_status.final_report
                else:
                    return "Deep research completed but no results were found."
            elif research_status.status == "failed":
                # If research failed, return the error
                return f"Deep research failed: {research_status.error or 'Unknown error'}\n\nFalling back to standard retrieval."
            
            # If still running and about to time out, return a progress message
            if total_waited + wait_interval >= max_wait_time and research_status.status == "running":
                # Research is still running, but we need to return something
                return (
                    f"Deep research is still in progress (current progress: {research_status.progress*100:.0f}%). "
                    f"This is taking longer than expected. I'll provide a preliminary answer based on available information, "
                    f"but you can check research ID `{research_id}` later for complete results."
                )
        
        # If we reach here, research timed out
        return "Deep research is taking longer than expected. Using standard retrieval instead."
        
    except Exception as e:
        logger.error(f"Error performing deep research: {e}")
        return f"Error performing deep research: {str(e)}. Using standard retrieval instead."


async def check_for_deep_research(inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Check if deep research should be used for the given inputs.
    
    Args:
        inputs: The input data including the user's question
        config: Optional configuration for deep research
        
    Returns:
        Dict: The updated inputs, possibly with deep research results
    """
    question = inputs.get("question", "")
    
    # Check if deep research is explicitly requested or query is complex
    is_explicit_request = detect_deep_research_request(question)
    is_complex = check_if_complex_query(question)
    
    logger.info(f"Question '{question[:50]}...' deep research check: explicit={is_explicit_request}, complex={is_complex}")
    
    # Only do deep research if explicitly requested or query is complex
    if is_explicit_request or is_complex:
        try:
            # Perform deep research
            logger.info(f"Starting deep research for question: '{question[:50]}...'")
            research_result = await perform_deep_research(question, config)
            
            # Return the research result with original inputs for later processing
            return {**inputs, "response": research_result, "used_deep_research": True}
        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            # Fall back to standard retrieval if deep research fails
            return inputs
    
    # Otherwise, just return the inputs unchanged
    return inputs


def create_deep_research_runnable() -> RunnableLambda:
    """Create a runnable that can be used to process deep research in a chain.
    
    Returns:
        RunnableLambda: A runnable that checks and performs deep research
    """
    return RunnableLambda(check_for_deep_research)