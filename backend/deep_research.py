"""Integration with open_deep_research for comprehensive research-based responses."""
import logging
import os
import datetime
from typing import Dict, List, Optional, Union, Any

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

# Import verification components
from backend.verification import get_verification_components

# Import when open_deep_research is installed as a dependency
try:
    from open_deep_research.graph import graph as odr_graph
    from open_deep_research.configuration import Configuration as ODRConfiguration
    from open_deep_research.configuration import SearchAPI
    ODR_AVAILABLE = True
except ImportError:
    ODR_AVAILABLE = False
    logging.warning("open_deep_research is not installed. Deep research features will not be available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchRequest(BaseModel):
    """Request to perform deep research on a topic."""
    topic: str
    knowledge_base_id: Optional[str] = None
    max_depth: int = 2
    number_of_queries: int = 2
    search_api: str = "tavily"


class ResearchResponse(BaseModel):
    """Response containing research results."""
    research_id: str
    status: str
    final_report: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None


def check_if_complex_query(question: str) -> bool:
    """Determine if a question requires deep research rather than simple RAG."""
    # Check for phrases that indicate comprehensive research is needed
    research_indicators = [
        "comprehensive", "in-depth", "analyze", "report", "investigation", 
        "study", "research", "explain in detail", "pros and cons",
        "advantages and disadvantages", "compare", "contrast"
    ]
    
    # Check for complex question structures
    complex_prefixes = [
        "how does", "why does", "what are the implications of", 
        "what are the causes of", "what is the relationship between",
        "what factors contribute to", "how would", "in what ways"
    ]
    
    # Check for question length
    is_long_question = len(question.split()) > 15
    
    # Check if contains research indicators
    contains_indicators = any(indicator in question.lower() for indicator in research_indicators)
    
    # Check if starts with complex prefix
    has_complex_prefix = any(question.lower().startswith(prefix) for prefix in complex_prefixes)
    
    # Return True if any of the criteria are met
    return (is_long_question and (contains_indicators or has_complex_prefix))


class DeepResearchEngine:
    """Engine for performing deep research using open_deep_research."""
    
    def __init__(self, default_models: Optional[Dict[str, str]] = None):
        """Initialize deep research engine with default models."""
        if not ODR_AVAILABLE:
            raise ImportError("open_deep_research is not installed. Please install the package.")
        
        self.default_models = default_models or {
            "planner_provider": "anthropic",
            "planner_model": "claude-3-7-sonnet-latest",
            "writer_provider": "anthropic",
            "writer_model": "claude-3-5-sonnet-latest"
        }
        
        # Initialize persistent storage for tracking research requests
        import json
        import os
        from pathlib import Path
        
        # Use the data mount path for persistent storage
        self.data_mount_path = os.environ.get("DATA_MOUNT_PATH", "/data")
        self.research_dir = Path(self.data_mount_path) / "deep_research"
        
        # Create directory if it doesn't exist
        self.research_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the in-memory cache, which will be backed by files
        self.research_store = {}
        
        # Load any existing research data
        self._load_research_data()
        
    def _get_research_file_path(self, research_id: str) -> Path:
        """Get the file path for a research request."""
        return self.research_dir / f"{research_id}.json"
        
    def _save_research_data(self, research_id: str) -> None:
        """Save research data to disk."""
        try:
            if research_id not in self.research_store:
                logger.warning(f"Attempted to save non-existent research ID: {research_id}")
                return
                
            # Get the data to save, making a copy to avoid modifying the original
            research_data = dict(self.research_store[research_id])
            
            # Remove non-serializable objects
            if "task" in research_data:
                del research_data["task"]
            if "llm" in research_data:
                del research_data["llm"]
                
            # Convert request object to dict
            if "request" in research_data and isinstance(research_data["request"], BaseModel):
                research_data["request"] = research_data["request"].dict()
            
            # Save to JSON file
            file_path = self._get_research_file_path(research_id)
            with open(file_path, 'w') as f:
                json.dump(research_data, f)
                
            logger.debug(f"Saved research data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving research data for {research_id}: {e}")
            
    def _load_research_data(self) -> None:
        """Load all research data from disk."""
        try:
            # Find all JSON files in the research directory
            json_files = list(self.research_dir.glob("*.json"))
            
            for file_path in json_files:
                try:
                    # Get research ID from filename
                    research_id = file_path.stem
                    
                    # Load JSON data
                    with open(file_path, 'r') as f:
                        research_data = json.load(f)
                    
                    # Convert request dict back to object if needed
                    if "request" in research_data and isinstance(research_data["request"], dict):
                        research_data["request"] = ResearchRequest(**research_data["request"])
                    
                    # Add to in-memory store
                    self.research_store[research_id] = research_data
                    
                except Exception as e:
                    logger.error(f"Error loading research data from {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.research_store)} research requests from disk")
        except Exception as e:
            logger.error(f"Error loading research data: {e}")
            # Start with an empty store on error
            self.research_store = {}
    
    async def start_research(self, request: ResearchRequest, llm: Optional[LanguageModelLike] = None) -> str:
        """Start a deep research task.
        
        Args:
            request: The research request with topic and configuration
            llm: Optional language model for verifying research results
            
        Returns:
            research_id: A unique identifier for the research task
        """
        from uuid import uuid4
        
        # Generate a unique ID for this research task
        research_id = f"research_{uuid4().hex[:8]}"
        
        # Create configuration for open_deep_research
        config = {
            "configurable": {
                "number_of_queries": request.number_of_queries,
                "max_search_depth": request.max_depth,
                "search_api": request.search_api,
                "planner_provider": self.default_models["planner_provider"],
                "planner_model": self.default_models["planner_model"],
                "writer_provider": self.default_models["writer_provider"],
                "writer_model": self.default_models["writer_model"],
            }
        }
        
        # Store research request in memory and persistent storage
        self.research_store[research_id] = {
            "request": request,
            "config": config,
            "status": "pending",
            "progress": 0.0,
            "result": None,
            "error": None,
            "llm": llm,  # Store the LLM for later verification
            "verification_applied": False,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Save to persistent storage
        self._save_research_data(research_id)
        
        # Start research in background as a proper async task
        import asyncio
        task = asyncio.create_task(self._run_research(research_id))
        
        # Store the task reference for potential cancellation and monitoring
        self.research_store[research_id]["task"] = task
        
        # Setup proper error handling and monitoring
        task.add_done_callback(lambda t: self._handle_task_completion(research_id, t))
        
        return research_id
    
    async def _run_research(self, research_id: str) -> None:
        """Execute the deep research task.
        
        Args:
            research_id: The unique identifier for the research task
        """
        try:
            # Get the research request and configuration
            research_data = self.research_store[research_id]
            request = research_data["request"]
            config = research_data["config"]
            
            # Update status to running
            self.research_store[research_id]["status"] = "running"
            self.research_store[research_id]["started_at"] = datetime.datetime.now().isoformat()
            
            # Save status update
            self._save_research_data(research_id)
            
            # Run the open_deep_research graph
            result = await odr_graph.ainvoke(
                {"topic": request.topic},
                config=config
            )
            
            # Verify research results
            self.research_store[research_id]["status"] = "verifying"
            self.research_store[research_id]["progress"] = 0.8
            
            # Save status update
            self._save_research_data(research_id)
            
            # Get the LLM for verification
            llm = research_data.get("llm")
            
            if llm:
                # Apply verification to research results
                verified_result = await self._verify_research_results(
                    request.topic, 
                    result, 
                    llm
                )
                
                # Store the verified result
                self.research_store[research_id]["result"] = verified_result
                self.research_store[research_id]["verification_applied"] = True
                
                # Save updates
                self._save_research_data(research_id)
            else:
                # No LLM available for verification, use original results
                self.research_store[research_id]["result"] = result
                self.research_store[research_id]["verification_applied"] = False
                
                # Save updates
                self._save_research_data(research_id)
            
            # Set status to completed
            self.research_store[research_id]["status"] = "completed"
            self.research_store[research_id]["progress"] = 1.0
            self.research_store[research_id]["completed_at"] = datetime.datetime.now().isoformat()
            
            # Save final status
            self._save_research_data(research_id)
            
        except Exception as e:
            # Update status to failed
            self.research_store[research_id]["status"] = "failed"
            self.research_store[research_id]["error"] = str(e)
            self.research_store[research_id]["failed_at"] = datetime.datetime.now().isoformat()
            logger.error(f"Error running deep research: {e}")
            
            # Save error status
            self._save_research_data(research_id)
            
    async def _verify_research_results(self, topic: str, results: Dict[str, Any], llm: LanguageModelLike) -> Dict[str, Any]:
        """Verify and improve research results for factual accuracy.
        
        Args:
            topic: The research topic
            results: The research results to verify
            llm: The language model to use for verification
            
        Returns:
            The verified and potentially improved research results
        """
        # Get verification components
        verification = get_verification_components(llm)
        
        # Extract the main findings and content from the results
        main_content = results.get("final_answer", "")
        
        if not main_content:
            logger.warning("No main content found in research results")
            return results
            
        # Extract supporting documents/sources from results
        sources = results.get("sources", [])
        supporting_documents = []
        
        # Convert sources to Documents
        for i, source in enumerate(sources):
            if isinstance(source, str):
                # Simple string source
                doc = Document(
                    page_content=source,
                    metadata={"source": f"research_source_{i}", "title": f"Source {i}"}
                )
                supporting_documents.append(doc)
            elif isinstance(source, dict):
                # Dict source with content and metadata
                doc = Document(
                    page_content=source.get("content", ""),
                    metadata={
                        "source": source.get("url", f"research_source_{i}"),
                        "title": source.get("title", f"Source {i}")
                    }
                )
                supporting_documents.append(doc)
        
        # Check for hallucinations in the research results
        hallucination_result = verification.check_answer_hallucination(
            main_content, supporting_documents
        )
        
        # Check if the research addresses the original topic
        addressing_result = verification.check_answer_addresses_question(
            topic, main_content
        )
        
        # Determine if improvement is needed
        needs_improvement = hallucination_result["is_hallucination"] or not addressing_result["addresses_question"]
        
        if needs_improvement:
            # Create improved research results
            improved_content = verification.create_improved_answer(
                topic,
                main_content,
                supporting_documents,
                hallucination_result,
                addressing_result
            )
            
            # Update the results with the improved content
            results["final_answer"] = improved_content
            results["verification_applied"] = True
            
            # Add verification metadata
            results["verification_metadata"] = {
                "hallucination_check": {
                    "is_hallucination": hallucination_result["is_hallucination"],
                    "reason": hallucination_result["reason"]
                },
                "topic_addressing_check": {
                    "addresses_topic": addressing_result["addresses_question"],
                    "reason": addressing_result["reason"]
                }
            }
        else:
            # Results are good, just add verification metadata
            results["verification_applied"] = True
            results["verification_metadata"] = {
                "hallucination_check": {
                    "is_hallucination": False,
                    "reason": hallucination_result["reason"]
                },
                "topic_addressing_check": {
                    "addresses_topic": True,
                    "reason": addressing_result["reason"]
                }
            }
            
        return results
    
    def _handle_task_completion(self, research_id: str, task: asyncio.Task) -> None:
        """Handle the completion of a research task, including error cases.
        
        Args:
            research_id: The unique identifier for the research task
            task: The completed asyncio Task object
        """
        if research_id not in self.research_store:
            logger.error(f"Task completed for unknown research ID: {research_id}")
            return
            
        try:
            # Check if the task failed with an exception
            if task.exception():
                exc = task.exception()
                logger.error(f"Research task {research_id} failed with error: {exc}")
                
                # Update the research store with the error information
                self.research_store[research_id]["status"] = "failed"
                self.research_store[research_id]["error"] = str(exc)
                self.research_store[research_id]["progress"] = 0.0
                self.research_store[research_id]["failed_at"] = datetime.datetime.now().isoformat()
                
                # Clean up task reference
                self.research_store[research_id]["task"] = None
                
                # Save error status
                self._save_research_data(research_id)
            else:
                # Task completed successfully, but result already handled in _run_research
                logger.info(f"Research task {research_id} completed successfully")
                
                # Clean up task reference
                self.research_store[research_id]["task"] = None
                
                # Save final status (in case anything was missed)
                if self.research_store[research_id]["status"] == "completed":
                    self._save_research_data(research_id)
        except Exception as e:
            # Handle any errors in the error handling itself
            logger.error(f"Error in task completion handler for {research_id}: {e}")
            
            # Ensure research is marked as failed
            self.research_store[research_id]["status"] = "failed"
            self.research_store[research_id]["error"] = f"Error in completion handler: {str(e)}"
            self.research_store[research_id]["task"] = None
            self.research_store[research_id]["failed_at"] = datetime.datetime.now().isoformat()
            
            # Try to save error status
            try:
                self._save_research_data(research_id)
            except Exception as save_err:
                logger.error(f"Failed to save error status for {research_id}: {save_err}")
    
    def get_research_status(self, research_id: str) -> ResearchResponse:
        """Get the status of a deep research task.
        
        Args:
            research_id: The unique identifier for the research task
            
        Returns:
            ResearchResponse with current status
        """
        if research_id not in self.research_store:
            return ResearchResponse(
                research_id=research_id,
                status="not_found",
                error="Research task not found"
            )
        
        research_data = self.research_store[research_id]
        
        response = ResearchResponse(
            research_id=research_id,
            status=research_data["status"],
            progress=research_data["progress"],
            error=research_data.get("error")
        )
        
        # Include the final report if completed
        if research_data["status"] == "completed" and research_data["result"]:
            response.final_report = research_data["result"].get("final_report")
        
        return response


# Initialize the deep research engine if available
deep_research_engine = None
if ODR_AVAILABLE:
    try:
        deep_research_engine = DeepResearchEngine()
    except Exception as e:
        logger.error(f"Failed to initialize deep research engine: {e}")


async def research(request: ResearchRequest, llm: Optional[LanguageModelLike] = None) -> str:
    """Perform deep research on a topic.
    
    Args:
        request: The research request with topic and configuration
        llm: Optional language model for verifying research results
        
    Returns:
        str: The research ID for tracking the status
    """
    if not deep_research_engine:
        raise RuntimeError("Deep research engine is not available")
    
    return await deep_research_engine.start_research(request, llm)


def get_research(research_id: str) -> ResearchResponse:
    """Get the status and results of a deep research task.
    
    Args:
        research_id: The unique identifier for the research task
        
    Returns:
        ResearchResponse with current status and results if available
    """
    if not deep_research_engine:
        raise RuntimeError("Deep research engine is not available")
    
    return deep_research_engine.get_research_status(research_id)