"""Inline implementation of open_deep_research for comprehensive research-based responses."""
import logging
import os
import asyncio
import datetime
import json
import uuid
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_fireworks import ChatFireworks
import httpx

from pydantic import BaseModel, Field, field_validator

# Import verification components
from backend.verification import get_verification_components
from backend.utils.env import get_env

# Set ODR_AVAILABLE to True since we're implementing it inline
ODR_AVAILABLE = True

logger = logging.getLogger(__name__)

# Internal implementation of open_deep_research functionality
class SearchAPI(str, Enum):
    """Enum of available search APIs."""
    TAVILY = "tavily"
    PERPLEXITY = "perplexity"
    EXXAMIND = "exxamind"
    DUCK_DUCK_GO = "duckduckgo"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"

class SearchResult(BaseModel):
    """Result from a search API."""
    title: str = ""
    url: str = ""
    content: str = ""
    
class ResearchPlan(BaseModel):
    """Plan for conducting research on a topic."""
    topic: str
    sections: List[Dict[str, str]] = Field(default_factory=list)
    outline: str = ""
    
class ResearchSection(BaseModel):
    """A section of research with content and sources."""
    title: str
    content: str
    sources: List[SearchResult] = Field(default_factory=list)
    
class ResearchReport(BaseModel):
    """Final research report with all sections and sources."""
    topic: str
    introduction: str = ""
    sections: List[ResearchSection] = Field(default_factory=list)
    conclusion: str = ""
    final_answer: str = ""
    sources: List[SearchResult] = Field(default_factory=list)
    
class SearchQuery(BaseModel):
    """Search query for a specific topic or subtopic."""
    query: str
    section_title: Optional[str] = None

class ResearchDepthLevel(str, Enum):
    """Enum of available research depth levels."""
    GENERAL_REFERENCE = "general_reference"
    ANALYSIS_INSIGHT = "analysis_insight"
    ACADEMIC_RESEARCH = "academic_research"

class ResearchRequest(BaseModel):
    """Request to perform deep research on a topic."""
    topic: str
    knowledge_base_id: Optional[str] = None
    max_depth: int = 2
    number_of_queries: int = 2
    search_api: str = "tavily"
    depth_level: Optional[ResearchDepthLevel] = None
    min_documents: Optional[int] = None
    max_documents: Optional[int] = None
    min_sites: Optional[int] = None
    max_sites: Optional[int] = None
    
    @field_validator("min_documents", "max_documents", "min_sites", "max_sites", mode="before")
    def set_defaults_based_on_depth(cls, v, info):
        """Set default values based on the depth level if not provided."""
        if v is not None:
            return v
            
        depth = info.data.get("depth_level")
        if depth is None:
            return v
            
        field_name = info.field_name
        
        if depth == ResearchDepthLevel.GENERAL_REFERENCE:
            if field_name == "min_documents":
                return 20
            elif field_name == "max_documents":
                return 50
            elif field_name == "min_sites":
                return 10
            elif field_name == "max_sites":
                return 30
        elif depth == ResearchDepthLevel.ANALYSIS_INSIGHT:
            if field_name == "min_documents":
                return 50
            elif field_name == "max_documents":
                return 150
            elif field_name == "min_sites":
                return 30
            elif field_name == "max_sites":
                return 75
        elif depth == ResearchDepthLevel.ACADEMIC_RESEARCH:
            if field_name == "min_documents":
                return 100
            elif field_name == "max_documents":
                return 500
            elif field_name == "min_sites":
                return 50
            elif field_name == "max_sites":
                return 150
                
        return v

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


def detect_deep_research_request(question: str) -> bool:
    """Detect if a user is explicitly requesting deep research in their query.
    
    Args:
        question: The user's query or question
        
    Returns:
        bool: True if the query contains explicit deep research request phrases
    """
    # Check for explicit deep research phrases
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
    
    question_lower = question.lower()
    
    # Check for explicit deep research requests
    for phrase in deep_research_phrases:
        if phrase in question_lower:
            return True
    
    return False


# Implement search APIs
async def search_with_tavily(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search using Tavily API."""
    api_key = get_env("TAVILY_API_KEY")
    if not api_key:
        logger.error("Tavily API key not found")
        return []
        
    try:
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_domains": [],
            "exclude_domains": [],
            "include_answer": False,
            "include_raw_content": True
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for result in data.get("results", []):
                results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", "")
                ))
                
            return results
        else:
            logger.error(f"Tavily search failed with status code {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error searching with Tavily: {e}")
        return []


async def search_with_perplexity(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search using Perplexity API."""
    api_key = get_env("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error("Perplexity API key not found")
        return []
        
    try:
        url = "https://api.perplexity.ai/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "query": query,
            "max_results": max_results,
            "highlight": False
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for result in data.get("results", []):
                results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("extract", "")
                ))
                
            return results
        else:
            logger.error(f"Perplexity search failed with status code {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error searching with Perplexity: {e}")
        return []


async def search_with_duckduckgo(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in list(ddgs.text(query, max_results=max_results)):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    content=r.get("body", "")
                ))
                
        return results
    except Exception as e:
        logger.error(f"Error searching with DuckDuckGo: {e}")
        return []


async def get_search_api(search_api: str):
    """Get the appropriate search API function based on the name."""
    search_apis = {
        "tavily": search_with_tavily,
        "perplexity": search_with_perplexity,
        "duckduckgo": search_with_duckduckgo
    }
    
    return search_apis.get(search_api.lower(), search_with_tavily)


# Implementation of the research graph
class ODRGraph:
    """Implementation of open_deep_research graph functionality."""
    
    @staticmethod
    async def create_research_plan(topic: str, llm: LanguageModelLike) -> ResearchPlan:
        """Create a research plan with section breakdown."""
        planning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a research planning expert. Your task is to create a detailed plan for researching a given topic.
Follow these guidelines:
1. Generate 3-6 sections that break down the topic into logical components
2. For each section, provide a clear title and brief description of what to research
3. Create a structured outline of the final report 
4. Ensure coverage of relevant aspects, historical context, current state, future implications, etc.
5. Return the plan in a detailed, well-structured format

Your response should be comprehensive yet focused on the key aspects of the topic."""),
            HumanMessage(content="I need to research the following topic thoroughly: {topic}. Please create a detailed research plan.")
        ])
        
        # Parse the response into sections and outline
        def parse_planning_response(response: str) -> Dict:
            sections = []
            outline = response
            
            # Extract section titles and descriptions
            import re
            section_pattern = r"(?:Section|Chapter|Part)\s*\d+:\s*(.+?)(?:\n|:)(.*?)(?=(?:Section|Chapter|Part)\s*\d+:|$)"
            matches = re.findall(section_pattern, response, re.DOTALL)
            
            for title, description in matches:
                sections.append({
                    "title": title.strip(),
                    "description": description.strip()
                })
                
            # If no sections were found, try another pattern
            if not sections:
                section_pattern = r"(\d+\.\s*.+?)(?:\n|:)(.*?)(?=\d+\.\s*|$)"
                matches = re.findall(section_pattern, response, re.DOTALL)
                
                for title, description in matches:
                    sections.append({
                        "title": title.strip(),
                        "description": description.strip()
                    })
            
            return {
                "topic": topic,
                "sections": sections,
                "outline": outline
            }
        
        plan_chain = planning_prompt | llm | StrOutputParser() | RunnableLambda(parse_planning_response)
        result = await plan_chain.ainvoke({"topic": topic})
        
        return ResearchPlan(**result)
    
    @staticmethod
    async def generate_search_queries(topic: str, section: Dict[str, str], num_queries: int, llm: LanguageModelLike) -> List[str]:
        """Generate search queries for a specific section."""
        query_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a search query generation expert. Your task is to create effective search queries for researching a specific section of a larger topic.
Follow these guidelines:
1. Generate specific, targeted search queries that will yield high-quality information
2. Ensure diversity in the queries to capture different aspects 
3. Avoid overly general or vague queries
4. Include relevant technical terms, important figures, or concepts
5. Format your response as a numbered list of queries

Your queries should be designed to retrieve comprehensive information for the section."""),
            HumanMessage(content="""Main Topic: {topic}
Section Title: {section_title}
Section Description: {section_description}
Number of queries to generate: {num_queries}

Please generate {num_queries} effective search queries for researching this section.""")
        ])
        
        # Parse the response into a list of queries
        def parse_queries(response: str) -> List[str]:
            import re
            # Extract numbered queries
            queries = re.findall(r"\d+\.\s*(.+)", response)
            # If no numbered queries found, split by newlines
            if not queries:
                queries = [q.strip() for q in response.split("\n") if q.strip()]
            return queries[:num_queries]  # Limit to requested number
        
        query_chain = query_prompt | llm | StrOutputParser() | RunnableLambda(parse_queries)
        
        result = await query_chain.ainvoke({
            "topic": topic,
            "section_title": section.get("title", ""),
            "section_description": section.get("description", ""),
            "num_queries": num_queries
        })
        
        return result
    
    @staticmethod
    async def write_section_content(topic: str, section: Dict[str, str], search_results: List[SearchResult], llm: LanguageModelLike) -> str:
        """Write content for a section based on search results."""
        # Prepare search results for the prompt
        combined_results = "\n\n".join([
            f"Source {i+1}: {result.title}\nURL: {result.url}\n{result.content}"
            for i, result in enumerate(search_results)
        ])
        
        writing_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a research content writer. Your task is to write a comprehensive section for a research report based on provided search results.
Follow these guidelines:
1. Synthesize information from the provided sources into a coherent, well-structured section
2. Focus on the specific section topic within the broader research area
3. Include relevant facts, data, examples, and expert opinions from the sources
4. Organize the content logically with appropriate headings and paragraphs
5. Use an academic, informative tone
6. Ensure accuracy and avoid adding unsupported information
7. Write at least 300 words, but focus on quality and comprehensiveness

Your section should be a thorough treatment of the topic that could be included in a professional research report."""),
            HumanMessage(content="""Main Research Topic: {topic}
Section Title: {section_title}
Section Description: {section_description}

Here are the search results to use for writing this section:

{search_results}

Please write a comprehensive, well-structured section for the research report.""")
        ])
        
        writing_chain = writing_prompt | llm | StrOutputParser()
        
        result = await writing_chain.ainvoke({
            "topic": topic,
            "section_title": section.get("title", ""),
            "section_description": section.get("description", ""),
            "search_results": combined_results
        })
        
        return result
    
    @staticmethod
    async def compile_final_report(topic: str, sections: List[ResearchSection], llm: LanguageModelLike) -> ResearchReport:
        """Compile the final research report from all sections."""
        # Prepare sections for the prompt
        combined_sections = "\n\n".join([
            f"## {section.title}\n{section.content}"
            for section in sections
        ])
        
        compilation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a research compilation expert. Your task is to compile a comprehensive research report from provided sections, adding an introduction and conclusion.
Follow these guidelines:
1. Write an engaging introduction that frames the topic and outlines the report's structure
2. Incorporate all provided sections without significant changes
3. Write a conclusion that summarizes key insights and offers final thoughts
4. Ensure smooth transitions between sections
5. Maintain a consistent academic tone throughout
6. The report should be well-structured with clear headings and logical flow

Your final output should be a complete, cohesive research report that could be published or presented professionally."""),
            HumanMessage(content="""Research Topic: {topic}

Here are the sections for the research report:

{sections}

Please compile these into a complete research report with an added introduction and conclusion.""")
        ])
        
        # Parse the response into introduction, main content, and conclusion
        def parse_report(response: str) -> Dict:
            import re
            
            # Extract introduction (everything before first section heading)
            intro_pattern = r"^(.*?)(?=##\s|#\s)"
            intro_match = re.search(intro_pattern, response, re.DOTALL)
            introduction = intro_match.group(1).strip() if intro_match else ""
            
            # Extract conclusion (everything after "Conclusion" heading)
            conclusion_pattern = r"(?:##\s*|#\s*)(?:Conclusion|Summary).*?\n(.*?)$"
            conclusion_match = re.search(conclusion_pattern, response, re.DOTALL)
            conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
            
            # If no conclusion found with heading, try taking the last paragraph
            if not conclusion:
                paragraphs = response.split("\n\n")
                if paragraphs:
                    conclusion = paragraphs[-1].strip()
            
            # Get all sources from the sections
            all_sources = []
            for section in sections:
                all_sources.extend(section.sources)
                
            # Remove duplicates based on URL
            unique_sources = []
            urls = set()
            for source in all_sources:
                if source.url not in urls:
                    urls.add(source.url)
                    unique_sources.append(source)
            
            return {
                "topic": topic,
                "introduction": introduction,
                "sections": sections,
                "conclusion": conclusion,
                "final_answer": response,
                "sources": unique_sources
            }
        
        compilation_chain = compilation_prompt | llm | StrOutputParser() | RunnableLambda(parse_report)
        
        result = await compilation_chain.ainvoke({
            "topic": topic,
            "sections": combined_sections
        })
        
        return ResearchReport(**result)
    
    async def ainvoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the research workflow."""
        topic = inputs.get("topic", "")
        if not topic:
            raise ValueError("Research topic is required")
            
        # Get configuration
        max_depth = config.get("configurable", {}).get("max_search_depth", 2)
        num_queries = config.get("configurable", {}).get("number_of_queries", 2)
        search_api_name = config.get("configurable", {}).get("search_api", "tavily")
        
        # Get research depth parameters
        depth_level = config.get("configurable", {}).get("depth_level")
        min_documents = config.get("configurable", {}).get("min_documents")
        max_documents = config.get("configurable", {}).get("max_documents")
        min_sites = config.get("configurable", {}).get("min_sites")
        max_sites = config.get("configurable", {}).get("max_sites")
        
        # Adjust depth and queries based on research depth level
        if depth_level == "academic_research":
            max_depth = max(max_depth, 4)  # Academic research needs deeper exploration
            num_queries = max(num_queries, 4)  # More queries for comprehensive results
        elif depth_level == "analysis_insight":
            max_depth = max(max_depth, 3)  # More depth for analysis
            num_queries = max(num_queries, 3)  # More queries for better insights
        elif depth_level == "general_reference":
            max_depth = max(max_depth, 2)  # Standard depth for general reference
            num_queries = max(num_queries, 2)  # Standard number of queries
            
        # Log the configuration being used
        logger.info(f"Deep research config: depth_level={depth_level}, max_depth={max_depth}, " 
                    f"num_queries={num_queries}, min_docs={min_documents}, max_docs={max_documents}, "
                    f"min_sites={min_sites}, max_sites={max_sites}")
        
        # Get model configuration
        planner_provider = config.get("configurable", {}).get("planner_provider", "anthropic")
        planner_model = config.get("configurable", {}).get("planner_model", "claude-3-7-sonnet-latest")
        writer_provider = config.get("configurable", {}).get("writer_provider", "anthropic")
        writer_model = config.get("configurable", {}).get("writer_model", "claude-3-5-sonnet-latest")
        
        # Initialize LLMs
        planner_llm = self._get_llm(planner_provider, planner_model)
        writer_llm = self._get_llm(writer_provider, writer_model)
        
        # Get search API
        search_api = await get_search_api(search_api_name)
        
        # 1. Create research plan
        research_plan = await self.create_research_plan(topic, planner_llm)
        
        # 2. For each section, generate queries and perform searches
        sections = []
        for section_data in research_plan.sections:
            # Generate search queries for this section
            queries = await self.generate_search_queries(
                topic, 
                section_data, 
                num_queries, 
                planner_llm
            )
            
            # Execute searches
            all_results = []
            for query in queries:
                search_query = SearchQuery(query=query, section_title=section_data.get("title"))
                results = await search_api(query)
                all_results.extend(results)
            
            # Write section content
            section_content = await self.write_section_content(
                topic,
                section_data,
                all_results,
                writer_llm
            )
            
            # Create section object
            section = ResearchSection(
                title=section_data.get("title", ""),
                content=section_content,
                sources=all_results
            )
            
            sections.append(section)
        
        # 3. Compile final report
        final_report = await self.compile_final_report(topic, sections, writer_llm)
        
        # Return the result
        return {
            "topic": topic,
            "sections": [{"title": s.title, "content": s.content} for s in sections],
            "final_answer": final_report.final_answer,
            "sources": [{"url": s.url, "title": s.title, "content": s.content} for s in final_report.sources]
        }

    def _get_llm(self, provider: str, model: str) -> LanguageModelLike:
        """Get the appropriate LLM based on provider and model name."""
        if provider == "anthropic":
            api_key = get_env("ANTHROPIC_API_KEY")
            if api_key:
                return ChatAnthropic(model=model, anthropic_api_key=api_key)
        elif provider == "openai":
            api_key = get_env("OPENAI_API_KEY")
            if api_key:
                return ChatOpenAI(model=model, openai_api_key=api_key)
        elif provider == "google":
            api_key = get_env("GOOGLE_API_KEY")
            if api_key:
                return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        elif provider == "fireworks":
            api_key = get_env("FIREWORKS_API_KEY")
            if api_key:
                return ChatFireworks(model=model, fireworks_api_key=api_key)
                
        # Default to OpenAI if available
        openai_key = get_env("OPENAI_API_KEY")
        if openai_key:
            return ChatOpenAI(model="gpt-4o", openai_api_key=openai_key)
            
        # Try Anthropic as another fallback
        anthropic_key = get_env("ANTHROPIC_API_KEY")
        if anthropic_key:
            return ChatAnthropic(model="claude-3-opus-20240229", anthropic_api_key=anthropic_key)
            
        raise ValueError(f"No API key available for provider {provider} or fallback providers")


# Initialize the graph
odr_graph = ODRGraph()


class DeepResearchEngine:
    """Engine for performing deep research using inline odr implementation."""
    
    def __init__(self, default_models: Optional[Dict[str, str]] = None):
        """Initialize deep research engine with default models."""
        if not ODR_AVAILABLE:
            raise ImportError("Deep research features are not available.")
        
        self.default_models = default_models or {
            "planner_provider": "anthropic",
            "planner_model": "claude-3-7-sonnet-latest",
            "writer_provider": "anthropic",
            "writer_model": "claude-3-5-sonnet-latest"
        }
        
        # Default research depth configurations
        self.depth_configs = {
            ResearchDepthLevel.GENERAL_REFERENCE: {
                "min_documents": 20,
                "max_documents": 50,
                "min_sites": 10,
                "max_sites": 30,
                "max_depth": 2,
                "number_of_queries": 2
            },
            ResearchDepthLevel.ANALYSIS_INSIGHT: {
                "min_documents": 50,
                "max_documents": 150,
                "min_sites": 30,
                "max_sites": 75,
                "max_depth": 3,
                "number_of_queries": 3
            },
            ResearchDepthLevel.ACADEMIC_RESEARCH: {
                "min_documents": 100,
                "max_documents": 500,
                "min_sites": 50,
                "max_sites": 150,
                "max_depth": 4,
                "number_of_queries": 4
            }
        }
        
        # Initialize persistent storage for tracking research requests
        import json
        import os
        from pathlib import Path
        
        # Use the data mount path for persistent storage
        try:
            self.data_mount_path = os.environ.get("DATA_MOUNT_PATH", "/tmp/data")
            self.research_dir = Path(self.data_mount_path) / "deep_research"
            
            # Create directory if it doesn't exist
            os.makedirs(self.research_dir, exist_ok=True)
            print(f"Created research directory at {self.research_dir}")
        except Exception as e:
            print(f"Error creating research directory: {e}")
            # Use a fallback directory in /tmp
            self.data_mount_path = "/tmp"
            self.research_dir = Path(self.data_mount_path) / "deep_research"
            os.makedirs(self.research_dir, exist_ok=True)
            print(f"Using fallback research directory: {self.research_dir}")
        
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
                research_data["request"] = research_data["request"].model_dump()
            
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
        
        # Apply depth level configurations if specified
        if request.depth_level:
            depth_config = self.depth_configs.get(request.depth_level, {})
            
            # Only override if not explicitly provided in the request
            if request.min_documents is None:
                request.min_documents = depth_config.get("min_documents")
            if request.max_documents is None:
                request.max_documents = depth_config.get("max_documents")
            if request.min_sites is None:
                request.min_sites = depth_config.get("min_sites")
            if request.max_sites is None:
                request.max_sites = depth_config.get("max_sites")
            
            # Always use the higher value between request and depth config for max_depth and queries
            request.max_depth = max(request.max_depth, depth_config.get("max_depth", request.max_depth))
            request.number_of_queries = max(request.number_of_queries, depth_config.get("number_of_queries", request.number_of_queries))
        
        # Create configuration for open_deep_research
        config = {
            "configurable": {
                "number_of_queries": request.number_of_queries,
                "max_search_depth": request.max_depth,
                "search_api": request.search_api,
                "depth_level": request.depth_level.value if request.depth_level else None,
                "min_documents": request.min_documents,
                "max_documents": request.max_documents,
                "min_sites": request.min_sites,
                "max_sites": request.max_sites,
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
            
            # Run the research implementation
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
            response.final_report = research_data["result"].get("final_answer")
        
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