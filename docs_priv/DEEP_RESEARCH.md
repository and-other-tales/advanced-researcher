# Deep Research Feature

This feature integrates the open_deep_research framework to provide comprehensive, well-structured research reports on complex topics. Unlike standard RAG which provides short answers to specific questions, deep research conducts in-depth investigation with multiple search iterations and structured report generation.

## Prerequisites

To use the deep research feature, you need to install the open_deep_research package:

```bash
pip install open-deep-research
```

You will also need API keys for at least one search provider. The recommended provider is Tavily:

```bash
export TAVILY_API_KEY=your_api_key_here
```

## API Endpoints

### Start Research

```
POST /api/research
```

Initiates a deep research process on a specified topic.

Request body:
```json
{
  "topic": "The impact of quantum computing on cryptography",
  "knowledge_base_id": "optional_custom_kb_id",
  "max_depth": 2,
  "number_of_queries": 2,
  "search_api": "tavily"
}
```

Parameters:
- `topic`: The research question or topic to investigate
- `knowledge_base_id` (optional): A custom knowledge base to use alongside web search
- `max_depth`: Maximum number of search iterations (higher values provide more thorough research but take longer)
- `number_of_queries`: Number of search queries per iteration
- `search_api`: Search provider to use (options: "tavily", "perplexity", "exa", "arxiv", "pubmed", "linkup")

Response:
```json
{
  "research_id": "research_1234abcd",
  "status": "started",
  "message": "Deep research task started successfully"
}
```

### Check Research Status

```
GET /api/research/{research_id}
```

Retrieves the current status and results of a research task.

Response:
```json
{
  "research_id": "research_1234abcd",
  "status": "completed",
  "progress": 1.0,
  "final_report": "# Research Report on Quantum Computing and Cryptography\n\n## Introduction\n..."
}
```

Status values:
- `pending`: Research task is queued
- `running`: Research is in progress
- `completed`: Research is complete with results available
- `failed`: Research encountered an error

### Check Query Complexity

```
POST /api/research/check_complexity
```

Analyzes a query to determine if it requires deep research rather than standard RAG.

Request body:
```json
{
  "question": "What are the implications of quantum computing on modern cryptography?"
}
```

Response:
```json
{
  "is_complex": true,
  "recommended_approach": "deep_research"
}
```

## Using Deep Research in the Chat Interface

The deep research capability can be triggered in several ways:

1. **Automatic Detection**: Complex queries are automatically detected and routed to deep research
2. **Explicit Request**: User can explicitly request research with phrases like "research" or "in-depth analysis"
3. **Follow-up to Standard Answers**: When a standard RAG answer is incomplete, deep research can be offered

Example user queries that trigger deep research:
- "Research the impact of AI on job markets"
- "Give me an in-depth analysis of blockchain technology"
- "Explain the advantages and disadvantages of nuclear energy"

## Research Process

The deep research process follows these steps:

1. **Planning Phase**: Analyzes the topic and generates a structured report plan
2. **Research Phase**: Iteratively searches for information section by section
3. **Writing Phase**: Compiles searched information into coherent section content
4. **Compilation**: Assembles all sections into a final comprehensive report

This process typically takes 1-5 minutes depending on topic complexity and configuration.

## Integration with Custom Knowledge Bases

Deep research can leverage custom knowledge bases created with the dynamic ingestion feature:

1. Create a knowledge base using the dynamic ingestion feature
2. Start research with the knowledge base ID specified
3. The research will combine web search with retrievals from your custom knowledge base

## Limitations

- Deep research takes significantly longer than standard RAG responses
- Requires more API calls and resources
- Not suitable for simple factual questions that can be answered directly
- Integration with multi-agent implementation is experimental and may have limitations

## Examples

### Example 1: Simple Question (standard RAG)

User: "What is the capital of France?"

This query is simple and factual, so it will be routed to standard RAG for a quick response.

### Example 2: Complex Question (deep research)

User: "Analyze the potential impact of quantum computing on modern cryptography systems"

This query will be routed to deep research, which will conduct multiple search iterations and produce a structured report with sections covering various aspects of the topic.

### Example 3: Explicit Research Request

User: "I need an in-depth research report on the ethical considerations of AI in healthcare"

This explicit request for research will trigger the deep research process, resulting in a comprehensive report with introduction, body sections, and conclusion.