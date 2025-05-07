# Auto-Learning Feature

The auto-learning feature enables the application to automatically search for information that's missing from its knowledge base, crawl relevant web pages, and store the information for future use. This creates a "learn and remember" capability that improves the system's knowledge over time.

## How It Works

1. **Insufficient Information Detection**: When the system responds to a user query, it analyzes its own response to detect phrases that indicate a lack of information (e.g., "I don't have information about that", "I'm not sure").

2. **Automatic Web Search**: Upon detecting insufficient information, the system generates optimized search queries using the LLM and searches the web using various search APIs (Tavily, DuckDuckGo, Perplexity, etc.).

3. **Content Extraction and Processing**: The system fetches the content from search result URLs, processes it to extract relevant information, and splits it into appropriate chunks.

4. **Vector Database Storage**: The processed information is stored in the vector database (Weaviate or Chroma) with appropriate metadata, making it available for future queries.

5. **Transparent Notification**: The user is informed that the system is learning new information, with a message appended to the response.

## Benefits

- **Self-Improving Knowledge Base**: The system becomes more knowledgeable over time as it encounters and learns about new topics.
- **Enhanced Response Accuracy**: Future queries about previously unknown topics receive better answers.
- **Reduced Need for Manual Ingestion**: The system automatically identifies and fills knowledge gaps.
- **Transparency to Users**: Users are aware when the system is learning new information.

## Implementation Details

### Detection of Insufficient Information

The system uses a combination of pattern matching and context analysis to identify responses that indicate a lack of information:

```python
def detect_insufficient_information(query: str, response: str) -> bool:
    insufficient_indicators = [
        "I don't have information",
        "I don't know",
        "I'm not sure",
        # ... more phrases
    ]
    
    for indicator in insufficient_indicators:
        if indicator.lower() in response.lower():
            return True
    
    # Additional checks for short, non-informative responses
    # ...
```

### Search Query Generation

When insufficient information is detected, the system uses the LLM to generate effective search queries:

```python
def generate_search_queries(query: str, llm: LanguageModelLike) -> List[str]:
    system_message = "Your task is to generate 2-3 effective web search queries..."
    human_message = f"Generate search queries for: {query}"
    
    response = llm.invoke([system_message, human_message])
    queries = [q.strip() for q in response.content.strip().split('\n')]
    return queries
```

### Web Search Integration

The system is configured to try multiple search APIs in order of preference:

1. Tavily (requires API key)
2. DuckDuckGo (no API key required)
3. Perplexity (requires API key)
4. Exa (requires API key)

If all APIs fail or are unavailable, the system falls back to a basic DuckDuckGo search.

### Content Processing and Storage

After retrieving web content, the system:

1. Extracts text content from HTML
2. Splits content into appropriate chunks
3. Adds metadata (source URL, title, etc.)
4. Stores the processed documents in the vector store

### Background Processing

Learning happens asynchronously in the background, allowing the system to respond to the user immediately while gathering information:

```python
async def learn_from_query(query: str, llm: LanguageModelLike) -> LearningStatus:
    # Generate search queries
    search_queries = generate_search_queries(query, llm)
    
    # Search the web
    search_results = await search_web(search_queries)
    
    # Fetch and process URLs
    documents = await fetch_and_process_urls(search_results)
    
    # Ingest documents
    ingestion_stats = await ingest_documents(documents)
    
    # Return status
    return LearningStatus(...)
```

## API Endpoints

### Get Learning Status

```
GET /api/learning/{task_id}
```

Retrieves the status of a learning task:

Response:
```json
{
  "task_id": "learn_1234abcd",
  "status": "completed",
  "query": "What is quantum cryptography?",
  "documents_found": 5,
  "documents_added": 12,
  "start_time": 1683457290.123,
  "end_time": 1683457295.456,
  "error": null
}
```

Status values:
- `pending`: Learning task is queued
- `running`: Learning is in progress
- `completed`: Learning task completed successfully
- `failed`: Learning task encountered an error

## Limitations and Considerations

1. **Search API Dependencies**: The effectiveness of auto-learning depends on available search APIs and their quotas/rate limits.

2. **Content Quality**: The quality of learned information depends on the search results and content extraction.

3. **Privacy Considerations**: All web content is processed and stored, which may include information of varying quality and reliability.

4. **Storage Growth**: Over time, the vector database will grow as more information is learned, potentially requiring maintenance and pruning.

## Configuration

The auto-learning feature can be configured through environment variables:

```
# Search API keys
TAVILY_API_KEY=your_tavily_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
EXA_API_KEY=your_exa_api_key

# Vector database configuration
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
# Or for local deployment:
COLLECTION_NAME=your_chroma_collection_name
```

## Example Interaction

**User**: "What is the impact of quantum computing on blockchain security?"

**System**: "I don't have specific information about quantum computing's impact on blockchain security."  
*System detects insufficient information and triggers learning*  
"I'm searching for more information on this topic and will update my knowledge base."

*The system then searches, processes, and stores information about quantum computing and blockchain security*

**User** (later): "What is the impact of quantum computing on blockchain security?"

**System**: "Quantum computing poses significant threats to blockchain security through several mechanisms:

1. Shor's algorithm could potentially break the cryptographic algorithms (like ECDSA) that secure many blockchains by efficiently factoring large numbers
2. Most current blockchain implementations use cryptographic methods vulnerable to quantum attacks
3. Post-quantum cryptography solutions are being developed to make blockchains quantum-resistant

Many blockchain platforms are now working on implementing quantum-resistant algorithms to protect against these future threats."