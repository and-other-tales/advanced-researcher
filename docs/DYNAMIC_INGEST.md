# Dynamic Document Ingestion

This feature allows you to dynamically ingest documents from various web sources, including GOV.UK pages, and use them as knowledge bases for question answering.

## API Endpoints

The following API endpoints are available for managing knowledge bases:

### List Knowledge Bases

```
GET /api/knowledge_bases
```

Returns a list of available knowledge bases.

Optional query parameters:
- `user_id`: Filter knowledge bases by user ID

### Get Knowledge Base

```
GET /api/knowledge_bases/{kb_id}
```

Returns details of a specific knowledge base.

### Create Knowledge Base

```
POST /api/knowledge_bases
```

Creates a new knowledge base by ingesting documents from a web source.

Request body:
```json
{
  "name": "HMRC Tax Guidance",
  "source_type": "gov_uk",
  "url": "https://www.gov.uk/hmrc/internal-manuals",
  "max_depth": 8,
  "user_id": "optional-user-id"
}
```

Source types:
- `gov_uk`: Special handling for GOV.UK pages (including HMRC manuals)
- `sitemap`: Ingest documents using a sitemap.xml URL
- `recursive_url`: Recursively crawl a website from a starting URL

### Delete Knowledge Base

```
DELETE /api/knowledge_bases/{kb_id}
```

Deletes a knowledge base.

## Using Knowledge Bases in Chat

When sending a chat request, you can specify a knowledge base to use for answering the question:

```json
{
  "question": "What is the tax treatment of dividends?",
  "knowledge_base_id": "hmrc_tax_guidance_1234abcd"
}
```

If no knowledge base is specified, the default LangChain documentation knowledge base will be used.

## Examples

### Creating an HMRC Tax Guidance Knowledge Base

Use the following curl command to create a new knowledge base for HMRC tax guidance:

```bash
curl -X POST http://localhost:8080/api/knowledge_bases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "HMRC Tax Guidance",
    "source_type": "gov_uk",
    "url": "https://www.gov.uk/hmrc/internal-manuals",
    "max_depth": 8
  }'
```

### Asking a Question Using the Knowledge Base

Once you have created a knowledge base, you can use it to answer questions:

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the tax treatment of dividends?",
    "knowledge_base_id": "hmrc_tax_guidance_1234abcd"
  }'
```

## Testing

You can use the included test script to test the document ingestion functionality:

```bash
python test_dynamic_ingest.py
```

This will create a test knowledge base for HMRC tax guidance and return its ID, which you can then use for chat requests.