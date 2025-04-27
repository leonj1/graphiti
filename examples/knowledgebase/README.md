# Graphiti Knowledge Base Example

This example demonstrates how to use Graphiti to create a knowledge base from text data. It includes scripts for ingesting data, querying the knowledge base, and reingesting data when the source changes.

## Prerequisites

- Neo4j database running (default: bolt://10.1.1.144:7687)
- Python 3.8+
- Required Python packages (install via `pip install -r requirements.txt`)

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
NEO4J_URI=bolt://10.1.1.144:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Adjust the values as needed for your Neo4j instance.

## Scripts

### 1. Initial Data Ingestion

The `ingest.py` script parses a text file into chunks and ingests them into the Graphiti knowledge graph.

```bash
python ingest.py
```

This script:
- Clears any existing data in the graph database
- Parses the text file (examples/wizard_of_oz/woo.txt) into manageable chunks
- Ingests each chunk as an episode in the knowledge graph

### 2. Querying the Knowledge Base

The `query.py` script allows you to search the knowledge base using natural language queries.

```bash
# Run in interactive mode
python query.py

# Or provide a query directly
python query.py "Who is Dorothy?"
```

Features:
- Natural language queries
- Hybrid search combining semantic similarity and BM25 retrieval
- Option to use a center node for reranking results based on graph distance
- Interactive mode with customizable result limits

### 3. Reingesting Changed Data

The `reingest.py` script handles updates to the source data by only ingesting new or changed chunks.

```bash
python reingest.py
```

This script:
- Parses the text file into chunks
- Compares chunks with previously ingested data using MD5 hashes
- Only ingests chunks that are new or have changed
- Maintains a cache to track which chunks have been ingested

## How It Works

1. **Text Chunking**: The source text is split into paragraphs and then grouped into chunks of approximately 1000 characters each.

2. **Knowledge Graph**: Graphiti processes these chunks to extract entities and relationships, building a knowledge graph.

3. **Querying**: When you query the knowledge base, Graphiti performs a hybrid search combining semantic similarity and BM25 text retrieval to find the most relevant information.

4. **Reingestion**: When the source text changes, only the modified portions are reingested, preserving the existing knowledge graph structure.

## Customization

You can modify these scripts to work with your own text data:

1. Change the file path in `main()` functions to point to your data source
2. Adjust the chunking strategy in `parse_text_file()` if needed
3. Customize the query interface in `query.py` to suit your needs

## Example Usage

```python
# Example of programmatically querying the knowledge base
import asyncio
from query import query_knowledge_base

async def example():
    await query_knowledge_base("What happens to the Tin Woodman?", limit=3)

asyncio.run(example())
