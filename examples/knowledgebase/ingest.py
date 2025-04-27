"""
Copyright 2025, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data


def setup_logging():
    """Configure logging for the application."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


def parse_text_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a text file into chunks for ingestion.
    
    Args:
        file_path: Path to the text file to parse
        
    Returns:
        List of dictionaries containing parsed chunks
    """
    print(f"Reading file: {file_path}")
    with open(file_path, encoding='utf-8') as file:
        content = file.read()
    
    print("Splitting content into paragraphs...")
    # Split content into paragraphs
    paragraphs = [p for p in content.split('\n\n') if p.strip()]
    
    print(f"Processing {len(paragraphs)} paragraphs into chunks...")
    # Group paragraphs into chunks of reasonable size (about 1000 characters)
    chunks = []
    current_chunk = ""
    chunk_number = 1
    
    # Add progress bar for paragraph processing
    for paragraph in tqdm(paragraphs, desc="Chunking text", unit="paragraph"):
        if len(current_chunk) + len(paragraph) > 1000 and current_chunk:
            chunks.append({
                "chunk_number": chunk_number,
                "content": current_chunk.strip()
            })
            chunk_number += 1
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append({
            "chunk_number": chunk_number,
            "content": current_chunk.strip()
        })
    
    print(f"Created {len(chunks)} chunks for ingestion")
    return chunks


async def ingest_text_file(file_path: str, clear_existing: bool = False) -> None:
    """
    Ingest a text file into the Graphiti knowledge graph.
    
    Args:
        file_path: Path to the text file to ingest
        clear_existing: Whether to clear existing data before ingestion
    """
    logger = setup_logging()
    logger.info(f"Starting ingestion of {file_path}")
    
    # Load environment variables
    load_dotenv()
    
    # Neo4j connection parameters
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://10.1.1.144:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    
    if not neo4j_uri or not neo4j_user or not neo4j_password:
        raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')
    
    # Initialize Graphiti
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing data...")
            await clear_data(graphiti.driver)
        
        # Initialize indices and constraints
        await graphiti.build_indices_and_constraints()
        
        # Parse the text file
        chunks = parse_text_file(file_path)
        logger.info(f"Parsed {len(chunks)} chunks from {file_path}")
        
        # Get file name for source description
        file_name = os.path.basename(file_path)
        
        # Add episodes to the graph
        now = datetime.now(timezone.utc)
        
        print(f"\nIngesting {len(chunks)} chunks into the knowledge graph...")
        # Use tqdm to create a progress bar for the ingestion process
        for i, chunk in enumerate(tqdm(chunks, desc="Ingesting chunks", unit="chunk")):
            await graphiti.add_episode(
                name=f"Chunk {chunk['chunk_number']}",
                episode_body=chunk['content'],
                source=EpisodeType.text,
                source_description=f"Ingested from {file_name}",
                reference_time=now + timedelta(seconds=i * 10),
            )
            # Don't log every chunk to avoid cluttering the console with the progress bar
            # Instead, we'll use the progress bar to show progress
        
        print(f"\n✅ Successfully ingested {len(chunks)} chunks from {file_path}")
    
    finally:
        # Close the connection
        await graphiti.close()
        logger.info("Connection closed")


async def main():
    """Main function to run the ingestion process."""
    print("\n=== Graphiti Knowledge Base Ingestion Tool ===\n")
    
    # Path to the Wizard of Oz text file
    woo_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "wizard_of_oz", "woo.txt")
    
    print(f"Starting ingestion process for: {woo_file_path}")
    print("This may take several minutes depending on the file size and Neo4j performance.")
    print("Progress bars will show you the status of each step.\n")
    
    # Create a progress bar for the overall process
    with tqdm(total=3, desc="Overall progress", unit="step") as pbar:
        pbar.set_description("Step 1: Clearing existing data")
        # Ingest the text file, clearing existing data
        await ingest_text_file(woo_file_path, clear_existing=True)
        pbar.update(3)  # Complete all steps
    
    print("\n=== Ingestion Complete ===")
    print("You can now query the knowledge base using the query.py script.")
    print("Example: python query.py \"Who is Dorothy?\"")


if __name__ == "__main__":
    asyncio.run(main())
