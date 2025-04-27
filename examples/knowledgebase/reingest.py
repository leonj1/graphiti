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
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType


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
    with open(file_path, encoding='utf-8') as file:
        content = file.read()
    
    # Split content into paragraphs
    paragraphs = [p for p in content.split('\n\n') if p.strip()]
    
    # Group paragraphs into chunks of reasonable size (about 1000 characters)
    chunks = []
    current_chunk = ""
    chunk_number = 1
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > 1000 and current_chunk:
            chunks.append({
                "chunk_number": chunk_number,
                "content": current_chunk.strip(),
                "hash": hashlib.md5(current_chunk.strip().encode()).hexdigest()
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
            "content": current_chunk.strip(),
            "hash": hashlib.md5(current_chunk.strip().encode()).hexdigest()
        })
    
    return chunks


def get_cache_file_path(file_path: str) -> str:
    """
    Get the path to the cache file for a given text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Path to the cache file
    """
    # Create a cache directory in the same directory as the script
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a cache file name based on the input file path
    file_name = os.path.basename(file_path)
    cache_file = os.path.join(cache_dir, f"{file_name}.cache.json")
    
    return cache_file


def load_cache(cache_file: str) -> Dict[str, Any]:
    """
    Load the cache file if it exists.
    
    Args:
        cache_file: Path to the cache file
        
    Returns:
        Dictionary containing cached data
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If there's an error reading the cache, return an empty cache
            return {"chunks": {}}
    
    return {"chunks": {}}


def save_cache(cache_file: str, cache_data: Dict[str, Any]) -> None:
    """
    Save data to the cache file.
    
    Args:
        cache_file: Path to the cache file
        cache_data: Dictionary containing data to cache
    """
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


async def reingest_text_file(file_path: str) -> None:
    """
    Reingest a text file into the Graphiti knowledge graph, only updating changed chunks.
    
    Args:
        file_path: Path to the text file to reingest
    """
    logger = setup_logging()
    logger.info(f"Starting reingestion of {file_path}")
    
    # Load environment variables
    load_dotenv()
    
    # Neo4j connection parameters
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://10.1.1.144:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    
    if not neo4j_uri or not neo4j_user or not neo4j_password:
        raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')
    
    # Parse the text file
    chunks = parse_text_file(file_path)
    logger.info(f"Parsed {len(chunks)} chunks from {file_path}")
    
    # Get cache file path and load cache
    cache_file = get_cache_file_path(file_path)
    cache = load_cache(cache_file)
    
    # Identify new or changed chunks
    new_or_changed_chunks = []
    for chunk in chunks:
        chunk_hash = chunk["hash"]
        if chunk_hash not in cache["chunks"]:
            new_or_changed_chunks.append(chunk)
            # Update cache with new chunk
            cache["chunks"][chunk_hash] = {
                "chunk_number": chunk["chunk_number"],
                "last_ingested": datetime.now().isoformat()
            }
    
    if not new_or_changed_chunks:
        logger.info("No changes detected in the file. Nothing to reingest.")
        return
    
    logger.info(f"Found {len(new_or_changed_chunks)} new or changed chunks to reingest")
    
    # Initialize Graphiti
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Initialize indices and constraints (this is idempotent)
        await graphiti.build_indices_and_constraints()
        
        # Get file name for source description
        file_name = os.path.basename(file_path)
        
        # Add episodes to the graph
        now = datetime.now(timezone.utc)
        for i, chunk in enumerate(new_or_changed_chunks):
            await graphiti.add_episode(
                name=f"Chunk {chunk['chunk_number']} (Updated)",
                episode_body=chunk['content'],
                source=EpisodeType.text,
                source_description=f"Reingested from {file_name}",
                reference_time=now + timedelta(seconds=i * 10),
            )
            logger.info(f"Added updated chunk {chunk['chunk_number']} to the graph")
        
        # Save updated cache
        save_cache(cache_file, cache)
        
        logger.info(f"Successfully reingested {len(new_or_changed_chunks)} chunks from {file_path}")
    
    finally:
        # Close the connection
        await graphiti.close()
        logger.info("Connection closed")


async def main():
    """Main function to run the reingestion process."""
    # Path to the Wizard of Oz text file
    woo_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "wizard_of_oz", "woo.txt")
    
    # Reingest the text file
    await reingest_text_file(woo_file_path)


if __name__ == "__main__":
    asyncio.run(main())
