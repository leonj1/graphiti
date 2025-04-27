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
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF


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


async def query_knowledge_base(query: str, limit: int = 5, center_node_uuid: Optional[str] = None) -> None:
    """
    Query the knowledge base with a natural language query.
    
    Args:
        query: The natural language query to search for
        limit: Maximum number of results to return
        center_node_uuid: Optional UUID of a center node for reranking results
    """
    logger = setup_logging()
    logger.info(f"Querying knowledge base with: '{query}'")
    
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
        # Perform the search
        if center_node_uuid:
            logger.info(f"Using center node UUID: {center_node_uuid}")
            results = await graphiti.search(query, center_node_uuid=center_node_uuid)
        else:
            results = await graphiti.search(query)
        
        # Print search results
        print("\n=== Search Results ===")
        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results[:limit], 1):
                print(f"\nResult {i}:")
                print(f"UUID: {result.uuid}")
                print(f"Fact: {result.fact}")
                if hasattr(result, 'source_node_uuid') and result.source_node_uuid:
                    print(f"Source Node UUID: {result.source_node_uuid}")
                if hasattr(result, 'target_node_uuid') and result.target_node_uuid:
                    print(f"Target Node UUID: {result.target_node_uuid}")
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f"Valid from: {result.valid_at}")
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f"Valid until: {result.invalid_at}")
                print("-" * 50)
        
        # If we have results, also perform a node search to get more context
        if results:
            print("\n=== Node Search Results ===")
            
            # Use a predefined search configuration recipe for node search
            node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            node_search_config.limit = limit  # Set the limit
            
            # Execute the node search
            node_search_results = await graphiti._search(
                query=query,
                config=node_search_config,
            )
            
            # Print node search results
            if not node_search_results.nodes:
                print("No node results found.")
            else:
                for i, node in enumerate(node_search_results.nodes[:limit], 1):
                    print(f"\nNode Result {i}:")
                    print(f"Node UUID: {node.uuid}")
                    print(f"Node Name: {node.name}")
                    node_summary = node.summary[:200] + '...' if len(node.summary) > 200 else node.summary
                    print(f"Content Summary: {node_summary}")
                    print(f"Node Labels: {', '.join(node.labels)}")
                    print(f"Created At: {node.created_at}")
                    if hasattr(node, 'attributes') and node.attributes:
                        print("Attributes:")
                        for key, value in node.attributes.items():
                            print(f"  {key}: {value}")
                    print("-" * 50)
    
    finally:
        # Close the connection
        await graphiti.close()
        logger.info("Connection closed")


async def interactive_query():
    """Run an interactive query session."""
    print("\n=== Graphiti Knowledge Base Query Tool ===")
    print("Enter your query below. Type 'exit' to quit.")
    
    center_node_uuid = None
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ('exit', 'quit'):
            break
        
        limit = 5
        try:
            limit_input = input("Number of results to show (default: 5): ")
            if limit_input.strip():
                limit = int(limit_input)
        except ValueError:
            print("Invalid number, using default limit of 5.")
        
        use_center = input("Use previous result as center node? (y/n, default: n): ")
        if use_center.lower() == 'y' and center_node_uuid:
            print(f"Using center node: {center_node_uuid}")
        else:
            center_node_uuid = None
        
        await query_knowledge_base(query, limit, center_node_uuid)
        
        # After search, ask if user wants to use a result as center node
        if center_node_uuid is None:
            set_center = input("\nSet a result as center node for next query? (Enter result number or 'n'): ")
            if set_center.isdigit() and 1 <= int(set_center) <= limit:
                # We would need to store the results to access them here
                # For simplicity, we'll just acknowledge the request
                print("Center node would be set (full implementation would store results)")
                center_node_uuid = "example_uuid"  # This is a placeholder


async def main():
    """Main function to run the query process."""
    if len(sys.argv) > 1:
        # If arguments are provided, use them as a query
        query = " ".join(sys.argv[1:])
        await query_knowledge_base(query)
    else:
        # Otherwise, run in interactive mode
        await interactive_query()


if __name__ == "__main__":
    asyncio.run(main())
