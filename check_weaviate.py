"""
Improved diagnostic script for Weaviate connection - handles low-level coroutines correctly.
"""

import os
import sys
import logging
import asyncio
import json
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import weaviate
try:
    import weaviate
    from weaviate.classes.query import Filter, MetadataQuery
except ImportError:
    logger.error("Weaviate client not installed. Run: pip install weaviate-client>=4.0.0")
    sys.exit(1)

# Run an async coroutine
def run_coro(coro):
    """Run a coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)

async def get_schema(client):
    """Get schema using the low-level connection."""
    try:
        # This is a coroutine in all versions
        schema = await client._connection.get("/v1/schema")
        return schema
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        return {"classes": []}

def check_weaviate_connection(url: str = "http://localhost:8080"):
    """Check if Weaviate is accessible and responsive."""
    logger.info(f"Checking connection to Weaviate at {url}")
    
    try:
        # Connect to Weaviate
        if url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
            client = weaviate.connect_to_local()
        else:
            client = weaviate.connect_to_wcs(
                cluster_url=url,
                auth_credentials=None  # Add auth if needed
            )
        
        # Check connection - not a coroutine in v4
        meta = client.get_meta()
        logger.info(f"Connected to Weaviate version: {meta}")
        
        # Get schema - use run_coro to handle the low-level coroutine
        schema = run_coro(get_schema(client))
        
        if isinstance(schema, dict) and "classes" in schema:
            collections = [cls["class"] for cls in schema.get("classes", [])]
            logger.info(f"Found {len(collections)} collections: {collections}")
        else:
            logger.warning(f"Unexpected schema response format: {type(schema)}")
            collections = []
            
            # Try alternate method to get collections
            try:
                collection_list = client.collections.list_all()
                if isinstance(collection_list, list):
                    collections = [c.name for c in collection_list]
                    logger.info(f"Found {len(collections)} collections using alternate method: {collections}")
            except Exception as alt_e:
                logger.warning(f"Error getting collections via alternate method: {alt_e}")
        
        return client, collections
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, []

async def get_objects(collection, limit=3):
    """Get sample objects - handle as async in case it's a coroutine."""
    try:
        # Check if the method is a coroutine by inspecting
        fetch_objects = collection.query.fetch_objects
        if asyncio.iscoroutinefunction(fetch_objects):
            # It's a coroutine function
            return await fetch_objects(limit=limit, include_vector=True)
        else:
            # It's a regular function
            return fetch_objects(limit=limit, include_vector=True)
    except Exception as e:
        logger.error(f"Error fetching objects: {e}")
        return None

def check_unique_strings_collection(client):
    """Check the UniqueStrings collection."""
    logger.info("Checking UniqueStrings collection")
    
    try:
        collection = client.collections.get("UniqueStrings")
        
        # Count objects
        try:
            count_result = collection.aggregate.over_all().count()
            total_objects = count_result.total_count if hasattr(count_result, 'total_count') else 0
            logger.info(f"UniqueStrings collection has {total_objects} objects")
        except Exception as count_e:
            logger.warning(f"Error counting objects: {count_e}")
            total_objects = "unknown"
        
        # Get sample objects - handle as potential coroutine
        sample_result = run_coro(get_objects(collection))
        
        if sample_result and hasattr(sample_result, 'objects') and sample_result.objects:
            logger.info(f"Retrieved {len(sample_result.objects)} sample objects")
            
            # Check the first object
            sample_obj = sample_result.objects[0]
            logger.info(f"Sample object properties: {sample_obj.properties}")
            
            # Check vector format
            if hasattr(sample_obj, 'vector') and sample_obj.vector:
                vector_type = type(sample_obj.vector)
                vector_len = len(sample_obj.vector) if hasattr(sample_obj.vector, '__len__') else 'unknown'
                logger.info(f"Sample object vector type: {vector_type}, length: {vector_len}")
                
                # Check vector format
                if isinstance(sample_obj.vector, list):
                    logger.info("Vector is in list format (good)")
                    # Check first few values
                    if len(sample_obj.vector) > 0:
                        logger.info(f"First few values: {sample_obj.vector[:3]}")
                elif isinstance(sample_obj.vector, dict):
                    logger.info(f"Vector is a dictionary with keys: {list(sample_obj.vector.keys())}")
                else:
                    logger.warning(f"Vector is in an unexpected format: {type(sample_obj.vector)}")
            else:
                logger.warning("Sample object does not have a vector")
            
            # Return a hash for testing
            sample_hash = sample_obj.properties.get("hash")
            if sample_hash:
                logger.info(f"Sample hash for testing: {sample_hash}")
                return True, sample_result.objects, sample_hash
            else:
                logger.warning("Sample object does not have a hash property")
                return True, sample_result.objects, None
        else:
            logger.warning("No objects found in UniqueStrings collection")
            return False, [], None
        
    except Exception as e:
        logger.error(f"Error checking UniqueStrings collection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, [], None

async def test_hash_query_async(client, hash_value: str):
    """Test querying objects by hash - async version."""
    logger.info(f"Testing hash query with: {hash_value}")
    
    try:
        collection = client.collections.get("UniqueStrings")
        
        # Query by hash - handle both coroutine and non-coroutine versions
        fetch_objects = collection.query.fetch_objects
        result = None
        
        if asyncio.iscoroutinefunction(fetch_objects):
            # It's a coroutine function
            result = await fetch_objects(
                filters=Filter.by_property("hash").equal(hash_value),
                limit=1,
                include_vector=True
            )
        else:
            # It's a regular function
            result = fetch_objects(
                filters=Filter.by_property("hash").equal(hash_value),
                limit=1,
                include_vector=True
            )
        
        if result and hasattr(result, 'objects') and result.objects:
            obj = result.objects[0]
            logger.info(f"Found object with properties: {obj.properties}")
            
            # Check vector
            if hasattr(obj, 'vector') and obj.vector:
                vector_type = type(obj.vector)
                vector_len = len(obj.vector) if hasattr(obj.vector, '__len__') else 'unknown'
                logger.info(f"Object vector type: {vector_type}, length: {vector_len}")
                
                # Detailed vector format check
                if isinstance(obj.vector, list):
                    logger.info(f"Vector is a list of length {len(obj.vector)}")
                    # Check first few values
                    if len(obj.vector) > 0:
                        logger.info(f"First few values: {obj.vector[:3]}")
                elif isinstance(obj.vector, dict):
                    logger.info(f"Vector is a dictionary with keys: {list(obj.vector.keys())}")
                    # Check structure of first value if possible
                    if len(obj.vector) > 0:
                        key = next(iter(obj.vector.keys()))
                        val = obj.vector[key]
                        logger.info(f"Value for key '{key}' is of type {type(val)}")
                else:
                    logger.warning(f"Vector has unexpected type: {type(obj.vector)}")
            else:
                logger.warning("Object does not have a vector")
            
            return True
        else:
            logger.warning(f"No object found for hash: {hash_value}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing hash query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_hash_query(client, hash_value: str):
    """Test querying objects by hash - wrapper for async version."""
    return run_coro(test_hash_query_async(client, hash_value))

async def test_problem_hashes_async(client):
    """Test querying objects with the problematic hashes from the error logs."""
    problem_hashes = [
        "17157bb4f736b44bfc499d4521e0950c",
        "d0c1c7a17041d6f2fb0c1a844afee857"
    ]
    
    logger.info(f"Testing {len(problem_hashes)} problematic hashes from error logs")
    
    for hash_value in problem_hashes:
        await test_hash_query_async(client, hash_value)

def test_problem_hashes(client):
    """Test problem hashes - wrapper for async version."""
    return run_coro(test_problem_hashes_async(client))

def safe_close_client(client):
    """Safely close the client connection."""
    if client:
        try:
            if hasattr(client, 'close'):
                client.close()
                logger.info("Weaviate client closed properly")
            # Also try to close any low-level connections
            if hasattr(client, '_connection') and hasattr(client._connection, 'close'):
                try:
                    run_coro(client._connection.close())
                    logger.info("Low-level connection closed properly")
                except Exception as e:
                    logger.warning(f"Error closing low-level connection: {e}")
        except Exception as close_e:
            logger.error(f"Error closing Weaviate client: {close_e}")

def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Check Weaviate connection and functionality")
    parser.add_argument("--url", default="http://localhost:8080", help="Weaviate URL")
    parser.add_argument("--hash", help="Specific hash to query")
    parser.add_argument("--setup", action="store_true", help="Set up UniqueStrings collection if missing")
    args = parser.parse_args()
    
    # Connect to Weaviate
    client, collections = check_weaviate_connection(args.url)
    
    if not client:
        logger.error("Failed to connect to Weaviate")
        return 1
    
    # Check if required collections exist
    if "UniqueStrings" not in collections:
        logger.error(f"Required collection 'UniqueStrings' not found in {collections}")
        
        # Option to set up the collection if it doesn't exist
        if args.setup:
            logger.info("Attempting to create UniqueStrings collection...")
            try:
                # Define collection properties
                from weaviate.classes.config import Configure, Property, DataType, Tokenization
                
                unique_strings_properties = [
                    Property(
                        name="text",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.WHITESPACE,
                        description="The unique text string"
                    ),
                    Property(
                        name="field_type",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="The type of field (e.g., person, title)"
                    ),
                    Property(
                        name="hash",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="Hash of the text string for exact lookup"
                    ),
                    Property(
                        name="frequency",
                        data_type=DataType.INT,
                        description="How many times this string appears in the dataset"
                    )
                ]
                
                # Create the collection
                client.collections.create(
                    name="UniqueStrings",
                    description="Collection for storing unique text strings and their vectors",
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=unique_strings_properties
                )
                
                logger.info("Successfully created UniqueStrings collection")
                collections.append("UniqueStrings")
            except Exception as e:
                logger.error(f"Error creating collection: {e}")
                safe_close_client(client)
                return 1
        else:
            logger.info("Use --setup flag to create the collection if it doesn't exist")
            safe_close_client(client)
            return 1
    
    # Check UniqueStrings collection
    collection_ok, sample_objects, sample_hash = check_unique_strings_collection(client)
    
    if not collection_ok:
        logger.error("Failed to check UniqueStrings collection")
        safe_close_client(client)
        return 1
    
    # Test hash query
    if args.hash:
        # Use hash from command line
        test_hash_query(client, args.hash)
    elif sample_hash:
        # Use hash from sample object
        test_hash_query(client, sample_hash)
    
    # Test problematic hashes
    test_problem_hashes(client)
    
    # Ensure proper closure of the Weaviate client
    safe_close_client(client)
    
    logger.info("Diagnostic checks completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)