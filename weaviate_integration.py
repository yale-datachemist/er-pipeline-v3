import atexit
import logging
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Iterable
import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.collections import Collection
from weaviate.classes.query import Filter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeaviateManager:
    """
    Manages interactions with Weaviate for string-centric vector storage and retrieval.
    
    This approach stores unique strings and their vectors rather than entire records,
    drastically improving efficiency and reducing duplication.
    """
    def __init__(self, config: Dict):
        """
        Initialize the Weaviate manager with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.url = config.get("weaviate_url", "http://localhost:8080")
        self.batch_size = config.get("weaviate_batch_size", 100)
        self.retry_count = config.get("weaviate_retry_count", 3)
        
        self.client = None
        
        # Fields to embed
        self.embedable_fields = [
            "record", "person", "roles", "title", "attribution", 
            "provision", "subjects", "genres", "relatedWork"
        ]
        
        # Define the collection schema
        self.unique_strings_properties = [
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
        
        # Define the entity map schema
        self.entity_map_properties = [
            Property(
                name="entity_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                description="Unique entity identifier"
            ),
            Property(
                name="field_hashes_json",
                data_type=DataType.TEXT,
                description="JSON string mapping field names to their string hashes"
            ),
            Property(
                name="person_name",
                data_type=DataType.TEXT,
                description="Person name for this entity"
            )
        ]
        
        # Register cleanup handler for program exit
        atexit.register(self.close)
    
    def connect(self) -> bool:
        """
        Connect to the Weaviate instance.
        
        Returns:
            Boolean indicating success
        """
        # If already connected, return True
        if self.client:
            try:
                # Try a simple operation to check if connection is still alive
                meta = self.client.get_meta()
                logger.info("Using existing Weaviate connection")
                return True
            except Exception:
                # Connection is stale, close it and reconnect
                self.close()
                
        for attempt in range(self.retry_count):
            try:
                # Try different connection methods based on configuration
                if self.url.startswith("http://localhost") or self.url.startswith("http://127.0.0.1"):
                    self.client = weaviate.connect_to_local()
                else:
                    self.client = weaviate.connect_to_wcs(
                        cluster_url=self.url,
                        auth_credentials=weaviate.auth.AuthApiKey(self.config.get("weaviate_api_key", ""))
                    )
                
                # Check if connection is successful
                meta = self.client.get_meta()
                logger.info(f"Connected to Weaviate version: {meta}")
                return True
                
            except weaviate.exceptions.WeaviateConnectionError as e:
                logger.warning(f"Weaviate connection error (attempt {attempt+1}): {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1} failed: {str(e)}")
                import traceback
                logger.warning(traceback.format_exc())
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to connect to Weaviate after {self.retry_count} attempts")
        return False
    
    def close(self) -> None:
        """
        Close the Weaviate connection and perform cleanup.
        """
        if self.client:
            try:
                # Close any active batch operations
                collections = getattr(self.client, 'collections', None)
                if collections:
                    collection_list = collections.list_all()
                    for collection in collection_list:
                        if hasattr(collection, 'batch') and collection.batch:
                            try:
                                collection.batch.close()
                                logger.info(f"Closed batch for collection: {collection.name}")
                            except Exception as e:
                                logger.warning(f"Error closing batch for {collection.name}: {str(e)}")
                
                # Close the client connection
                if hasattr(self.client, 'close'):
                    self.client.close()
                    logger.info("Weaviate connection closed")
                
                # Also attempt to close using _connection attribute if available
                if hasattr(self.client, '_connection') and hasattr(self.client._connection, 'close'):
                    self.client._connection.close()
                    logger.info("Weaviate low-level connection closed")
                
                # Set client to None to indicate it's been closed
                self.client = None
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
    
    def __del__(self):
        """
        Destructor to ensure connection is closed when object is garbage collected.
        """
        self.close()
    
    def setup_collections(self) -> bool:
        """
        Set up the required collections in Weaviate with robust existence checking.
        
        Returns:
            Boolean indicating success
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return False
                
        try:
            # Check if collections already exist - using direct schema endpoint for reliability
            try:
                logger.info("Checking existing collections using schema endpoint")
                schema = self.client._connection.get("/v1/schema")
                existing_classes = [cls["class"] for cls in schema.get("classes", [])]
                logger.info(f"Found existing collections via schema: {existing_classes}")
            except Exception as schema_e:
                logger.warning(f"Error checking schema directly: {schema_e}")
                # Fall back to list_all method
                logger.info("Falling back to collections.list_all()")
                try:
                    existing_collections = self.client.collections.list_all()
                    
                    # Handle different return types based on Weaviate client version
                    if isinstance(existing_collections, list):
                        try:
                            existing_names = [c.name for c in existing_collections]
                        except AttributeError:
                            # If the items don't have name attribute, they might be strings
                            existing_names = existing_collections
                    else:
                        existing_names = []
                        
                    logger.info(f"Found existing collections via list_all: {existing_names}")
                    existing_classes = existing_names
                except Exception as list_e:
                    logger.warning(f"Error using list_all: {list_e}")
                    # Assume no collections exist if we can't check
                    existing_classes = []
                    
            # Get vector index configuration based on environment
            hnsw_config = self._get_vector_index_config()
            
            # Setup UniqueStrings collection
            if "UniqueStrings" not in existing_classes:
                logger.info("Creating collection: UniqueStrings")
                try:
                    self.client.collections.create(
                        name="UniqueStrings",
                        description="Collection for storing unique text strings and their vectors",
                        vectorizer_config=Configure.Vectorizer.none(),
                        properties=self.unique_strings_properties,
                        vector_index_config=hnsw_config
                    )
                    logger.info("Created collection: UniqueStrings")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.info("Collection UniqueStrings already exists (caught from error)")
                    else:
                        raise
            else:
                logger.info("Collection already exists: UniqueStrings")
            
            # Setup EntityMap collection (optional)
            if "EntityMap" not in existing_classes:
                logger.info("Creating collection: EntityMap")
                try:
                    self.client.collections.create(
                        name="EntityMap",
                        description="Collection for mapping entities to their component strings",
                        vectorizer_config=Configure.Vectorizer.none(),  # No vectors needed
                        properties=self.entity_map_properties
                    )
                    logger.info("Created collection: EntityMap")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.info("Collection EntityMap already exists (caught from error)")
                    else:
                        raise
            else:
                logger.info("Collection already exists: EntityMap")
            
            # Verify collections exist
            try:
                unique_strings_collection = self.client.collections.get("UniqueStrings")
                entity_map_collection = self.client.collections.get("EntityMap")
                logger.info("Successfully verified collections exist")
                return True
            except Exception as e:
                logger.error(f"Error verifying collections: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up collections: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_vector_index_config(self) -> Any:
        """
        Get appropriate vector index configuration based on environment settings.
        
        Returns:
            Vector index configuration object
        """
        # Get environment-specific parameters
        env_settings = self.config.get("environment", {})
        env_type = "production" if self.config.get("mode") == "production" else "development"
        vector_params = {}
        
        if env_type in env_settings and "vector_index_params" in env_settings[env_type]:
            vector_params = env_settings[env_type]["vector_index_params"]
        
        # Set default values if not specified
        ef_construction = vector_params.get("ef_construction", 128)
        ef = vector_params.get("ef", 128)
        max_connections = vector_params.get("max_connections", 64)
        
        # Create HNSW configuration
        return Configure.VectorIndex.hnsw(
            distance_metric=weaviate.classes.config.VectorDistances.COSINE,
            ef_construction=ef_construction,
            max_connections=max_connections,
            ef=ef,
            dynamic_ef_factor=8,
            dynamic_ef_min=100,
            dynamic_ef_max=500,
        )
    
    def index_unique_strings(
        self, 
        unique_strings: Dict[str, str], 
        embeddings: Dict[str, List[float]], 
        string_counts: Dict[str, int],
        field_types: Dict[str, str]
    ) -> bool:
        """
        Index unique strings and their vectors in Weaviate.
        
        Args:
            unique_strings: Dictionary mapping hash -> original string
            embeddings: Dictionary mapping hash -> vector embedding
            string_counts: Dictionary mapping hash -> frequency count
            field_types: Dictionary mapping hash -> field type
            
        Returns:
            Boolean indicating success
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return False
        
        try:
            collection = self.client.collections.get("UniqueStrings")
            total_strings = len(unique_strings)
            logger.info(f"Indexing {total_strings} unique strings")
            
            # Process in batches
            batch_size = self.batch_size
            string_hashes = list(unique_strings.keys())
            
            with collection.batch.dynamic() as batch:
                for i in tqdm(range(0, total_strings, batch_size), desc="Indexing strings"):
                    batch_hashes = string_hashes[i:i+batch_size]
                    
                    for hash_value in batch_hashes:
                        # Skip if hash isn't in all dictionaries
                        if (hash_value not in unique_strings or 
                            hash_value not in embeddings or 
                            hash_value not in field_types):
                            continue
                        
                        # Get properties
                        text = unique_strings[hash_value]
                        vector = embeddings[hash_value]
                        field_type = field_types[hash_value]
                        frequency = string_counts.get(hash_value, 1)
                        
                        # Add to batch
                        batch.add_object(
                            properties={
                                "text": text,
                                "field_type": field_type,
                                "hash": hash_value,
                                "frequency": frequency
                            },
                            vector=vector
                        )
            
            logger.info(f"Indexed {total_strings} unique strings")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing unique strings: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def index_entity_maps(self, entity_maps: List[Dict]) -> bool:
        """
        Index entity-to-string maps in the EntityMap collection.
        
        Args:
            entity_maps: List of entity map dictionaries
            
        Returns:
            Boolean indicating success
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return False
        
        try:
            collection = self.client.collections.get("EntityMap")
            total_entities = len(entity_maps)
            logger.info(f"Indexing {total_entities} entity maps")
            
            with collection.batch.dynamic() as batch:
                for entity_map in tqdm(entity_maps, desc="Indexing entity maps"):
                    batch.add_object(
                        properties={
                            "entity_id": entity_map["entity_id"],
                            "field_hashes_json": json.dumps(entity_map["field_hashes"]),  # Serialize to JSON string
                            "person_name": entity_map.get("person_name", "")
                        }
                    )
            
            logger.info(f"Indexed {total_entities} entity maps")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing entity maps: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def get_record_by_id(self, record_id: str) -> Optional[Dict]:
        """
        Get a record by its ID.
        
        This reconstructs a record from the entity map and unique strings.
        
        Args:
            record_id: ID of the record to retrieve
            
        Returns:
            Record dictionary or None if not found
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return None
        
        try:
            # First, try to get the entity map for this record ID
            field_hashes = self.get_entity_field_hashes(record_id)
            
            if not field_hashes:
                logger.warning(f"No entity map found for record ID: {record_id}")
                return None
            
            # Reconstruct the record from its field hashes
            record = {"id": record_id}
            
            # Get the actual string values for each hash
            for field, hash_value in field_hashes.items():
                if hash_value == "NULL":
                    record[field] = None
                    continue
                    
                # Get the string from UniqueStrings collection
                string_collection = self.client.collections.get("UniqueStrings")
                string_result = string_collection.query.fetch_objects(
                    filters=string_collection.query.filter.by_property("hash").equal(hash_value),
                    limit=1
                )
                
                if string_result.objects:
                    record[field] = string_result.objects[0].properties.get("text", "")
                else:
                    logger.warning(f"No string found for hash {hash_value} (field {field}, record {record_id})")
                    record[field] = ""
            
            return record
            
        except Exception as e:
            logger.error(f"Error retrieving record by ID {record_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None    

    def get_entity_field_hashes(self, entity_id: str) -> Optional[Dict[str, str]]:
        """
        Retrieve field hashes for an entity from the EntityMap collection.
        
        Args:
            entity_id: Entity ID to retrieve
            
        Returns:
            Dictionary mapping field name -> hash, or None if not found
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return None
        
        try:
            collection = self.client.collections.get("EntityMap")
            
            # Query by entity_id for exact match
            result = collection.query.fetch_objects(
                filters=collection.query.filter.by_property("entity_id").equal(entity_id),
                limit=1
            )
            
            if result.objects:
                # Deserialize the JSON string back to a dictionary
                field_hashes_json = result.objects[0].properties.get("field_hashes_json", "{}")
                try:
                    return json.loads(field_hashes_json)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing field_hashes_json for entity {entity_id}: {e}")
                    return {}
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving entity field hashes: {str(e)}")
            return None
    
    def get_vector_by_hash(self, hash_value: str) -> Optional[List[float]]:
        """
        Retrieve a vector by its string hash.
        
        Args:
            hash_value: Hash of the string
            
        Returns:
            Vector embedding if found, None otherwise
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return None
        
        try:
            collection = self.client.collections.get("UniqueStrings")
            
            # Query by hash for exact match
            result = collection.query.fetch_objects(
                filters=collection.query.filter.by_property("hash").equal(hash_value),
                limit=1,
                include_vector=True
            )
            
            if result.objects:
                return result.objects[0].vector
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving vector by hash: {str(e)}")
            return None
    
    def get_vectors_by_hashes(self, hash_values: List[str]) -> Dict[str, List[float]]:
        """
        Retrieve multiple vectors by their string hashes in batch.
        
        Args:
            hash_values: List of string hashes
            
        Returns:
            Dictionary mapping hash -> vector
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return {}
        
        result_vectors = {}
        unique_hashes = list(set(hash_values))  # Remove duplicates
        
        # Process in reasonable batch sizes
        batch_size = 100
        for i in range(0, len(unique_hashes), batch_size):
            batch_hashes = unique_hashes[i:i+batch_size]
            
            try:
                collection = self.client.collections.get("UniqueStrings")
                
                # Build filter for batch of hashes
                hash_filter = Filter.by_property("hash").contains_any(batch_hashes)
                
                # Execute query
                result = collection.query.fetch_objects(
                    filters=hash_filter,
                    limit=len(batch_hashes) + 10,  # Add buffer for safety
                    include_vector=True
                )
                
                # Extract vectors
                for obj in result.objects:
                    hash_value = obj.properties.get("hash")
                    if hash_value and obj.vector:
                        result_vectors[hash_value] = obj.vector
                
            except Exception as e:
                logger.error(f"Error batch retrieving vectors: {str(e)}")
        
        # Report on missing hashes
        missing_hashes = set(hash_values) - set(result_vectors.keys())
        if missing_hashes:
            logger.warning(f"Could not retrieve vectors for {len(missing_hashes)} hashes")
        
        return result_vectors
    
    def get_field_vectors_for_records(self, records_field_hashes: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, List[float]]]:
        """
        Retrieve vectors for multiple records in batch.
        
        Args:
            records_field_hashes: Dictionary mapping record ID -> {field -> hash}
            
        Returns:
            Dictionary mapping record ID -> {field -> vector}
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return {}
        
        # Collect all unique hashes
        all_hashes = set()
        for record_id, field_hashes in records_field_hashes.items():
            for field, hash_value in field_hashes.items():
                if hash_value != "NULL":
                    all_hashes.add(hash_value)
        
        # Get all vectors in one batch request
        hash_vectors = self.get_vectors_by_hashes(list(all_hashes))
        
        # Map vectors to records and fields
        record_field_vectors = {}
        for record_id, field_hashes in records_field_hashes.items():
            field_vectors = {}
            for field, hash_value in field_hashes.items():
                if hash_value != "NULL" and hash_value in hash_vectors:
                    field_vectors[field] = hash_vectors[hash_value]
            
            record_field_vectors[record_id] = field_vectors
        
        return record_field_vectors
    
    def search_similar_strings(
        self, 
        query_vector: List[float], 
        field_type: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for strings similar to the query vector, optionally filtering by field type.
        
        Args:
            query_vector: Query vector embedding
            field_type: Optional field type to filter results
            limit: Maximum number of results
            
        Returns:
            List of similar strings with their vectors and distances
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return []
        
        try:
            collection = self.client.collections.get("UniqueStrings")
            
            # Build query
            query = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=Configure.return_metadata(distance=True),
                include_vector=True
            )
            
            # Add field type filter if specified
            if field_type:
                query = query.with_filter(
                    collection.query.filter.by_property("field_type").equal(field_type)
                )
            
            # Execute query
            result = query.fetch_objects()
            
            # Process results
            similar_strings = []
            for obj in result.objects:
                similar_strings.append({
                    "text": obj.properties.get("text", ""),
                    "hash": obj.properties.get("hash", ""),
                    "field_type": obj.properties.get("field_type", ""),
                    "frequency": obj.properties.get("frequency", 0),
                    "vector": obj.vector,
                    "distance": obj.metadata.distance
                })
            
            return similar_strings
            
        except Exception as e:
            logger.error(f"Error searching similar strings: {str(e)}")
            return []
    
    def get_imputation_candidates(
        self, 
        query_vector: List[float], 
        field_type: str, 
        limit: int = 10
    ) -> List[Dict]:
        """
        Get candidate values for imputing a missing field using vector similarity.
        
        Args:
            query_vector: Query vector embedding (typically the record vector)
            field_type: Field type to impute
            limit: Maximum number of candidates
            
        Returns:
            List of candidate values with their vectors and distances
        """
        # Use search_similar_strings with field type filter
        candidates = self.search_similar_strings(
            query_vector=query_vector,
            field_type=field_type,
            limit=limit * 2  # Get more candidates to filter
        )
        
        # Extract unique text values
        unique_texts = []
        seen_texts = set()
        
        for candidate in candidates:
            text = candidate["text"]
            if text and text.strip() and text not in seen_texts:
                seen_texts.add(text)
                unique_texts.append({
                    "text": text,
                    "vector": candidate["vector"],
                    "distance": candidate["distance"]
                })
                
                if len(unique_texts) >= limit:
                    break
        
        return unique_texts
    
    def clear_collections(self) -> bool:
        """
        Clear existing collections for fresh indexing.
        
        Returns:
            Boolean indicating success
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return False
        
        try:
            for collection_name in ["UniqueStrings", "EntityMap"]:
                try:
                    collection = self.client.collections.get(collection_name)
                    logger.info(f"Deleting all objects in collection: {collection_name}")
                    collection.data.delete_all()
                    logger.info(f"Successfully cleared collection: {collection_name}")
                except:
                    logger.info(f"Collection {collection_name} doesn't exist or is already empty")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collections: {str(e)}")
            return False    

    