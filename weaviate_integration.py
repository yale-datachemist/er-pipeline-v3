    def get_imputation_candidates(self, query_vector: List[float], field: str, limit: int = 10) -> List[str]:
        """
        Get candidate values for imputing a missing field.
        
        Args:
            query_vector: Query vector embedding
            field: Field to impute
            limit: Maximum number of results
            
        Returns:
            List of candidate values
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return []
            
        try:
            collection = self.client.collections.get("EntityRecord")
            
            # Perform vector search, only retrieving records where the field is not empty
            result = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit * 2,  # Request more to account for filtering
                return_metadata=Configure.return_metadata(distance=True),
                return_properties=["id", field]
            )
            
            # Filter out empty values and take up to the limit
            candidates = []
            for obj in result.objects:
                field_value = obj.properties.get(field)
                if field_value and field_value.strip():
                    candidates.append(field_value)
                    if len(candidates) >= limit:
                        break
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting imputation candidates: {str(e)}")
            return []import logging
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
import weaviate
from weaviate.client import WeaviateClient
from weaviate.classes.config import Configure
from weaviate.collections import Collection
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeaviateManager:
    """
    Manages interactions with Weaviate for vector storage and retrieval.
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
        self.collection_configs = {
            "EntityRecord": {
                "description": "Collection for storing entity records",
                "vectorizer": "none",  # We provide our own vectors
                "properties": [
                    {
                        "name": "record",
                        "dataType": ["text"],
                        "description": "Complete record text",
                        "indexSearchable": True,
                        "indexFilterable": True,
                    },
                    {
                        "name": "person",
                        "dataType": ["text"],
                        "description": "Person name",
                        "indexSearchable": True,
                        "indexFilterable": True,
                    },
                    {
                        "name": "roles",
                        "dataType": ["text"],
                        "description": "Person roles",
                        "indexFilterable": True,
                    },
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "Work title",
                        "indexSearchable": True,
                        "indexFilterable": True,
                    },
                    {
                        "name": "attribution",
                        "dataType": ["text"],
                        "description": "Attribution statement",
                        "indexFilterable": True,
                    },
                    {
                        "name": "provision",
                        "dataType": ["text"],
                        "description": "Publication details",
                        "indexFilterable": True,
                    },
                    {
                        "name": "subjects",
                        "dataType": ["text"],
                        "description": "Subject classifications",
                        "indexFilterable": True,
                    },
                    {
                        "name": "genres",
                        "dataType": ["text"],
                        "description": "Genre classifications",
                        "indexFilterable": True,
                    },
                    {
                        "name": "relatedWork",
                        "dataType": ["text"],
                        "description": "Related work title",
                        "indexFilterable": True,
                    },
                    {
                        "name": "recordId",
                        "dataType": ["text"],
                        "description": "Original record ID",
                        "indexFilterable": True,
                        "tokenization": "field",
                    },
                    {
                        "name": "id",
                        "dataType": ["text"],
                        "description": "Unique entity identifier",
                        "indexFilterable": True,
                        "tokenization": "field",
                    },
                    {
                        "name": "field_vectors",
                        "dataType": ["object"],
                        "description": "Map of field names to vector embeddings",
                        "indexFilterable": False,
                    }
                ],
                "vectorIndexConfig": {
                    "skip": False,
                    "distance": "cosine",
                },
                "multiTenancyConfig": {
                    "enabled": False,
                },
                "replicationConfig": {
                    "factor": 1
                }
            }
        }
    
    def connect(self) -> bool:
        """
        Connect to the Weaviate instance.
        
        Returns:
            Boolean indicating success
        """
        for attempt in range(self.retry_count):
            try:
                self.client = weaviate.Client(
                    url=self.url,
                    additional_headers={
                        "X-OpenAI-Api-Key": self.config.get("openai_api_key", "")  # For potential hybrid search
                    }
                )
                
                # Check if connection is successful
                meta = self.client.get_meta()
                logger.info(f"Connected to Weaviate version: {meta.version}")
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
    
    def setup_collections(self) -> bool:
        """
        Set up the required collections in Weaviate.
        
        Returns:
            Boolean indicating success
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return False
            
        try:
            # Check if collections already exist
            existing_collections = self.client.collections.list_all()
            existing_names = [c.name for c in existing_collections]
            
            # Create collections that don't exist or update them if needed
            for name, config in self.collection_configs.items():
                if name not in existing_names:
                    logger.info(f"Creating collection: {name}")
                    
                    # Create collection with properties and optimized HNSW parameters
                    collection = self.client.collections.create(
                        name=name,
                        description=config["description"],
                        vectorizer_config=Configure.Vectorizer.none(),
                        properties=config["properties"],
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=Configure.VectorDistance.cosine,
                            ef=128,               # Higher values: better recall, slower queries
                            ef_construction=128,   # Higher values: better recall, slower indexing
                            max_connections=64     # Higher values: better recall, more memory
                        )
                    )
                    
                    logger.info(f"Created collection: {name}")
                else:
                    # Collection exists, should validate if schema matches expectations
                    collection = self.client.collections.get(name)
                    logger.info(f"Collection already exists: {name}")
                    
                    # Simple validation of existing properties
                    existing_props = [prop.name for prop in collection.config.properties]
                    expected_props = [prop["name"] for prop in config["properties"]]
                    
                    missing_props = set(expected_props) - set(existing_props)
                    if missing_props:
                        logger.warning(f"Collection {name} is missing properties: {missing_props}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up collections: {str(e)}")
            return False
    
    def index_record(self, record: Dict, field_embeddings: Dict[str, List[float]]) -> str:
        """
        Index a record in Weaviate with its associated embeddings.
        
        Args:
            record: Dictionary containing record fields
            field_embeddings: Dictionary mapping field names to embeddings
            
        Returns:
            UUID of the indexed object
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return None
            
        try:
            collection = self.client.collections.get("EntityRecord")
            
            # Prepare record for indexing
            properties = {
                "record": record.get("record", ""),
                "person": record.get("person", ""),
                "roles": record.get("roles", ""),
                "title": record.get("title", ""),
                "attribution": record.get("attribution", ""),
                "provision": record.get("provision", ""),
                "subjects": record.get("subjects", ""),
                "genres": record.get("genres", ""),
                "relatedWork": record.get("relatedWork", ""),
                "recordId": record.get("recordId", ""),
                "id": record.get("id", ""),
                "field_vectors": json.dumps(field_embeddings)
            }
            
            # Use person field vector as the primary vector for the record
            vector = field_embeddings.get("person", [])
            
            # Add the object
            result = collection.data.insert(
                properties=properties,
                vector=vector
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error indexing record: {str(e)}")
            return None
    
    def batch_index_records(self, records: List[Dict], embeddings_dict: Dict[str, Dict[str, List[float]]]) -> List[str]:
        """
        Index a batch of records in Weaviate.
        
        Args:
            records: List of record dictionaries
            embeddings_dict: Dictionary mapping record IDs to field embeddings
            
        Returns:
            List of UUIDs for the indexed objects
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return []
            
        try:
            collection = self.client.collections.get("EntityRecord")
            
            with collection.batch.dynamic() as batch:
                for record in tqdm(records, desc="Indexing records"):
                    record_id = record.get("id")
                    if record_id not in embeddings_dict:
                        logger.warning(f"No embeddings found for record: {record_id}")
                        continue
                        
                    field_embeddings = embeddings_dict[record_id]
                    
                    # Prepare record for indexing
                    properties = {
                        "record": record.get("record", ""),
                        "person": record.get("person", ""),
                        "roles": record.get("roles", ""),
                        "title": record.get("title", ""),
                        "attribution": record.get("attribution", ""),
                        "provision": record.get("provision", ""),
                        "subjects": record.get("subjects", ""),
                        "genres": record.get("genres", ""),
                        "relatedWork": record.get("relatedWork", ""),
                        "recordId": record.get("recordId", ""),
                        "id": record_id,
                        "field_vectors": json.dumps(field_embeddings)
                    }
                    
                    # Use person field vector as the primary vector for the record
                    vector = field_embeddings.get("person", [])
                    
                    # Add the object to the batch
                    batch.add_object(
                        properties=properties,
                        vector=vector
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Error batch indexing records: {str(e)}")
            return False
    
    def search_neighbors(self, query_vector: List[float], limit: int = 10, distance_threshold: float = 0.2) -> List[Dict]:
        """
        Search for nearest neighbors of a query vector.
        
        Args:
            query_vector: Query vector embedding
            limit: Maximum number of results to return
            distance_threshold: Maximum distance threshold
            
        Returns:
            List of neighbor objects
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return []
            
        try:
            collection = self.client.collections.get("EntityRecord")
            
            # Perform vector search
            result = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=Configure.return_metadata(distance=True),
                return_properties=["id", "person", "title", "record"]
            )
            
            # Filter by distance threshold
            filtered_results = []
            for obj in result.objects:
                distance = obj.metadata.distance
                if distance <= distance_threshold:
                    filtered_results.append({
                        "id": obj.properties["id"],
                        "person": obj.properties["person"],
                        "title": obj.properties["title"],
                        "record": obj.properties["record"],
                        "distance": distance
                    })
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching neighbors: {str(e)}")
            return []
    
    def get_record_by_id(self, record_id: str) -> Optional[Dict]:
        """
        Retrieve a record by its ID.
        
        Args:
            record_id: Record ID to retrieve
            
        Returns:
            Record object if found, None otherwise
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return None
            
        try:
            collection = self.client.collections.get("EntityRecord")
            
            # Query by ID
            result = collection.query.fetch_objects(
                filters=collection.query.filter.by_property("id").equal(record_id),
                limit=1
            )
            
            if result.objects:
                return result.objects[0].properties
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving record: {str(e)}")
            return None
    
    def get_imputation_candidate_records(self, query_vector: List[float], field: str, limit: int = 10) -> List[Dict]:
        """
        Get candidate records with field vectors for imputing a missing field.
        
        Args:
            query_vector: Query vector embedding
            field: Field to impute
            limit: Maximum number of results
            
        Returns:
            List of candidate records with their field vectors
        """
        if not self.client:
            logger.error("Not connected to Weaviate")
            return []
            
        try:
            collection = self.client.collections.get("EntityRecord")
            
            # Perform vector search, only retrieving records where the field is not empty
            result = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit * 2,  # Request more to account for filtering
                return_metadata=Configure.return_metadata(distance=True),
                return_properties=["id", field, "field_vectors"]
            )
            
            # Filter out empty values and take up to the limit
            candidates = []
            for obj in result.objects:
                field_value = obj.properties.get(field)
                if field_value and field_value.strip():
                    # Include the field value, distance, and field_vectors
                    candidate = {
                        "id": obj.properties.get("id"),
                        field: field_value,
                        "distance": obj.metadata.distance
                    }
                    
                    # Add field_vectors if available
                    if "field_vectors" in obj.properties:
                        candidate["field_vectors"] = obj.properties["field_vectors"]
                    
                    candidates.append(candidate)
                    if len(candidates) >= limit:
                        break
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting imputation candidate records: {str(e)}")
            return []