    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from a YAML file with environment-specific scaling parameters.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set environment-specific defaults
        mode = config.get("mode", "dev")
        
        # Apply environment-specific settings based on mode
        env_settings = None
        if mode == "dev":
            env_type = "development"
            config.setdefault("max_records", 10000)
            config.setdefault("checkpoint_frequency", 10)
        else:  # production mode
            env_type = "production"
            config.setdefault("max_records", None)  # Process all records
            config.setdefault("checkpoint_frequency", 50)
        
        # Apply environment scaling parameters if available
        if "environment" in config and env_type in config["environment"]:
            env_settings = config["environment"][env_type]
            
            # Update batch sizes
            if "batch_sizes" in env_settings:
                batch_sizes = env_settings["batch_sizes"]
                if "embedding" in batch_sizes:
                    config["embedding_batch_size"] = batch_sizes["embedding"]
                if "weaviate" in batch_sizes:
                    config["weaviate_batch_size"] = batch_sizes["weaviate"]
                if "graph" in batch_sizes:
                    config["graph_batch_size"] = batch_sizes["graph"]
            
            # Update parallelism
            if "parallelism" in env_settings:
                config["embedding_workers"] = env_settings["parallelism"]
            
            # Update memory limits
            if "memory_limits" in env_settings:
                for key, value in env_settings["memory_limits"].items():
                    config[key] = value
            
            # Update vector index parameters for Weaviate
            if "vector_index_params" in env_settings:
                config["vector_index_params"] = env_settings["vector_index_params"]
            
            # Update Weaviate resource limits (for Docker Compose)
            if "weaviate_resources" in env_settings:
                config["weaviate_resources"] = env_settings["weaviate_resources"]
            
            logger.info(f"Applied {env_type} environment scaling parameters")
        
        # Add checkpointing settings
        config.setdefault("enable_checkpointing", True)
        config.setdefault("checkpoint_after_stage", True)
        
        return configimport os
import json
import logging
import argparse
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
import time
from datetime import datetime
import yaml

# Import modules
from preprocessing import DataProcessor, TextDeduplicator
from embedding import EmbeddingGenerator
from weaviate_integration import WeaviateManager
from feature_engineering import FeatureEngineer, LogisticRegressionClassifier
from imputation_clustering import NullValueImputer, EntityClusterer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("entity_resolution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EntityResolutionPipeline:
    """
    Main pipeline for entity resolution.
    """
    def __init__(self, config_path: str):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        
        # Validate configuration
        self._validate_config()
        
        self.output_dir = self.config.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize checkpointing system
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_state = self._load_checkpoint_state()
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.weaviate_manager = WeaviateManager(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.classifier = LogisticRegressionClassifier(self.config)
        
        # These components require initialized instances of other components
        self.imputer = None
        self.clusterer = None
        
        # Track performance
        self.start_time = None
        self.stats = self.checkpoint_state.get("stats", {})
        
        # Restore state from checkpoint if needed
        self.record_embeddings = {}
        self._restore_checkpoint_data()
    
    def _validate_config(self) -> None:
        """Validate that the configuration contains all required parameters."""
        required_params = [
            "data_dir", "mode", "openai_api_key", "weaviate_url",
            "embedding_model", "ground_truth_file"
        ]
        
        missing_params = [param for param in required_params if param not in self.config]
        if missing_params:
            raise ValueError(f"Missing required configuration parameters: {', '.join(missing_params)}")
        
        # Validate parameter values
        if self.config.get("mode") not in ["dev", "production"]:
            raise ValueError("Mode must be either 'dev' or 'production'")
            
        # Validate directories
        data_dir = self.config.get("data_dir")
        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
            
        ground_truth_file = self.config.get("ground_truth_file")
        if not os.path.exists(ground_truth_file):
            raise ValueError(f"Ground truth file does not exist: {ground_truth_file}")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set environment-specific defaults
        mode = config.get("mode", "dev")
        
        if mode == "dev":
            config.setdefault("max_records", 10000)
            config.setdefault("checkpoint_frequency", 10)
            config.setdefault("embedding_batch_size", 50)
            config.setdefault("weaviate_batch_size", 100)
            config.setdefault("max_neighbors", 20)
            config.setdefault("max_llm_requests", 50)
            config.setdefault("graph_batch_size", 1000)  # For memory-efficient graph building
        else:  # production mode
            config.setdefault("max_records", None)  # Process all records
            config.setdefault("checkpoint_frequency", 50)
            config.setdefault("embedding_batch_size", 100)
            config.setdefault("weaviate_batch_size", 500)
            config.setdefault("max_neighbors", 50)
            config.setdefault("max_llm_requests", 1000)
            config.setdefault("graph_batch_size", 5000)  # For memory-efficient graph building
        
        # Add checkpointing settings
        config.setdefault("enable_checkpointing", True)
        config.setdefault("checkpoint_after_stage", True)
        
        return config
    
    def _load_checkpoint_state(self) -> Dict:
        """
        Load checkpoint state from file.
        
        Returns:
            Checkpoint state dictionary
        """
        checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoint_state.json")
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded checkpoint state from {checkpoint_file}")
                return state
            except Exception as e:
                logger.warning(f"Error loading checkpoint state: {str(e)}")
        
        # Initialize fresh state
        return {
            "completed_stages": [],
            "stats": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_checkpoint_state(self) -> None:
        """
        Save checkpoint state to file.
        """
        if not self.config.get("enable_checkpointing"):
            return
            
        checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoint_state.json")
        
        # Update timestamp
        self.checkpoint_state["last_updated"] = datetime.now().isoformat()
        
        # Save state
        with open(checkpoint_file, 'w') as f:
            json.dump(self.checkpoint_state, f, indent=2)
        
        logger.info(f"Saved checkpoint state to {checkpoint_file}")
    
    def _restore_checkpoint_data(self) -> None:
        """
        Restore data from checkpoints.
        """
        completed_stages = self.checkpoint_state.get("completed_stages", [])
        
        # Restore embeddings if embedding stage was completed
        if "embedding_generation" in completed_stages:
            embeddings_file = os.path.join(self.output_dir, "embeddings.npz")
            if os.path.exists(embeddings_file):
                self.embedding_generator.load_embeddings(embeddings_file)
                logger.info(f"Restored embeddings from {embeddings_file}")
        
        # Restore classifier if training stage was completed
        if "classifier_training" in completed_stages:
            model_file = os.path.join(self.output_dir, "classifier_model.pkl")
            scaler_file = os.path.join(self.output_dir, "feature_scaler.pkl")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.classifier.load_model(model_file)
                self.feature_engineer.load_scaler(scaler_file)
                logger.info(f"Restored classifier model and scaler")
        
        # Restore record embeddings if indexing stage was completed
        if "record_indexing" in completed_stages:
            embeddings_file = os.path.join(self.checkpoint_dir, "record_embeddings.pkl")
            if os.path.exists(embeddings_file):
                try:
                    with open(embeddings_file, 'rb') as f:
                        self.record_embeddings = pickle.load(f)
                    logger.info(f"Restored record embeddings from {embeddings_file}")
                except Exception as e:
                    logger.warning(f"Error loading record embeddings: {str(e)}")
    
    def _mark_stage_complete(self, stage_name: str) -> None:
        """
        Mark a pipeline stage as complete in the checkpoint state.
        
        Args:
            stage_name: Name of the completed stage
        """
        if stage_name not in self.checkpoint_state.get("completed_stages", []):
            completed_stages = self.checkpoint_state.get("completed_stages", [])
            completed_stages.append(stage_name)
            self.checkpoint_state["completed_stages"] = completed_stages
            
            # Update stats
            self.checkpoint_state["stats"] = self.stats
            
            # Save the updated state
            self._save_checkpoint_state()
    
    def _is_stage_completed(self, stage_name: str) -> bool:
        """
        Check if a pipeline stage is marked as complete.
        
        Args:
            stage_name: Name of the stage to check
            
        Returns:
            Boolean indicating if the stage is complete
        """
        return stage_name in self.checkpoint_state.get("completed_stages", [])
    
    def _save_record_embeddings(self) -> None:
        """
        Save record embeddings to checkpoint.
        """
        if not self.config.get("enable_checkpointing") or not self.record_embeddings:
            return
            
        embeddings_file = os.path.join(self.checkpoint_dir, "record_embeddings.pkl")
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.record_embeddings, f)
            
        logger.info(f"Saved record embeddings to {embeddings_file}")
        
    def _run_stage_with_checkpoint(self, stage_func, stage_name: str) -> bool:
        """
        Run a pipeline stage with checkpointing.
        
        Args:
            stage_func: Function to run
            stage_name: Name of the stage
            
        Returns:
            Boolean indicating success
        """
        # Skip if already completed (unless forced)
        if self._is_stage_completed(stage_name) and not self.config.get("force_rerun", False):
            logger.info(f"Skipping completed stage: {stage_name}")
            return True
            
        # Run the stage
        result = self._run_stage(stage_func, stage_name)
        
        # Mark as complete if successful
        if result:
            self._mark_stage_complete(stage_name)
            
        return result
    
    def preprocess_data(self) -> None:
        """
        Preprocess data files.
        """
        logger.info("Starting preprocessing")
        
        mode = self.config.get("mode", "dev")
        self.data_processor.run_preprocessing(mode)
        
        # Save stats
        self.stats["preprocessing"] = {
            "unique_strings": len(self.data_processor.deduplicator.unique_strings),
            "total_records": len(self.data_processor.deduplicator.record_field_hashes)
        }
    
    def generate_embeddings(self) -> None:
        """
        Generate embeddings for unique strings.
        """
        logger.info("Generating embeddings")
        
        # Try to load from cache first
        embedding_cache = os.path.join(self.output_dir, "embeddings.npz")
        if os.path.exists(embedding_cache) and self.config.get("use_cached_embeddings", True):
            logger.info("Loading embeddings from cache")
            self.embedding_generator.load_embeddings(embedding_cache)
        else:
            # Generate new embeddings
            unique_strings = self.data_processor.deduplicator.unique_strings
            
            embeddings = self.embedding_generator.generate_embeddings_for_unique_strings(unique_strings)
            self.embedding_generator.save_embeddings(embedding_cache)
        
        # Save stats
        self.stats["embeddings"] = {
            "total_embeddings": len(self.embedding_generator.embeddings)
        }
    
    def setup_weaviate(self) -> None:
        """
        Set up and connect to Weaviate.
        """
        logger.info("Setting up Weaviate")
        
        connected = self.weaviate_manager.connect()
        if not connected:
            raise RuntimeError("Failed to connect to Weaviate")
            
        setup_success = self.weaviate_manager.setup_collections()
        if not setup_success:
            raise RuntimeError("Failed to set up Weaviate collections")
    
    def index_records(self) -> None:
        """
        Index records in Weaviate with improved efficiency.
        """
        logger.info("Indexing records in Weaviate")
        
        # Create an index of record_id to file_path first for efficient lookup
        record_file_map = {}
        logger.info("Building record ID to file mapping...")
        for file_name in os.listdir(self.config.get("data_dir")):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.config.get("data_dir"), file_name)
                try:
                    # Only read the ID column for mapping
                    df_ids = pd.read_csv(file_path, sep='\t', usecols=['id'], dtype=str, on_bad_lines='warn')
                    for record_id in df_ids['id']:
                        record_file_map[record_id] = file_path
                except Exception as e:
                    logger.warning(f"Error reading IDs from {file_path}: {str(e)}")
        
        logger.info(f"Built mapping for {len(record_file_map)} record IDs")
        
        # Prepare records and their embeddings
        records = []
        record_embeddings = {}
        
        # Process each record with efficient lookup
        for record_id, field_hashes in tqdm(self.data_processor.deduplicator.record_field_hashes.items(), 
                                           desc="Preparing records"):
            if record_id in record_file_map:
                file_path = record_file_map[record_id]
                try:
                    df = pd.read_csv(file_path, sep='\t', dtype=str, on_bad_lines='warn')
                    record_rows = df[df['id'] == record_id]
                    if not record_rows.empty:
                        original_record = record_rows.iloc[0].to_dict()
                        
                        # Get embeddings for each field
                        field_embeddings = {}
                        for field, hash_value in field_hashes.items():
                            if hash_value != "NULL" and hash_value in self.embedding_generator.embeddings:
                                field_embeddings[field] = self.embedding_generator.embeddings[hash_value]
                        
                        # Store record and embeddings
                        records.append(original_record)
                        record_embeddings[record_id] = field_embeddings
                except Exception as e:
                    logger.warning(f"Error retrieving record {record_id}: {str(e)}")
            
            # Check if we've reached the record limit (for dev mode)
            max_records = self.config.get("max_records")
            if max_records and len(records) >= max_records:
                logger.info(f"Reached maximum records limit: {max_records}")
                break
        
        # Index records in batches
        success = self.weaviate_manager.batch_index_records(records, record_embeddings)
        if not success:
            logger.error("Failed to index records in Weaviate")
            raise RuntimeError("Record indexing failed")
        
        # Store record embeddings for later use
        self.record_embeddings = record_embeddings
        
        # Save stats
        self.stats["indexing"] = {
            "records_indexed": len(records)
        }
    
    def prepare_training_data(self) -> Tuple[List[Dict], List[Dict], List[Tuple]]:
        """
        Prepare data for classifier training.
        
        Returns:
            Tuple of (train_records, test_records, ground_truth_pairs)
        """
        logger.info("Preparing training data")
        
        # Parse ground truth file
        ground_truth_file = self.config.get("ground_truth_file")
        ground_truth_pairs = self.data_processor.parse_ground_truth(ground_truth_file)
        
        # Collect records for training
        record_ids = set()
        for left_id, right_id, _ in ground_truth_pairs:
            record_ids.add(left_id)
            record_ids.add(right_id)
        
        # Retrieve records
        all_records = {}
        for record_id in record_ids:
            record = self.weaviate_manager.get_record_by_id(record_id)
            if record:
                all_records[record_id] = record
        
        # Create train/test split
        from sklearn.model_selection import train_test_split
        
        train_pairs, test_pairs = train_test_split(
            ground_truth_pairs,
            test_size=0.2,
            random_state=42,
            stratify=[int(is_match) for _, _, is_match in ground_truth_pairs]
        )
        
        train_ids = set()
        for left_id, right_id, _ in train_pairs:
            train_ids.add(left_id)
            train_ids.add(right_id)
            
        test_ids = set()
        for left_id, right_id, _ in test_pairs:
            test_ids.add(left_id)
            test_ids.add(right_id)
            
        train_records = {rid: all_records[rid] for rid in train_ids if rid in all_records}
        test_records = {rid: all_records[rid] for rid in test_ids if rid in all_records}
        
        logger.info(f"Prepared {len(train_pairs)} training pairs, {len(test_pairs)} test pairs")
        
        # Save stats
        self.stats["training_data"] = {
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
            "train_records": len(train_records),
            "test_records": len(test_records)
        }
        
        return train_records, test_records, ground_truth_pairs
    
    def train_classifier(self, train_records: Dict[str, Dict], test_records: Dict[str, Dict], 
                        ground_truth_pairs: List[Tuple]) -> None:
        """
        Train and evaluate the classifier.
        
        Args:
            train_records: Dictionary of training records
            test_records: Dictionary of test records
            ground_truth_pairs: List of ground truth pairs
        """
        logger.info("Training classifier")
        
        # Create training pairs
        train_pairs = []
        for left_id, right_id, is_match in ground_truth_pairs:
            if left_id in train_records and right_id in train_records:
                train_pairs.append((train_records[left_id], train_records[right_id], is_match))
        
        # Prepare training data
        X_train, y_train, feature_names = self.feature_engineer.prepare_training_data(
            train_pairs, self.record_embeddings
        )
        
        # Train the classifier
        training_stats = self.classifier.train(X_train, y_train, feature_names)
        
        # Save the trained model
        self.classifier.save_model(os.path.join(self.output_dir, "classifier_model.pkl"))
        self.feature_engineer.save_scaler(os.path.join(self.output_dir, "feature_scaler.pkl"))
        
        # Create test pairs
        test_pairs = []
        for left_id, right_id, is_match in ground_truth_pairs:
            if left_id in test_records and right_id in test_records:
                test_pairs.append((test_records[left_id], test_records[right_id], is_match))
        
        # Prepare test data
        X_test, y_test, _ = self.feature_engineer.prepare_training_data(
            test_pairs, self.record_embeddings
        )
        
        # Evaluate on test data
        evaluation_stats = self.classifier.evaluate(X_test, y_test)
        
        logger.info(f"Classifier training complete. Test accuracy: {evaluation_stats['accuracy']:.4f}")
        
        # Save stats
        self.stats["classifier"] = {
            "training": training_stats,
            "evaluation": evaluation_stats
        }
    
    def initialize_imputer_clusterer(self) -> None:
        """
        Initialize the imputer and clusterer components.
        """
        logger.info("Initializing imputer and clusterer")
        
        self.imputer = NullValueImputer(self.config, self.weaviate_manager)
        self.clusterer = EntityClusterer(
            self.config, self.classifier, self.weaviate_manager, 
            self.imputer, self.feature_engineer
        )
    
    def run_clustering(self) -> List[Dict]:
        """
        Run the entity clustering pipeline.
        
        Returns:
            List of entity clusters
        """
        logger.info("Running entity clustering")
        
        # Get all records
        records = []
        for record_id in self.record_embeddings.keys():
            record = self.weaviate_manager.get_record_by_id(record_id)
            if record:
                records.append(record)
        
        # Build match graph
        match_graph = self.clusterer.build_match_graph(records, self.record_embeddings)
        
        # Extract clusters
        clusters = self.clusterer.extract_clusters(match_graph)
        
        # Save clusters
        self.clusterer.save_clusters(clusters, os.path.join(self.output_dir, "entity_clusters.jsonl"))
        
        # Save stats
        self.stats["clustering"] = {
            "total_clusters": len(clusters),
            "total_records_clustered": sum(cluster["size"] for cluster in clusters),
            "average_cluster_size": sum(cluster["size"] for cluster in clusters) / len(clusters) if clusters else 0,
            "average_confidence": sum(cluster["confidence"] for cluster in clusters) / len(clusters) if clusters else 0,
            "llm_requests_made": self.clusterer.llm_requests_made
        }
        
        return clusters
    
    def run_pipeline(self) -> None:
        """
        Run the complete entity resolution pipeline with improved error handling and checkpointing.
        """
        self.start_time = time.time()
        
        try:
            logger.info("Starting entity resolution pipeline")
            logger.info(f"Mode: {self.config.get('mode', 'dev')}")
            logger.info(f"Checkpointing enabled: {self.config.get('enable_checkpointing', True)}")
            
            # Run stages with checkpointing
            if not self._run_stage_with_checkpoint(self.preprocess_data, "preprocessing"):
                return
                
            if not self._run_stage_with_checkpoint(self.generate_embeddings, "embedding_generation"):
                return
                
            if not self._run_stage_with_checkpoint(self.setup_weaviate, "weaviate_setup"):
                return
                
            if not self._run_stage_with_checkpoint(self.index_records, "record_indexing"):
                return
                
            # Save record embeddings for checkpointing
            self._save_record_embeddings()
            
            train_records, test_records, ground_truth_pairs = None, None, None
            
            # Use a lambda to capture the return values
            training_data_func = lambda: self._prepare_training_data_wrapper()
            if not self._run_stage_with_checkpoint(training_data_func, "training_data_preparation"):
                return
                
            train_records, test_records, ground_truth_pairs = self._prepare_training_data_wrapper()
                
            # Use another lambda to pass the arguments
            classifier_func = lambda: self.train_classifier(train_records, test_records, ground_truth_pairs)
            if not self._run_stage_with_checkpoint(classifier_func, "classifier_training"):
                return
                
            if not self._run_stage_with_checkpoint(self.initialize_imputer_clusterer, "component_initialization"):
                return
                
            if not self._run_stage_with_checkpoint(self.run_clustering, "entity_clustering"):
                return
                
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Cleanup resources
            self._cleanup()
            
            # Save overall stats
            elapsed_time = time.time() - self.start_time
            self.stats["overall"] = {
                "elapsed_time": elapsed_time,
                "records_processed": len(self.record_embeddings) if hasattr(self, 'record_embeddings') else 0,
                "pipeline_run_date": datetime.now().isoformat()
            }
            
            # Update checkpoint state with final stats
            self.checkpoint_state["stats"] = self.stats
            self._save_checkpoint_state()
            
            # Log completion
            logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
            
            # Save stats to file
            with open(os.path.join(self.output_dir, "pipeline_stats.json"), 'w') as f:
                json.dump(self.stats, f, indent=2)
    
    def _prepare_training_data_wrapper(self) -> Tuple[Dict, Dict, List]:
        """
        Wrapper for prepare_training_data that returns the results.
        
        Returns:
            Tuple of (train_records, test_records, ground_truth_pairs)
        """
        result = self.prepare_training_data()
        return result
    
    def _run_stage(self, stage_func, stage_name: str) -> bool:
        """
        Run a pipeline stage with error handling.
        
        Args:
            stage_func: Function to run
            stage_name: Name of the stage
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Starting stage: {stage_name}")
        stage_start = time.time()
        
        try:
            result = stage_func()
            stage_time = time.time() - stage_start
            logger.info(f"Completed stage: {stage_name} in {stage_time:.2f} seconds")
            
            # Save stage-specific stats
            self.stats[stage_name] = {
                "execution_time": stage_time,
                "completed_at": datetime.now().isoformat()
            }
            
            return True
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Record failure in stats
            self.stats[stage_name] = {
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
            
            return False
    
    def _cleanup(self) -> None:
        """
        Clean up resources.
        """
        # Close any open file handles
        
        # Remove temporary files if configured
        if self.config.get("cleanup_temp_files", True):
            temp_dirs = ["temp"]  # Don't remove checkpoints
            for temp_dir in temp_dirs:
                temp_path = os.path.join(self.output_dir, temp_dir)
                if os.path.exists(temp_path):
                    import shutil
                    shutil.rmtree(temp_path)
                    logger.info(f"Removed temporary directory: {temp_path}")