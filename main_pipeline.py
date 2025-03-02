import os
import json
import logging
import argparse
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
import time
from datetime import datetime
import yaml
import pickle
from logging_setup import setup_logging  # Import the new logging setup

# Set up logging with proper handler management
logger = setup_logging()

# Import modules
from preprocessing import DataProcessor, TextDeduplicator
from embedding import EmbeddingGenerator
from weaviate_integration import WeaviateManager
from feature_engineering import FeatureEngineer, LogisticRegressionClassifier
from imputation_clustering import NullValueImputer, EntityClusterer

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
        
        logger.info(f"Pipeline initialized with config from {config_path}")
        logger.info(f"Mode: {self.config.get('mode', 'unknown')}")
        logger.info(f"Output directory: {self.output_dir}")
    
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
        Preprocess data files with explicit checkpoint saving.
        """
        logger.info("Starting preprocessing")
        
        mode = self.config.get("mode", "dev")
        
        # Make sure checkpoint directory exists
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Update checkpoint path in config to ensure it's correctly used
        self.config["checkpoint_dir"] = checkpoint_dir
        
        # Also set the checkpoint_dir in the deduplicator's config
        self.data_processor.deduplicator.config["checkpoint_dir"] = checkpoint_dir
        
        # Run preprocessing
        self.data_processor.run_preprocessing(mode)
        
        # Explicitly save checkpoint files after preprocessing
        logger.info("Explicitly saving checkpoint files")
        save_success = self.data_processor.deduplicator.save_checkpoint()
        
        if save_success:
            logger.info("Checkpoint files saved successfully")
        else:
            logger.error("Failed to save checkpoint files")
            raise RuntimeError("Failed to save preprocessing results to checkpoint files")
        
        # Verify checkpoint files were created
        required_files = [
            "unique_strings.json",
            "string_counts.json", 
            "field_types.json", 
            "record_field_hashes.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(checkpoint_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"Some required checkpoint files are missing: {missing_files}")
            raise RuntimeError("Preprocessing completed but checkpoint files are missing")
        
        # Save stats
        self.stats["preprocessing"] = {
            "unique_strings": len(self.data_processor.deduplicator.unique_strings),
            "total_records": len(self.data_processor.deduplicator.record_field_hashes),
            "checkpoint_files_created": len(required_files) - len(missing_files),
            "checkpoint_dir": checkpoint_dir
        }
        
        logger.info(f"Preprocessing complete: {len(self.data_processor.deduplicator.unique_strings)} unique strings from {len(self.data_processor.deduplicator.record_field_hashes)} records")
    
    def generate_embeddings(self) -> None:
        """
        Generate embeddings for unique strings with direct checkpoint file access.
        This version bypasses the data_processor loading mechanism for better reliability.
        """
        logger.info("Starting embedding generation with direct checkpoint access")
        
        # Define checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        
        # Direct loading of unique strings from file
        unique_strings_path = os.path.join(checkpoint_dir, "unique_strings.json")
        if not os.path.exists(unique_strings_path):
            logger.error(f"Unique strings file not found: {unique_strings_path}")
            raise FileNotFoundError(f"Unique strings file not found: {unique_strings_path}")
        
        try:
            logger.info(f"Loading unique strings directly from: {unique_strings_path}")
            with open(unique_strings_path, 'r') as f:
                unique_strings = json.load(f)
            logger.info(f"Loaded {len(unique_strings)} unique strings directly from file")
        except Exception as e:
            logger.error(f"Error loading unique strings file: {str(e)}")
            raise RuntimeError(f"Failed to load unique strings: {str(e)}")
        
        if not unique_strings:
            logger.error("No unique strings found in file. Preprocessing may have failed.")
            raise ValueError("Empty unique strings file. Run preprocessing again.")
        
        # Try to load existing embeddings from cache
        embedding_cache = os.path.join(self.output_dir, "embeddings.npz")
        cache_loaded = False
        
        if os.path.exists(embedding_cache) and self.config.get("use_cached_embeddings", True):
            logger.info(f"Loading embeddings from cache: {embedding_cache}")
            cache_loaded = self.embedding_generator.load_embeddings(embedding_cache)
            logger.info(f"Loaded {len(self.embedding_generator.embeddings)} embeddings from cache")
        
        # Determine which strings need embedding
        missing_strings = {}
        for hash_key, string_value in unique_strings.items():
            if hash_key not in self.embedding_generator.embeddings:
                missing_strings[hash_key] = string_value
        
        logger.info(f"Need to generate embeddings for {len(missing_strings)} strings")
        
        if missing_strings:
            # Validate OpenAI API key
            if not self.check_openai_api_key():
                logger.error("Cannot generate embeddings without a valid OpenAI API key")
                raise ValueError("Missing or invalid OpenAI API key")
            
            # Generate new embeddings
            logger.info(f"Generating embeddings using OpenAI API")
            try:
                # Add some sample strings for debugging
                sample_keys = list(missing_strings.keys())[:3]
                if sample_keys:
                    logger.info(f"Sample strings to embed (first 3):")
                    for key in sample_keys:
                        logger.info(f"  - {key[:8]}: {missing_strings[key][:50]}...")
                
                # Actually generate the embeddings
                new_embeddings = self.embedding_generator.generate_embeddings_for_unique_strings(missing_strings)
                logger.info(f"Successfully generated {len(new_embeddings)} new embeddings")
                
                # Save updated embeddings
                self.embedding_generator.save_embeddings(embedding_cache)
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        elif not self.embedding_generator.embeddings:
            logger.warning("No embeddings loaded from cache and no new strings to embed")
        else:
            logger.info("All strings already have embeddings in cache, no need to generate new ones")
    
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
        Index string data in Weaviate with the string-centric approach.
        """
        logger.info("Indexing string data in Weaviate")
        
        # If required, clear existing data
        if self.config.get("clear_existing_data", False):
            self.weaviate_manager.clear_collections()
        
        # Prepare string data for indexing
        unique_strings = self.data_processor.deduplicator.unique_strings
        string_counts = self.data_processor.deduplicator.string_counts
        
        logger.info(f"Prepared {len(unique_strings)} unique strings for indexing")
        
        # Get field type mapping (hash -> field type)
        field_types = self.data_processor.deduplicator.field_types
        
        # Ensure all strings have field types (fallback to field mappings if needed)
        if len(field_types) < len(unique_strings):
            logger.warning(f"Field types incomplete ({len(field_types)} vs {len(unique_strings)} strings), generating missing mappings")
            for record_id, field_hash_map in self.data_processor.deduplicator.record_field_hashes.items():
                for field, hash_value in field_hash_map.items():
                    if hash_value != "NULL" and hash_value not in field_types:
                        field_types[hash_value] = field
        
        logger.info(f"Using {len(field_types)} field type mappings")
        
        # Index unique strings with their vectors
        success = self.weaviate_manager.index_unique_strings(
            unique_strings=unique_strings,
            embeddings=self.embedding_generator.embeddings,
            string_counts=string_counts,
            field_types=field_types
        )
        
        if not success:
            logger.error("Failed to index unique strings in Weaviate")
            raise RuntimeError("String indexing failed")
        
        # Prepare entity maps
        entity_maps = []
        for record_id, field_hashes in self.data_processor.deduplicator.record_field_hashes.items():
            person_name = ""
            person_hash = field_hashes.get("person")
            if person_hash != "NULL":
                person_name = unique_strings.get(person_hash, "")
                
            entity_maps.append({
                "entity_id": record_id,
                "field_hashes": field_hashes,
                "person_name": person_name
            })
            
            # Limit number of entities in development mode
            max_records = self.config.get("max_records")
            if max_records and len(entity_maps) >= max_records:
                logger.info(f"Reached maximum records limit: {max_records}")
                break
        
        # Index entity maps (optional)
        if self.config.get("index_entity_maps", True):
            success = self.weaviate_manager.index_entity_maps(entity_maps)
            if not success:
                logger.warning("Failed to index entity maps (non-critical)")
        
        # Retrieve field vectors for all records
        logger.info("Retrieving field vectors for records")
        records_to_process = {}
        for record_id, field_hashes in self.data_processor.deduplicator.record_field_hashes.items():
            # Apply record limit
            if max_records and len(records_to_process) >= max_records:
                break
                
            records_to_process[record_id] = field_hashes
        
        # Batch retrieve field vectors
        self.record_embeddings = self.weaviate_manager.get_field_vectors_for_records(records_to_process)
        
        # Save retrieved embeddings for later use
        self._save_record_embeddings()
        
        # Save stats
        self.stats["indexing"] = {
            "unique_strings_indexed": len(unique_strings),
            "entity_maps_indexed": len(entity_maps),
            "records_processed": len(self.record_embeddings)
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
        Train and evaluate the classifier, and perform clustering on the training data.
        
        Args:
            train_records: Dictionary of training records
            test_records: Dictionary of test records
            ground_truth_pairs: List of ground truth pairs
        """
        logger.info("Training classifier")
        
        if not train_records or not test_records:
            logger.error("Training or test records are empty, cannot train classifier")
            return
        
        # Create training pairs
        train_pairs = []
        for left_id, right_id, is_match in ground_truth_pairs:
            if left_id in train_records and right_id in train_records:
                train_pairs.append((train_records[left_id], train_records[right_id], is_match))
        
        if not train_pairs:
            logger.error("No valid training pairs available")
            return
        
        logger.info(f"Created {len(train_pairs)} training pairs")
        
        # Prepare training data
        try:
            X_train, y_train, feature_names = self.feature_engineer.prepare_training_data(
                train_pairs, self.record_embeddings
            )
            
            if len(X_train) == 0 or len(y_train) == 0:
                logger.error("Feature engineering resulted in empty training data")
                return
                
            logger.info(f"Prepared training data with {len(X_train)} examples and {len(feature_names)} features")
            
            # Check class balance in training data
            pos_count = sum(y_train)
            neg_count = len(y_train) - pos_count
            logger.info(f"Training data class balance: {pos_count} positives, {neg_count} negatives")
            
            if pos_count == 0 or neg_count == 0:
                logger.error("Training data has only one class, cannot train classifier")
                return
        
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
            
            if not test_pairs:
                logger.warning("No valid test pairs available, skipping evaluation")
                return
                
            logger.info(f"Created {len(test_pairs)} test pairs")
            
            # Prepare test data
            X_test, y_test, _ = self.feature_engineer.prepare_training_data(
                test_pairs, self.record_embeddings
            )
            
            if len(X_test) == 0 or len(y_test) == 0:
                logger.warning("Empty test data, skipping evaluation")
                return
            
            # Evaluate on test data
            evaluation_stats = self.classifier.evaluate(X_test, y_test)
            
            logger.info(f"Classifier training complete. Test accuracy: {evaluation_stats['accuracy']:.4f}, " 
                    f"Precision: {evaluation_stats['precision']:.4f}, "
                    f"Recall: {evaluation_stats['recall']:.4f}, "
                    f"F1: {evaluation_stats['f1_score']:.4f}")
            
            # Initialize components needed for clustering (temporary for training data only)
            self._initialize_components_for_training()
            
            # Perform clustering on training data
            training_clusters = self._cluster_training_data(train_records)
            
            # Evaluate clustering against ground truth
            clustering_evaluation = self._evaluate_training_clusters(training_clusters, ground_truth_pairs)
            
            # Save stats
            self.stats["classifier"] = {
                "training": training_stats,
                "evaluation": evaluation_stats,
                "clustering": clustering_evaluation
            }
        except Exception as e:
            logger.error(f"Error during classifier training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
    def _initialize_components_for_training(self) -> None:
        """
        Initialize imputer and clusterer components for training data clustering.
        This is separate from the component initialization for the full dataset.
        """
        logger.info("Initializing components for training data clustering")
        
        try:
            # These are temporary instances just for training evaluation
            self.training_imputer = NullValueImputer(self.config, self.weaviate_manager)
            self.training_clusterer = EntityClusterer(
                self.config, self.classifier, self.weaviate_manager, 
                self.training_imputer, self.feature_engineer
            )
            logger.info("Successfully initialized training components")
        except Exception as e:
            logger.error(f"Error initializing training components: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
    def _cluster_training_data(self, train_records: Dict[str, Dict]) -> List[Dict]:
        """
        Apply the clustering process to the training data.
        
        Args:
            train_records: Dictionary of training records
            
        Returns:
            List of clusters from training data
        """
        logger.info("Clustering training data")
        
        # Convert dictionary to list
        records_list = list(train_records.values())
        
        # Build match graph for training data
        train_graph = self.training_clusterer.build_match_graph(records_list, self.record_embeddings)
        
        # Extract clusters
        train_clusters = self.training_clusterer.extract_clusters(train_graph)
        
        # Save training clusters for analysis
        self.training_clusterer.save_clusters(
            train_clusters, 
            os.path.join(self.output_dir, "training_clusters.jsonl")
        )
        
        logger.info(f"Created {len(train_clusters)} clusters from training data")
        return train_clusters
    
    def _evaluate_training_clusters(self, clusters: List[Dict], ground_truth_pairs: List[Tuple]) -> Dict:
        """
        Evaluate training data clustering against ground truth.
        
        Args:
            clusters: List of clusters from training data
            ground_truth_pairs: List of ground truth pairs (left_id, right_id, is_match)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating training clusters against ground truth")
        
        # Build a mapping of record_id to cluster_id
        record_to_cluster = {}
        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            for record_id in cluster["records"]:
                record_to_cluster[record_id] = cluster_id
        
        # Count metrics
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Compare clustering results with ground truth
        for left_id, right_id, is_match in ground_truth_pairs:
            # Skip if either record is not in clustering results
            if left_id not in record_to_cluster or right_id not in record_to_cluster:
                continue
                
            same_cluster = record_to_cluster[left_id] == record_to_cluster[right_id]
            
            if is_match and same_cluster:
                true_positives += 1
            elif is_match and not same_cluster:
                false_negatives += 1
            elif not is_match and same_cluster:
                false_positives += 1
            elif not is_match and not same_cluster:
                true_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        evaluation = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }
        
        logger.info(f"Training clusters evaluation: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        return evaluation
    
    def initialize_imputer_clusterer(self) -> None:
        """
        Initialize the imputer and clusterer components for the full pipeline.
        """
        logger.info("Initializing imputer and clusterer for full dataset")
        
        try:
            # Make sure the components don't already exist
            if hasattr(self, 'imputer') and self.imputer is not None:
                logger.info("Imputer already initialized, skipping")
            else:
                self.imputer = NullValueImputer(self.config, self.weaviate_manager)
                logger.info("Initialized imputer")
                
            if hasattr(self, 'clusterer') and self.clusterer is not None:
                logger.info("Clusterer already initialized, skipping")
            else:
                self.clusterer = EntityClusterer(
                    self.config, self.classifier, self.weaviate_manager, 
                    self.imputer, self.feature_engineer
                )
                logger.info("Initialized clusterer")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize components: {str(e)}")
    
    def run_clustering(self) -> List[Dict]:
        """
        Run the entity clustering pipeline on the full dataset.
        Clearly separated from the model training stage.
        
        Returns:
            List of entity clusters
        """
        logger.info("Running entity clustering on full dataset")
        
        # Check if classifier is trained
        if not hasattr(self, 'classifier') or self.classifier.weights is None:
            logger.error("Classifier is not trained, cannot perform clustering")
            return []
        
        # Check if components are initialized
        if not hasattr(self, 'imputer') or not hasattr(self, 'clusterer'):
            logger.error("Imputer and clusterer are not initialized, cannot perform clustering")
            return []
        
        # If we don't have vectors for the full dataset yet, retrieve them
        if not hasattr(self, 'record_embeddings') or len(self.record_embeddings) < len(self.data_processor.deduplicator.record_field_hashes):
            self._retrieve_vectors_for_full_dataset()
        
        # Check if we have embeddings
        if not self.record_embeddings:
            logger.error("No record embeddings available, cannot perform clustering")
            return []
        
        # Get all records
        records = []
        total_records = len(self.record_embeddings)
        processed_count = 0
        error_count = 0
        
        logger.info(f"Reconstructing {total_records} records for clustering")
        
        for record_id in self.record_embeddings.keys():
            field_hashes = self.data_processor.deduplicator.record_field_hashes.get(record_id)
            if field_hashes:
                record = self._reconstruct_record_from_hashes(record_id, field_hashes)
                if record:
                    records.append(record)
                    processed_count += 1
                else:
                    error_count += 1
            else:
                logger.warning(f"No field hashes found for record ID: {record_id}")
                error_count += 1
                
            # Log progress for large datasets
            if (processed_count + error_count) % 1000 == 0:
                logger.info(f"Reconstructed {processed_count}/{total_records} records ({error_count} errors)")
        
        if not records:
            logger.error("No valid records reconstructed, cannot perform clustering")
            return []
        
        logger.info(f"Clustering {len(records)} records from full dataset (skipped {error_count} records with errors)")
        
        try:
            # Build match graph
            match_graph = self.clusterer.build_match_graph(records, self.record_embeddings)
            
            if not match_graph.nodes:
                logger.error("Match graph has no nodes, clustering failed")
                return []
                
            if not match_graph.edges:
                logger.warning("Match graph has no edges, all records will be single-entity clusters")
            
            logger.info(f"Built match graph with {len(match_graph.nodes)} nodes and {len(match_graph.edges)} edges")
            
            # Extract clusters
            clusters = self.clusterer.extract_clusters(match_graph)
            
            if not clusters:
                logger.warning("No clusters were formed during extraction")
            
            logger.info(f"Extracted {len(clusters)} clusters from match graph")
            
            # Calculate cluster size statistics
            if clusters:
                sizes = [cluster["size"] for cluster in clusters]
                avg_size = sum(sizes) / len(clusters)
                max_size = max(sizes)
                min_size = min(sizes)
                
                logger.info(f"Cluster sizes - Min: {min_size}, Avg: {avg_size:.2f}, Max: {max_size}")
                
                # Count clusters by size range
                size_ranges = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, "21-50": 0, "51+": 0}
                for size in sizes:
                    if size == 1:
                        size_ranges["1"] += 1
                    elif 2 <= size <= 5:
                        size_ranges["2-5"] += 1
                    elif 6 <= size <= 10:
                        size_ranges["6-10"] += 1
                    elif 11 <= size <= 20:
                        size_ranges["11-20"] += 1
                    elif 21 <= size <= 50:
                        size_ranges["21-50"] += 1
                    else:
                        size_ranges["51+"] += 1
                        
                logger.info(f"Cluster size distribution: {size_ranges}")
            
            # Save clusters
            output_file = os.path.join(self.output_dir, "entity_clusters.jsonl")
            self.clusterer.save_clusters(clusters, output_file)
            logger.info(f"Saved clusters to {output_file}")
            
            # Save stats
            self.stats["clustering"] = {
                "total_clusters": len(clusters),
                "records_processed": len(records),
                "total_records": total_records,
                "records_with_errors": error_count,
                "total_records_clustered": sum(cluster["size"] for cluster in clusters),
                "average_cluster_size": sum(cluster["size"] for cluster in clusters) / len(clusters) if clusters else 0,
                "average_confidence": sum(cluster["confidence"] for cluster in clusters) / len(clusters) if clusters else 0,
                "cluster_size_distribution": size_ranges if clusters else {},
                "llm_requests_made": self.clusterer.llm_requests_made
            }
            
            return clusters
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def run_pipeline(self) -> None:
        """
        Run the complete entity resolution pipeline with improved separation of stages.
        """
        self.start_time = time.time()
        
        try:
            logger.info("Starting entity resolution pipeline")
            logger.info(f"Mode: {self.config.get('mode', 'dev')}")
            logger.info(f"Data directory: {self.config.get('data_dir')}")
            logger.info(f"Ground truth file: {self.config.get('ground_truth_file')}")
            logger.info(f"Checkpointing enabled: {self.config.get('enable_checkpointing', True)}")
            
            # Validate critical directories and files
            data_dir = self.config.get('data_dir')
            if not os.path.exists(data_dir):
                logger.error(f"Data directory does not exist: {data_dir}")
                return
                
            ground_truth_file = self.config.get('ground_truth_file')
            if not os.path.exists(ground_truth_file):
                logger.error(f"Ground truth file does not exist: {ground_truth_file}")
                return
                
            # Phase 1: Data Processing & Indexing
            # These stages must be run before any classification/clustering
            
            # Stage 1: Preprocessing - Process all data to extract unique strings
            if not self._run_stage_with_checkpoint(self.preprocess_data, "preprocessing"):
                logger.error("Preprocessing stage failed. Cannot continue pipeline.")
                return
                
            # Validate preprocessing results
            if not self.data_processor.deduplicator.record_field_hashes:
                logger.error("No records were processed during preprocessing. Check your data files.")
                return
                
            logger.info(f"Preprocessing completed with {len(self.data_processor.deduplicator.record_field_hashes)} records")
                    
            # Stage 2: Embedding Generation - Create vectors for all unique strings
            if not self._run_stage_with_checkpoint(self.generate_embeddings, "embedding_generation"):
                logger.error("Embedding generation stage failed. Cannot continue pipeline.")
                return
                
            # Validate embedding results
            if not self.embedding_generator.embeddings:
                logger.error("No embeddings were generated. Check preprocessing outputs.")
                return
                
            logger.info(f"Embedding generation completed with {len(self.embedding_generator.embeddings)} embeddings")
                    
            # Stage 3: Weaviate Setup - Configure vector database
            if not self._run_stage_with_checkpoint(self.setup_weaviate, "weaviate_setup"):
                logger.error("Weaviate setup stage failed. Cannot continue pipeline.")
                return
                    
            # Stage 4: String Indexing - Index all strings and retrieve vectors
            if not self._run_stage_with_checkpoint(self.index_strings, "string_indexing"):
                logger.error("String indexing stage failed. Cannot continue pipeline.")
                return
                
            # Validate indexing results
            if not hasattr(self, 'record_embeddings') or not self.record_embeddings:
                logger.error("No record embeddings were created during indexing. Check Weaviate connection.")
                return
                
            logger.info(f"Indexing completed with {len(self.record_embeddings)} record embeddings")
            
            # Phase 2: Model Training with Training Data Clustering
            # Train classification model and cluster training dataset for evaluation
            
            # Stage 5: Training Data Preparation
            training_data_func = lambda: self._prepare_training_data_wrapper()
            if not self._run_stage_with_checkpoint(training_data_func, "training_data_preparation"):
                logger.error("Training data preparation stage failed. Cannot continue pipeline.")
                return
                    
            train_records, test_records, ground_truth_pairs = self._prepare_training_data_wrapper()
            
            # Validate training data
            if not train_records or not test_records:
                logger.error("Training or test records are empty. Check ground truth file and preprocessing.")
                return
                
            logger.info(f"Training data preparation completed with {len(train_records)} train records and {len(test_records)} test records")
                    
            # Stage 6: Classifier Training and Training Data Clustering
            classifier_func = lambda: self.train_classifier(train_records, test_records, ground_truth_pairs)
            if not self._run_stage_with_checkpoint(classifier_func, "classifier_training"):
                logger.error("Classifier training stage failed. Cannot continue pipeline.")
                return
                    
            # Phase 3: Full Dataset Classification & Clustering
            # Apply trained model to complete dataset
            
            # Stage 7: Component Initialization for Full Dataset
            if not self._run_stage_with_checkpoint(self.initialize_imputer_clusterer, "component_initialization"):
                logger.error("Component initialization stage failed. Cannot continue pipeline.")
                return
                    
            # Stage 8: Entity Clustering on Complete Dataset
            if not self._run_stage_with_checkpoint(self.run_clustering, "entity_clustering"):
                logger.error("Entity clustering stage failed.")
                return
                
            logger.info("All pipeline stages completed successfully")
                    
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
                
    def index_strings(self) -> None:
        """
        Index string data in Weaviate with the string-centric approach.
        This is now a separate stage from the record indexing.
        """
        logger.info("Indexing string data in Weaviate")
        
        # First, connect to Weaviate
        if not self.weaviate_manager.connect():
            logger.error("Failed to connect to Weaviate")
            raise RuntimeError("Cannot connect to Weaviate")
        
        # If required, clear existing data
        if self.config.get("clear_existing_data", False):
            logger.info("Clearing existing data from Weaviate")
            self.weaviate_manager.clear_collections()
        
        # Load checkpoint data to ensure we have unique strings
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        unique_strings_path = os.path.join(checkpoint_dir, "unique_strings.json")
        
        if not os.path.exists(unique_strings_path):
            logger.error(f"Unique strings file not found: {unique_strings_path}")
            raise FileNotFoundError(f"Unique strings file not found: {unique_strings_path}")
        
        # Load unique strings directly
        try:
            logger.info(f"Loading unique strings from: {unique_strings_path}")
            with open(unique_strings_path, 'r') as f:
                unique_strings = json.load(f)
            logger.info(f"Loaded {len(unique_strings)} unique strings")
        except Exception as e:
            logger.error(f"Error loading unique strings: {str(e)}")
            raise RuntimeError(f"Failed to load unique strings: {str(e)}")
        
        # Load string counts and field types
        string_counts_path = os.path.join(checkpoint_dir, "string_counts.json")
        field_types_path = os.path.join(checkpoint_dir, "field_types.json")
        
        string_counts = {}
        field_types = {}
        
        try:
            if os.path.exists(string_counts_path):
                with open(string_counts_path, 'r') as f:
                    string_counts = json.load(f)
                logger.info(f"Loaded {len(string_counts)} string counts")
            else:
                # Generate default counts (all 1) if file doesn't exist
                logger.warning("String counts file not found, using default counts")
                string_counts = {hash_key: 1 for hash_key in unique_strings.keys()}
            
            if os.path.exists(field_types_path):
                with open(field_types_path, 'r') as f:
                    field_types = json.load(f)
                logger.info(f"Loaded {len(field_types)} field types")
            else:
                # Generate field types from the first field type in record_field_hashes
                logger.warning("Field types file not found, will be inferred")
                record_hashes_path = os.path.join(checkpoint_dir, "record_field_hashes.json")
                
                if os.path.exists(record_hashes_path):
                    with open(record_hashes_path, 'r') as f:
                        record_hashes = json.load(f)
                    
                    # Map hashes to field types
                    for record_id, fields in record_hashes.items():
                        for field, hash_value in fields.items():
                            if hash_value != "NULL" and hash_value not in field_types:
                                field_types[hash_value] = field
                    
                    logger.info(f"Inferred {len(field_types)} field types from record hashes")
                else:
                    # Default to "unknown" field type
                    field_types = {hash_key: "unknown" for hash_key in unique_strings.keys()}
                    logger.warning("Record hashes file not found, using 'unknown' field type")
        except Exception as e:
            logger.error(f"Error loading supporting files: {str(e)}")
            logger.warning("Continuing with default values")
        
        # Load embeddings
        embedding_cache = os.path.join(self.output_dir, "embeddings.npz")
        if not os.path.exists(embedding_cache):
            logger.error(f"Embeddings file not found: {embedding_cache}")
            raise FileNotFoundError(f"Embeddings file not found: {embedding_cache}")
        
        logger.info(f"Loading embeddings from: {embedding_cache}")
        if not self.embedding_generator.load_embeddings(embedding_cache):
            logger.error("Failed to load embeddings")
            raise RuntimeError("Failed to load embeddings")
        
        logger.info(f"Loaded {len(self.embedding_generator.embeddings)} embeddings")
        
        # Ensure embeddings exist for all unique strings
        missing_embeddings = [hash_key for hash_key in unique_strings.keys() 
                            if hash_key not in self.embedding_generator.embeddings]
        
        if missing_embeddings:
            logger.error(f"Missing embeddings for {len(missing_embeddings)} unique strings")
            if len(missing_embeddings) <= 10:
                logger.error(f"Missing embeddings for: {missing_embeddings}")
            else:
                logger.error(f"First 10 missing embeddings: {missing_embeddings[:10]}")
            raise RuntimeError("Not all unique strings have embeddings. Run embedding stage first.")
        
        # Prepare for indexing
        logger.info(f"Preparing {len(unique_strings)} unique strings for indexing")
        
        # Index unique strings with their vectors
        success = self.weaviate_manager.index_unique_strings(
            unique_strings=unique_strings,
            embeddings=self.embedding_generator.embeddings,
            string_counts=string_counts,
            field_types=field_types
        )
        
        if not success:
            logger.error("Failed to index unique strings in Weaviate")
            raise RuntimeError("String indexing failed")
        
        # Load record field hashes to prepare entity maps
        record_hashes_path = os.path.join(checkpoint_dir, "record_field_hashes.json")
        if not os.path.exists(record_hashes_path):
            logger.error(f"Record hashes file not found: {record_hashes_path}")
            raise FileNotFoundError(f"Record hashes file not found: {record_hashes_path}")
        
        try:
            with open(record_hashes_path, 'r') as f:
                record_field_hashes = json.load(f)
            logger.info(f"Loaded {len(record_field_hashes)} record field hashes")
        except Exception as e:
            logger.error(f"Error loading record field hashes: {str(e)}")
            raise RuntimeError(f"Failed to load record field hashes: {str(e)}")
        
        # Prepare entity maps
        entity_maps = []
        max_records = self.config.get("max_records")
        
        for record_id, field_hashes in record_field_hashes.items():
            person_name = ""
            person_hash = field_hashes.get("person")
            if person_hash != "NULL":
                person_name = unique_strings.get(person_hash, "")
                    
            entity_maps.append({
                "entity_id": record_id,
                "field_hashes": field_hashes,
                "person_name": person_name
            })
                
            # Limit number of entities in development mode
            if max_records and len(entity_maps) >= max_records:
                logger.info(f"Reached maximum records limit: {max_records}")
                break
        
        # Index entity maps (optional)
        if self.config.get("index_entity_maps", True):
            logger.info(f"Indexing {len(entity_maps)} entity maps")
            success = self.weaviate_manager.index_entity_maps(entity_maps)
            if not success:
                logger.warning("Failed to index entity maps (non-critical)")
        
        # Retrieve field vectors for all records
        logger.info("Retrieving field vectors for records")
        records_to_process = {}
        for record_id, field_hashes in record_field_hashes.items():
            # Apply record limit
            if max_records and len(records_to_process) >= max_records:
                break
                    
            records_to_process[record_id] = field_hashes
        
        # Batch retrieve field vectors
        logger.info(f"Retrieving vectors for {len(records_to_process)} records")
        self.record_embeddings = self.weaviate_manager.get_field_vectors_for_records(records_to_process)
        
        # Save retrieved embeddings for later use
        self._save_record_embeddings()
        
        # Save stats
        self.stats["string_indexing"] = {
            "unique_strings_indexed": len(unique_strings),
            "entity_maps_indexed": len(entity_maps),
            "field_types_mapped": len(field_types),
            "records_processed": len(self.record_embeddings)
        }
        
        logger.info(f"String indexing complete: {len(unique_strings)} unique strings indexed")
    
    def _prepare_training_data_wrapper(self) -> Tuple[Dict, Dict, List]:
        """
        Wrapper for prepare_training_data that returns the results.
        Clearly separates training data preparation from model training.
        
        Returns:
            Tuple of (train_records, test_records, ground_truth_pairs)
        """
        logger.info("Preparing training data from ground truth")
        
        # Parse ground truth file to get record IDs
        ground_truth_file = self.config.get("ground_truth_file")
        ground_truth_pairs = self.data_processor.parse_ground_truth(ground_truth_file)
        
        if not ground_truth_pairs:
            logger.error("No ground truth pairs found. Check your ground truth file format.")
            return {}, {}, []
        
        # First, retrieve vectors for all records in ground truth
        self._retrieve_vectors_for_training(ground_truth_pairs)
        
        # Then prepare the training data
        result = self.prepare_training_data()
        
        logger.info("Completed training data preparation")
        return result

    def check_openai_api_key(self) -> bool:
        """
        Check if a valid OpenAI API key is available.
        
        Returns:
            bool: True if API key is valid, False otherwise
        """
        logger.info("Checking OpenAI API key")
        
        # First, look in config
        api_key = self.config.get("openai_api_key")
        
        # If not in config, check environment variable
        if not api_key or api_key == "your-openai-api-key":
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
            
            if api_key:
                logger.info("Using OpenAI API key from environment variable")
                # Update config to use the key from environment
                self.config["openai_api_key"] = api_key
        
        # Validate the key (basic validation)
        if not api_key:
            logger.error("No OpenAI API key found in config or environment variables")
            return False
            
        if api_key == "your-openai-api-key" or len(api_key) < 10:
            logger.error("Invalid OpenAI API key. Please set a valid API key.")
            return False
        
        # If the embedding generator is already initialized, update its client's API key
        if hasattr(self, 'embedding_generator') and self.embedding_generator:
            try:
                self.embedding_generator.client.api_key = api_key
                logger.info("Updated embedding generator with API key")
            except Exception as e:
                logger.warning(f"Could not update embedding generator client: {e}")
        
        # Do a test API call to verify key works
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # Try a very small models call just to validate the key
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input="test",
                encoding_format="float"
            )
            logger.info("Successfully verified OpenAI API key with a test call")
            return True
        except Exception as e:
            logger.error(f"Failed to verify OpenAI API key: {e}")
            return False

    def _retrieve_vectors_for_training(self, ground_truth_pairs: List[Tuple[str, str, bool]]) -> None:
        """
        Retrieve vectors specifically for the training dataset.
        This is a separate step to maintain clear stage separation.
        
        Args:
            ground_truth_pairs: List of ground truth pairs
        """
        logger.info("Retrieving vectors for training records")
        
        # Collect all record IDs from ground truth
        record_ids = set()
        for left_id, right_id, _ in ground_truth_pairs:
            record_ids.add(left_id)
            record_ids.add(right_id)
        
        logger.info(f"Found {len(record_ids)} unique records in ground truth data")
        
        # Get field hashes for these records
        records_to_process = {}
        for record_id in record_ids:
            if record_id in self.data_processor.deduplicator.record_field_hashes:
                records_to_process[record_id] = self.data_processor.deduplicator.record_field_hashes[record_id]
            else:
                logger.warning(f"Record ID {record_id} from ground truth not found in processed data")
        
        if not records_to_process:
            logger.error("No matching records found between ground truth and processed data")
            return
        
        # Batch retrieve field vectors
        self.record_embeddings = self.weaviate_manager.get_field_vectors_for_records(records_to_process)
        
        # Save retrieved embeddings for later use
        self._save_record_embeddings()
        
        logger.info(f"Retrieved vectors for {len(self.record_embeddings)} training records")
    
    def prepare_training_data(self) -> Tuple[Dict[str, Dict], Dict[str, Dict], List[Tuple]]:
        """
        Prepare data for classifier training, separate from the full dataset classification.
        
        Returns:
            Tuple of (train_records, test_records, ground_truth_pairs)
        """
        logger.info("Preparing training data")
        
        # Parse ground truth file
        ground_truth_file = self.config.get("ground_truth_file")
        ground_truth_pairs = self.data_processor.parse_ground_truth(ground_truth_file)
        
        # Initialize imputer for training data
        training_imputer = NullValueImputer(self.config, self.weaviate_manager)
        
        # Collect records for training
        all_records = {}
        for left_id, right_id, _ in ground_truth_pairs:
            # Get record hashes
            for record_id in [left_id, right_id]:
                if record_id not in all_records and record_id in self.data_processor.deduplicator.record_field_hashes:
                    
                    field_hashes = self.data_processor.deduplicator.record_field_hashes[record_id]
                    
                    # Reconstruct record from field hashes
                    record = self._reconstruct_record_from_hashes(record_id, field_hashes)
                    
                    if record:
                        # Apply imputation if needed
                        if record_id in self.record_embeddings:
                            record_embeddings = self.record_embeddings[record_id]
                            for field in training_imputer.imputable_fields:
                                if not record.get(field) or pd.isna(record.get(field)):
                                    record = training_imputer.impute_null_fields(record, record_embeddings)
                                    break
                        
                        all_records[record_id] = record
        
        logger.info(f"Collected {len(all_records)} records for training")
        
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
            "test_records": len(test_records),
            "ground_truth_total": len(ground_truth_pairs)
        }
        
        return train_records, test_records, ground_truth_pairs
    
    def _reconstruct_record_from_hashes(self, record_id: str, field_hashes: Dict[str, str]) -> Optional[Dict]:
        """
        Reconstruct a record from its field hashes by looking up the original strings.
        
        Args:
            record_id: ID of the record
            field_hashes: Dictionary of field->hash mappings
            
        Returns:
            Reconstructed record dictionary or None if failed
        """
        try:
            record = {"id": record_id}
            unique_strings = self.data_processor.deduplicator.unique_strings
            
            # Loop through each field and get its original string
            for field, hash_value in field_hashes.items():
                if hash_value != "NULL":
                    if hash_value in unique_strings:
                        record[field] = unique_strings.get(hash_value, "")
                    else:
                        logger.warning(f"Hash value {hash_value} for field {field} in record {record_id} not found in unique strings")
                        record[field] = ""  # Provide empty string instead of None to avoid errors
                else:
                    record[field] = None
            
            # Validate that we have the minimum required fields
            required_fields = ['person', 'roles', 'title']
            missing_fields = [field for field in required_fields if field not in record or not record[field]]
            
            if missing_fields:
                logger.warning(f"Reconstructed record {record_id} missing required fields: {missing_fields}")
                if len(missing_fields) == len(required_fields):
                    logger.error(f"Record {record_id} is missing all required fields, cannot use for training")
                    return None
            
            return record
        except Exception as e:
            logger.error(f"Error reconstructing record {record_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def run_clustering(self) -> List[Dict]:
        """
        Run the entity clustering pipeline on the full dataset.
        Clearly separated from the model training stage.
        
        Returns:
            List of entity clusters
        """
        logger.info("Running entity clustering on full dataset")
        
        # If we don't have vectors for the full dataset yet, retrieve them
        if not hasattr(self, 'record_embeddings') or len(self.record_embeddings) < len(self.data_processor.deduplicator.record_field_hashes):
            self._retrieve_vectors_for_full_dataset()
        
        # Get all records
        records = []
        for record_id in self.record_embeddings.keys():
            field_hashes = self.data_processor.deduplicator.record_field_hashes.get(record_id)
            if field_hashes:
                record = self._reconstruct_record_from_hashes(record_id, field_hashes)
                if record:
                    records.append(record)
        
        logger.info(f"Clustering {len(records)} records from full dataset")
        
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
    
    def _retrieve_vectors_for_full_dataset(self) -> None:
        """
        Retrieve vectors for all records in the full dataset.
        This is done after model training, as part of the clustering stage.
        """
        logger.info("Retrieving vectors for full dataset")
        
        # Get field hashes for all records (or up to max_records)
        records_to_process = {}
        max_records = self.config.get("max_records")
        
        total_records = len(self.data_processor.deduplicator.record_field_hashes)
        logger.info(f"Total records in processed data: {total_records}")
        
        if max_records and max_records < total_records:
            logger.info(f"Limiting to {max_records} records as specified in config")
        
        count = 0
        for record_id, field_hashes in self.data_processor.deduplicator.record_field_hashes.items():
            records_to_process[record_id] = field_hashes
            count += 1
            
            if max_records and count >= max_records:
                logger.info(f"Reached maximum records limit: {max_records}")
                break
                
            # Log progress for large datasets
            if count % 10000 == 0:
                logger.info(f"Prepared {count}/{min(total_records, max_records or total_records)} records for vector retrieval")
        
        if not records_to_process:
            logger.error("No records to process for vector retrieval")
            return
        
        logger.info(f"Retrieving vectors for {len(records_to_process)} records")
        
        try:
            # Batch retrieve field vectors
            start_time = time.time()
            self.record_embeddings = self.weaviate_manager.get_field_vectors_for_records(records_to_process)
            retrieval_time = time.time() - start_time
            
            # Save retrieved embeddings
            self._save_record_embeddings()
            
            retrieved_count = len(self.record_embeddings)
            missing_count = len(records_to_process) - retrieved_count
            
            logger.info(f"Retrieved vectors for {retrieved_count} records in {retrieval_time:.2f} seconds")
            
            if missing_count > 0:
                logger.warning(f"Failed to retrieve vectors for {missing_count} records")
                
                # Check for a sample of missing records
                missing_ids = set(records_to_process.keys()) - set(self.record_embeddings.keys())
                if missing_ids:
                    sample_size = min(5, len(missing_ids))
                    sample_ids = list(missing_ids)[:sample_size]
                    logger.warning(f"Sample of missing record IDs: {sample_ids}")
        except Exception as e:
            logger.error(f"Error retrieving vectors for full dataset: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
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
        Clean up resources and close connections.
        """
        logger.info("Performing cleanup operations")
        
        # Close Weaviate connection if it exists
        if hasattr(self, 'weaviate_manager') and self.weaviate_manager:
            try:
                self.weaviate_manager.close()
                logger.info("Weaviate connection closed")
            except Exception as e:
                logger.warning(f"Error closing Weaviate connection: {str(e)}")
        
        # Close any OpenAI client connection if needed
        if hasattr(self, 'embedding_generator') and hasattr(self.embedding_generator, 'client'):
            try:
                # Some API clients have close methods
                if hasattr(self.embedding_generator.client, 'close'):
                    self.embedding_generator.client.close()
                    logger.info("OpenAI client connection closed")
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {str(e)}")
        
        # Close any open file handles
        try:
            import gc
            gc.collect()  # Force garbage collection to close any lingering file handles
        except Exception:
            pass
            
        # Check for any resources that need explicit cleanup
        for attr_name in ['imputer', 'clusterer', 'training_imputer', 'training_clusterer']:
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)
        
        # Remove temporary files if configured
        if self.config.get("cleanup_temp_files", True):
            temp_dirs = ["temp"]  # Don't remove checkpoints
            for temp_dir in temp_dirs:
                temp_path = os.path.join(self.output_dir, temp_dir)
                if os.path.exists(temp_path):
                    try:
                        import shutil
                        shutil.rmtree(temp_path)
                        logger.info(f"Removed temporary directory: {temp_path}")
                    except Exception as e:
                        logger.warning(f"Error removing temporary directory: {str(e)}")
        
        logger.info("Cleanup completed")