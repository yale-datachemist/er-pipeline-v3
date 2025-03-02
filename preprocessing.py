import pandas as pd
import os
import json
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, DefaultDict, Optional
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextDeduplicator:
    """
    Handles deduplication of text fields to optimize embedding generation.
    """
    def __init__(self, config: Dict):
        """
        Initialize the deduplicator with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        # Fields to deduplicate and embed
        self.embed_fields = [
            'record', 'person', 'roles', 'title', 'attribution', 
            'provision', 'subjects', 'genres', 'relatedWork'
        ]
        # Fields that might contain null values
        self.nullable_fields = [
            'attribution', 'provision', 'subjects', 'genres', 'relatedWork'
        ]
        # Track unique strings and their frequencies
        self.unique_strings: Dict[str, str] = {}  # hash -> original string
        self.string_counts: DefaultDict[str, int] = defaultdict(int)  # hash -> count
        # Track field types for string-centric architecture
        self.field_types: Dict[str, str] = {}  # hash -> field_type
        # Map record_id to field hashes
        self.record_field_hashes: Dict[str, Dict[str, str]] = {}  # record_id -> {field -> hash}
        
    def _hash_text(self, text: str) -> str:
        """
        Generate a hash for a text string.
        
        Args:
            text: Input text string
            
        Returns:
            Hash of the input string
        """
        if pd.isna(text) or text is None:
            return "NULL"
        
        # Normalize the text (lowercase, strip whitespace)
        normalized = str(text).lower().strip()
        # Generate a hash
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def process_record(self, record: Dict) -> Dict[str, str]:
        """
        Process a single record, deduplicating its text fields.
        
        Args:
            record: Dictionary containing record fields
            
        Returns:
            Dictionary mapping field names to their hashes
        """
        field_hashes = {}
        record_id = record.get('id')
        
        for field in self.embed_fields:
            text = record.get(field)
            if pd.isna(text) or text is None:
                field_hashes[field] = "NULL"
                continue
                
            text_hash = self._hash_text(text)
            field_hashes[field] = text_hash
            
            # Store the unique string if we haven't seen it before
            if text_hash not in self.unique_strings and text_hash != "NULL":
                self.unique_strings[text_hash] = str(text)
                # For string-centric architecture, track which field type this string belongs to
                self.field_types[text_hash] = field
            
            # Increment the count for this string
            if text_hash != "NULL":
                self.string_counts[text_hash] += 1
        
        # Store the field hashes for this record
        if record_id:
            self.record_field_hashes[record_id] = field_hashes
            
        return field_hashes
    
    def process_file(self, file_path: str) -> List[Dict]:
        """
        Process a single CSV file, deduplicating all records.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of processed records
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Add error handling for CSV parsing with on_bad_lines parameter
            df = pd.read_csv(file_path, sep='\t', dtype=str, on_bad_lines='warn')
            processed_records = []
            
            for _, row in df.iterrows():
                record_dict = row.to_dict()
                field_hashes = self.process_record(record_dict)
                
                # Add the original record with field hashes
                processed_records.append({
                    "original": record_dict,
                    "field_hashes": field_hashes
                })
                
            logger.info(f"Processed {len(processed_records)} records from {file_path}")
            return processed_records
        
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error in file {file_path}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def process_directory(self, directory: str) -> None:
        """
        Process all CSV files in a directory.
        
        Args:
            directory: Path to the directory containing CSV files
        """
        logger.info(f"Processing directory: {directory}")
        
        files_processed = 0
        records_processed = 0
        
        for file_name in os.listdir(directory):
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory, file_name)
                processed_records = self.process_file(file_path)
                
                files_processed += 1
                records_processed += len(processed_records)
                
                # Save checkpoint if configured
                if files_processed % self.config.get("checkpoint_frequency", 50) == 0:
                    self.save_checkpoint()
        
        logger.info(f"Completed processing {files_processed} files with {records_processed} records")
        self.save_checkpoint()
        
    def save_checkpoint(self) -> None:
        """
        Save the current state to checkpoint files.
        """
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save unique strings
        with open(os.path.join(checkpoint_dir, "unique_strings.json"), 'w') as f:
            json.dump(self.unique_strings, f)
            
        # Save string counts
        with open(os.path.join(checkpoint_dir, "string_counts.json"), 'w') as f:
            json.dump(self.string_counts, f)
            
        # Save field types (for string-centric architecture)
        with open(os.path.join(checkpoint_dir, "field_types.json"), 'w') as f:
            json.dump(self.field_types, f)
            
        # Save record field hashes (might be large, consider chunking)
        with open(os.path.join(checkpoint_dir, "record_field_hashes.json"), 'w') as f:
            json.dump(self.record_field_hashes, f)
            
        logger.info(f"Saved checkpoint with {len(self.unique_strings)} unique strings")
    
    def load_checkpoint(self) -> bool:
        """
        Load state from checkpoint files.
        
        Returns:
            Boolean indicating success
        """
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        
        try:
            # Load unique strings
            with open(os.path.join(checkpoint_dir, "unique_strings.json"), 'r') as f:
                self.unique_strings = json.load(f)
                
            # Load string counts
            with open(os.path.join(checkpoint_dir, "string_counts.json"), 'r') as f:
                self.string_counts = defaultdict(int, json.load(f))
            
            # Load field types (for string-centric architecture)
            field_types_path = os.path.join(checkpoint_dir, "field_types.json")
            if os.path.exists(field_types_path):
                with open(field_types_path, 'r') as f:
                    self.field_types = json.load(f)
            else:
                # If field_types.json doesn't exist (backward compatibility),
                # generate field types from record_field_hashes
                logger.warning("field_types.json not found, generating from record_field_hashes")
                self.field_types = {}
                
            # Load record field hashes
            with open(os.path.join(checkpoint_dir, "record_field_hashes.json"), 'r') as f:
                self.record_field_hashes = json.load(f)
                
                # If field_types is empty, populate it from record_field_hashes
                if not self.field_types:
                    for record_id, field_hash_map in self.record_field_hashes.items():
                        for field, hash_value in field_hash_map.items():
                            if hash_value != "NULL":
                                self.field_types[hash_value] = field
                
            logger.info(f"Loaded checkpoint with {len(self.unique_strings)} unique strings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the deduplication process.
        
        Returns:
            Dictionary containing statistics
        """
        total_records = len(self.record_field_hashes)
        total_unique_strings = len(self.unique_strings)
        
        # Count null values per field
        null_counts = defaultdict(int)
        for record_id, field_hashes in self.record_field_hashes.items():
            for field, hash_value in field_hashes.items():
                if hash_value == "NULL":
                    null_counts[field] += 1
        
        # Calculate duplication ratio
        total_strings = sum(self.string_counts.values())
        duplication_ratio = (total_strings - total_unique_strings) / total_strings if total_strings > 0 else 0
        
        return {
            "total_records": total_records,
            "total_unique_strings": total_unique_strings,
            "total_strings": total_strings,
            "duplication_ratio": duplication_ratio,
            "null_counts": dict(null_counts)
        }


class DataProcessor:
    """
    Top-level class for data preprocessing and management.
    """
    def __init__(self, config: Dict):
        """
        Initialize the data processor with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.deduplicator = TextDeduplicator(config)
        
    def prepare_development_subset(self, data_dir: str, sample_size: int = 10000) -> None:
        """
        Prepare a subset of data for development purposes.
        
        Args:
            data_dir: Directory containing the full dataset
            sample_size: Number of records to include in the subset
        """
        logger.info(f"Preparing development subset with {sample_size} records")
        
        output_dir = self.config.get("dev_data_dir", "dev_data")
        os.makedirs(output_dir, exist_ok=True)
        
        all_records = []
        
        # Collect records from files until we have enough
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(data_dir, file_name)
                df = pd.read_csv(file_path, sep='\t', dtype=str)
                all_records.extend(df.to_dict('records'))
                
                if len(all_records) >= sample_size:
                    break
        
        # Take a random sample if we have more than needed
        if len(all_records) > sample_size:
            import random
            random.seed(42)  # For reproducibility
            all_records = random.sample(all_records, sample_size)
        
        # Save to a single CSV file
        pd.DataFrame(all_records).to_csv(
            os.path.join(output_dir, "dev_subset.csv"), 
            sep='\t', 
            index=False
        )
        
        logger.info(f"Saved development subset with {len(all_records)} records")
    
    def parse_ground_truth(self, ground_truth_file: str) -> List[Tuple[str, str, bool]]:
        """
        Parse ground truth file containing labeled matches.
        
        Args:
            ground_truth_file: Path to the ground truth file
            
        Returns:
            List of tuples (left_id, right_id, is_match)
        """
        try:
            df = pd.read_csv(ground_truth_file, sep='\s+')
            
            # Convert to list of tuples
            ground_truth = []
            for _, row in df.iterrows():
                left_id = row['left']
                right_id = row['right']
                is_match = str(row['match']).lower() == 'true'
                ground_truth.append((left_id, right_id, is_match))
                
            logger.info(f"Parsed {len(ground_truth)} ground truth pairs")
            return ground_truth
            
        except Exception as e:
            logger.error(f"Error parsing ground truth file: {str(e)}")
            return []
    
    def run_preprocessing(self, mode: str = "full") -> None:
        """
        Run the preprocessing pipeline based on the specified mode.
        
        Args:
            mode: 'full' for production or 'dev' for development mode
        """
        if mode == "dev":
            # For development mode, create subset and process it
            data_dir = self.config.get("data_dir")
            self.prepare_development_subset(data_dir)
            self.deduplicator.process_directory(self.config.get("dev_data_dir", "dev_data"))
        else:
            # For full mode, process all data
            self.deduplicator.process_directory(self.config.get("data_dir"))
        
        # Print statistics
        stats = self.deduplicator.get_statistics()
        logger.info(f"Preprocessing statistics: {json.dumps(stats, indent=2)}")
