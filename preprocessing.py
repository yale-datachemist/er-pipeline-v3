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
        # Get CSV delimiter from config
        self.csv_delimiter = config.get("csv_delimiter", ",")
        
        # Fields to deduplicate and embed
        self.embed_fields = [
            'record', 'person', 'roles', 'title' 
            'provision', 'subjects', 'genres'
        ]
        # Fields that might contain null values
        self.nullable_fields = [
            'provision', 'subjects', 'genres'
        ]
        # Track unique strings and their frequencies
        self.unique_strings: Dict[str, str] = {}  # hash -> original string
        self.string_counts: DefaultDict[str, int] = defaultdict(int)  # hash -> count
        # Track field types for string-centric architecture
        self.field_types: Dict[str, str] = {}  # hash -> field_type
        # Map record_id to field hashes
        self.record_field_hashes: Dict[str, Dict[str, str]] = {}  # record_id -> {field -> hash}
        
        # Enable verbose logging based on config
        self.verbose_logging = config.get("verbose_logging", False)
        self.log_csv_parsing = config.get("log_csv_parsing", False)
        
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
            # Changed delimiter from tab to comma
            df = pd.read_csv(file_path, sep=',', dtype=str, on_bad_lines='warn')
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
        
        # Check if directory exists
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return
            
        # Log the files found
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {directory}: {csv_files}")
        
        if not csv_files:
            logger.warning(f"No CSV files found in directory: {directory}")
            return
        
        files_processed = 0
        records_processed = 0
        
        for file_name in csv_files:
            file_path = os.path.join(directory, file_name)
            processed_records = self.process_file(file_path)
            
            files_processed += 1
            records_processed += len(processed_records)
            
            # Save checkpoint if configured
            if files_processed % self.config.get("checkpoint_frequency", 50) == 0:
                self.save_checkpoint()
                logger.info(f"Progress: {files_processed}/{len(csv_files)} files processed")
        
        logger.info(f"Completed processing {files_processed} files with {records_processed} records")
        self.save_checkpoint()
        
    def save_checkpoint(self) -> bool:
        """
        Save the current state to checkpoint files.
        
        Returns:
            Boolean indicating success
        """
        try:
            checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            logger.info(f"Saving checkpoints to {checkpoint_dir}")
            
            # Get file paths for checkpoints
            unique_strings_path = os.path.join(checkpoint_dir, "unique_strings.json")
            string_counts_path = os.path.join(checkpoint_dir, "string_counts.json")
            field_types_path = os.path.join(checkpoint_dir, "field_types.json")
            record_field_hashes_path = os.path.join(checkpoint_dir, "record_field_hashes.json")
            
            # Convert defaultdict to regular dict for serialization
            string_counts_dict = dict(self.string_counts)
            
            # Save unique strings
            with open(unique_strings_path, 'w') as f:
                json.dump(self.unique_strings, f)
            logger.info(f"Saved {len(self.unique_strings)} unique strings to {unique_strings_path}")
                
            # Save string counts
            with open(string_counts_path, 'w') as f:
                json.dump(string_counts_dict, f)
            logger.info(f"Saved {len(string_counts_dict)} string counts to {string_counts_path}")
                
            # Save field types
            with open(field_types_path, 'w') as f:
                json.dump(self.field_types, f)
            logger.info(f"Saved {len(self.field_types)} field types to {field_types_path}")
                
            # Save record field hashes - split into batches if large
            # If record_field_hashes is too large, consider chunking it
            num_records = len(self.record_field_hashes)
            if num_records > 100000:  # arbitrary threshold for large datasets
                logger.info(f"Large dataset detected ({num_records} records), chunking record_field_hashes")
                # Todo: Implement chunking if needed
                pass
            
            with open(record_field_hashes_path, 'w') as f:
                json.dump(self.record_field_hashes, f)
            logger.info(f"Saved field hashes for {len(self.record_field_hashes)} records to {record_field_hashes_path}")
                
            logger.info(f"Checkpoint saved successfully with {len(self.unique_strings)} unique strings")
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_checkpoint(self) -> bool:
        """
        Load state from checkpoint files with improved error handling.
        
        Returns:
            Boolean indicating success
        """
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        
        # Check if checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return False
        
        # Define paths to checkpoint files
        unique_strings_path = os.path.join(checkpoint_dir, "unique_strings.json")
        
        # Check if the required files exist
        if not os.path.exists(unique_strings_path):
            logger.error(f"Unique strings checkpoint file not found: {unique_strings_path}")
            return False
        
        try:
            # Load unique strings
            with open(unique_strings_path, 'r') as f:
                logger.info(f"Loading unique strings from {unique_strings_path}")
                self.unique_strings = json.load(f)
                logger.info(f"Loaded {len(self.unique_strings)} unique strings")
                    
            # Load other checkpoint files...
                    
            logger.info(f"Successfully loaded checkpoint data")
            return True
                
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
            logger.info(f"Parsing ground truth file: {ground_truth_file}")
            
            if not os.path.exists(ground_truth_file):
                logger.error(f"Ground truth file does not exist: {ground_truth_file}")
                return []
                
            # Read with automatic delimiter detection
            df = pd.read_csv(ground_truth_file)
            
            # Log column names to help debugging
            logger.info(f"Ground truth columns: {df.columns.tolist()}")
            
            # Check required columns
            required_columns = ['left', 'right', 'match']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in ground truth file: {missing_columns}")
                return []
            
            # Convert to list of tuples
            ground_truth = []
            for _, row in df.iterrows():
                left_id = row['left']
                right_id = row['right']
                
                # Handle different formats of truth values
                if isinstance(row['match'], bool):
                    is_match = row['match']
                elif isinstance(row['match'], str):
                    is_match = row['match'].lower() in ('true', 'yes', '1', 't')
                elif isinstance(row['match'], (int, float)):
                    is_match = bool(row['match'])
                else:
                    is_match = False
                    
                ground_truth.append((left_id, right_id, is_match))
                    
            # Log statistics for debugging
            match_count = sum(1 for _, _, is_match in ground_truth if is_match)
            non_match_count = len(ground_truth) - match_count
            
            logger.info(f"Parsed {len(ground_truth)} ground truth pairs: "
                    f"{match_count} matches, {non_match_count} non-matches")
            
            return ground_truth
            
        except Exception as e:
            logger.error(f"Error parsing ground truth file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def run_preprocessing(self, mode: str = "full") -> None:
        """
        Run the preprocessing pipeline based on the specified mode with explicit checkpoint saving.
        
        Args:
            mode: 'full' for production or 'dev' for development mode
        """
        logger.info(f"Running preprocessing in {mode} mode")
        
        # Get CSV delimiter from config
        csv_delimiter = self.config.get("csv_delimiter", ",")
        logger.info(f"Using CSV delimiter: '{csv_delimiter}'")
        
        # Set the delimiter in the deduplicator
        self.deduplicator.csv_delimiter = csv_delimiter
        
        # Ensure checkpoint directory exists
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Using checkpoint directory: {checkpoint_dir}")
        
        if mode == "dev":
            # For development mode, create subset and process it
            data_dir = self.config.get("data_dir")
            logger.info(f"Creating development subset from {data_dir}")
            
            if not os.path.exists(data_dir):
                logger.error(f"Data directory does not exist: {data_dir}")
                return
            
            max_records = self.config.get("max_records", 10000)
            self.prepare_development_subset(data_dir, max_records)
            
            # Process the development subset
            dev_data_dir = self.config.get("dev_data_dir", "dev_data")
            if not os.path.exists(dev_data_dir):
                logger.warning(f"Development data directory does not exist: {dev_data_dir}")
                os.makedirs(dev_data_dir, exist_ok=True)
                
            self.deduplicator.process_directory(dev_data_dir)
        else:
            # For full mode, process all data
            data_dir = self.config.get("data_dir")
            if not os.path.exists(data_dir):
                logger.error(f"Data directory does not exist: {data_dir}")
                return
                
            self.deduplicator.process_directory(data_dir)
        
        # Explicitly save checkpoints after processing
        logger.info("Explicitly saving checkpoint after preprocessing")
        if not self.deduplicator.save_checkpoint():
            logger.error("Failed to save checkpoint after preprocessing")
            raise RuntimeError("Failed to save preprocessing results")
        
        # Print statistics
        stats = self.deduplicator.get_statistics()
        logger.info(f"Preprocessing statistics: {json.dumps(stats, indent=2)}")
        
        # Check if we processed any records
        if stats["total_records"] == 0:
            logger.error("No records were processed. Check your data files and CSV format.")
        else:
            logger.info(f"Successfully processed {stats['total_records']} records with {stats['total_unique_strings']} unique strings")
            
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
        else:
            logger.info("All required checkpoint files were created successfully")
