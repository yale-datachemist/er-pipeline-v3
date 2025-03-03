#!/usr/bin/env python
"""
This is a diagnostic script to pinpoint issues with embeddings for training data.
It focuses only on reading the ground truth file and loading/retrieving embeddings.
"""

import os
import json
import pickle
import argparse
import logging
import yaml
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.collections import Collection
from weaviate.classes.query import Filter, MetadataQuery

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("diagnostics")

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_ground_truth(ground_truth_file):
    """Parse ground truth file containing labeled matches."""
    import pandas as pd
    
    logger.info(f"Parsing ground truth file: {ground_truth_file}")
    
    if not os.path.exists(ground_truth_file):
        logger.error(f"Ground truth file does not exist: {ground_truth_file}")
        return []
        
    # Read with automatic delimiter detection
    df = pd.read_csv(ground_truth_file)
    
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

def load_record_field_hashes(checkpoint_dir):
    """Load record field hashes from checkpoint file."""
    record_hashes_path = os.path.join(checkpoint_dir, "record_field_hashes.json")
    
    if not os.path.exists(record_hashes_path):
        logger.error(f"Record hashes file not found: {record_hashes_path}")
        return {}
        
    try:
        with open(record_hashes_path, 'r') as f:
            record_field_hashes = json.load(f)
        logger.info(f"Loaded {len(record_field_hashes)} record field hashes")
        return record_field_hashes
    except Exception as e:
        logger.error(f"Error loading record field hashes: {e}")
        return {}

def load_unique_strings(checkpoint_dir):
    """Load unique strings from checkpoint file."""
    unique_strings_path = os.path.join(checkpoint_dir, "unique_strings.json")
    
    if not os.path.exists(unique_strings_path):
        logger.error(f"Unique strings file not found: {unique_strings_path}")
        return {}
        
    try:
        with open(unique_strings_path, 'r') as f:
            unique_strings = json.load(f)
        logger.info(f"Loaded {len(unique_strings)} unique strings")
        return unique_strings
    except Exception as e:
        logger.error(f"Error loading unique strings: {e}")
        return {}

def load_embeddings(output_dir):
    """Load global embeddings from NPZ file."""
    embeddings_file = os.path.join(output_dir, "embeddings.npz")
    
    if not os.path.exists(embeddings_file):
        logger.error(f"Embeddings file not found: {embeddings_file}")
        return {}
        
    try:
        import numpy as np
        embeddings = {}
        with np.load(embeddings_file, allow_pickle=True) as data:
            for key in data.files:
                embeddings[key] = data[key].tolist()
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return {}

def load_record_embeddings(checkpoint_dir):
    """Load record embeddings from pickle file."""
    embeddings_file = os.path.join(checkpoint_dir, "record_embeddings.pkl")
    
    if not os.path.exists(embeddings_file):
        logger.error(f"Record embeddings file not found: {embeddings_file}")
        return {}
        
    try:
        with open(embeddings_file, 'rb') as f:
            record_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(record_embeddings)} record embeddings")
        return record_embeddings
    except Exception as e:
        logger.error(f"Error loading record embeddings: {e}")
        return {}

def connect_to_weaviate(weaviate_url):
    """Connect to Weaviate."""
    try:
        import weaviate
        
        logger.info(f"Connecting to Weaviate at {weaviate_url}")
        
        if weaviate_url.startswith("http://localhost") or weaviate_url.startswith("http://127.0.0.1"):
            client = weaviate.connect_to_local()
        else:
            client = weaviate.connect_to_wcs(
                cluster_url=weaviate_url
            )
        
        # Check if connection is successful
        meta = client.get_meta()
        logger.info(f"Connected to Weaviate version: {meta}")
        return client
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        return None

def check_string_in_weaviate(client, hash_value):
    """Check if a string hash exists in Weaviate."""
    if not client:
        return False
        
    try:
        collection = client.collections.get("UniqueStrings")
        result = collection.query.fetch_objects(
            filters=collection.Filter.by_property("hash").equal(hash_value),
            limit=1
        )
        
        return len(result.objects) > 0
    except Exception as e:
        logger.error(f"Error checking string in Weaviate: {e}")
        return False

def retrieve_vector_from_weaviate(client, hash_value):
    """Retrieve vector for a hash from Weaviate."""
    if not client:
        return None
        
    try:
        collection = client.collections.get("UniqueStrings")
        result = collection.query.fetch_objects(
            filters=collection.Filter.by_property("hash").equal(hash_value),
            limit=1,
            include_vector=True
        )
        
        if result.objects:
            return result.objects[0].vector
        return None
    except Exception as e:
        logger.error(f"Error retrieving vector from Weaviate: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Diagnose embeddings for training data')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    output_dir = config.get("output_dir", "output")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    ground_truth_file = config.get("ground_truth_file")
    weaviate_url = config.get("weaviate_url", "http://localhost:8080")
    
    # Parse ground truth
    ground_truth_pairs = parse_ground_truth(ground_truth_file)
    if not ground_truth_pairs:
        return
    
    # Collect all record IDs from ground truth
    record_ids = set()
    for left_id, right_id, _ in ground_truth_pairs:
        record_ids.add(left_id)
        record_ids.add(right_id)
    
    logger.info(f"Found {len(record_ids)} unique records in ground truth")
    
    # Load record field hashes
    record_field_hashes = load_record_field_hashes(checkpoint_dir)
    if not record_field_hashes:
        return
    
    # Check if ground truth records exist in field hashes
    missing_record_ids = record_ids - set(record_field_hashes.keys())
    if missing_record_ids:
        logger.error(f"{len(missing_record_ids)} ground truth records not found in field hashes")
        if len(missing_record_ids) <= 10:
            logger.error(f"Missing record IDs: {list(missing_record_ids)}")
    
    # Load unique strings
    unique_strings = load_unique_strings(checkpoint_dir)
    if not unique_strings:
        return
    
    # Load global embeddings
    embeddings = load_embeddings(output_dir)
    if not embeddings:
        return
    
    # Load record embeddings
    record_embeddings = load_record_embeddings(checkpoint_dir)
    
    # Connect to Weaviate
    client = connect_to_weaviate(weaviate_url)
    
    # Check if record embeddings match ground truth records
    if record_embeddings:
        matching_records = record_ids & set(record_embeddings.keys())
        logger.info(f"{len(matching_records)} of {len(record_ids)} ground truth records have embeddings")
    else:
        logger.info("No record embeddings found")
    
    # Sample a few record IDs for deeper analysis
    sample_ids = list(record_ids)[:5]
    logger.info(f"Analyzing {len(sample_ids)} sample records in depth")
    
    for record_id in sample_ids:
        logger.info(f"\nAnalyzing record: {record_id}")
        
        # Check if record ID exists in field hashes
        if record_id not in record_field_hashes:
            logger.error(f"Record ID not found in field hashes")
            continue
        
        # Get field hashes for the record
        field_hashes = record_field_hashes[record_id]
        logger.info(f"Record has {len(field_hashes)} fields")
        
        # Check each field hash
        for field, hash_value in field_hashes.items():
            if hash_value == "NULL":
                logger.info(f"Field {field}: NULL value")
                continue
            
            # Check if hash exists in unique strings
            hash_in_strings = hash_value in unique_strings
            logger.info(f"Field {field}, Hash {hash_value}: In unique_strings? {hash_in_strings}")
            
            # Check if hash has an embedding
            hash_has_embedding = hash_value in embeddings
            logger.info(f"Field {field}, Hash {hash_value}: Has embedding? {hash_has_embedding}")
            
            # Check if hash exists in Weaviate
            if client:
                hash_in_weaviate = check_string_in_weaviate(client, hash_value)
                logger.info(f"Field {field}, Hash {hash_value}: In Weaviate? {hash_in_weaviate}")
                
                # Try to retrieve vector from Weaviate
                if hash_in_weaviate:
                    vector = retrieve_vector_from_weaviate(client, hash_value)
                    has_vector = vector is not None
                    vector_len = len(vector) if has_vector else 0
                    logger.info(f"Field {field}, Hash {hash_value}: Retrieved vector? {has_vector} (length: {vector_len})")
        
        # Check if record has embeddings
        if record_embeddings and record_id in record_embeddings:
            fields_with_embeddings = list(record_embeddings[record_id].keys())
            logger.info(f"Record has embeddings for fields: {fields_with_embeddings}")
        else:
            logger.info(f"Record has no embeddings in record_embeddings")

    # Print overall statistics
    logger.info("\n--- OVERALL STATISTICS ---")
    logger.info(f"Ground truth records: {len(record_ids)}")
    logger.info(f"Records with field hashes: {len(record_ids & set(record_field_hashes.keys()))}")
    logger.info(f"Records with embeddings: {len(record_ids & set(record_embeddings.keys())) if record_embeddings else 0}")
    
    # Check specific embedding functionality
    # 1. Print record_embeddings format
    if record_embeddings and len(record_embeddings) > 0:
        sample_key = next(iter(record_embeddings.keys()))
        sample_value = record_embeddings[sample_key]
        logger.info(f"\nSample record_embeddings entry: {sample_key}")
        logger.info(f"Type: {type(sample_value)}")
        logger.info(f"Keys: {list(sample_value.keys())}")
        
        for field, vector in sample_value.items():
            if vector is not None:
                logger.info(f"Field {field} vector type: {type(vector)}, length: {len(vector)}")
    
    # Suggest next steps based on findings
    logger.info("\n--- RECOMMENDATIONS ---")
    if record_embeddings and len(record_ids & set(record_embeddings.keys())) / len(record_ids) < 0.5:
        logger.info("ISSUE DETECTED: Less than 50% of ground truth records have embeddings")
        logger.info("SOLUTION: Run the index stage again or manually create embeddings for ground truth records")
    
    if embeddings and len(embeddings) > 0 and (not record_embeddings or len(record_embeddings) == 0):
        logger.info("ISSUE DETECTED: Global embeddings exist but no record embeddings")
        logger.info("SOLUTION: Run the index stage to create record embeddings")
    
    if client and not record_embeddings:
        logger.info("ISSUE DETECTED: Connected to Weaviate but no record embeddings found")
        logger.info("SOLUTION: Check if records are indexed in Weaviate or run the index stage again")
    
    logger.info("\nAnalysis completed.")

if __name__ == "__main__":
    main()
