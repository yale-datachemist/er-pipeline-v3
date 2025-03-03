#!/usr/bin/env python3
"""
Diagnostic script to check embeddings in checkpoints and debug their structure.
"""

import os
import sys
import pickle
import json
import logging
import numpy as np
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_file(file_path):
    """Inspect a file to determine its type and basic info."""
    try:
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"File: {file_path}")
        logger.info(f"Size: {file_size} bytes")
        logger.info(f"Extension: {file_ext}")
        
        return file_ext
    except Exception as e:
        logger.error(f"Error inspecting file: {e}")
        return None

def check_npz_file(file_path):
    """Check the contents of a NumPy .npz file."""
    try:
        logger.info(f"Loading NPZ file: {file_path}")
        with np.load(file_path, allow_pickle=True) as data:
            # Get the keys (file names in the archive)
            keys = data.files
            logger.info(f"NPZ contains {len(keys)} arrays")
            
            # Sample some keys
            sample_size = min(5, len(keys))
            logger.info(f"Sample keys: {keys[:sample_size]}")
            
            # Check first few arrays
            for key in keys[:sample_size]:
                array = data[key]
                logger.info(f"  {key}: shape={array.shape}, dtype={array.dtype}")
                
                # If it's a 1D array with object dtype, it might be pickle-serialized
                if array.dtype == 'object' and array.ndim == 0:
                    logger.info("  This is a scalar object array (possibly pickle-serialized)")
                    # Try to convert it to a regular Python object
                    try:
                        obj = array.item()
                        logger.info(f"  Converted to Python object of type: {type(obj)}")
                        
                        if isinstance(obj, dict):
                            logger.info(f"  Dictionary with {len(obj)} keys")
                            sample_dict_keys = list(obj.keys())[:5]
                            logger.info(f"  Sample dictionary keys: {sample_dict_keys}")
                            
                            # If there are values, check their type
                            if obj:
                                first_key = next(iter(obj.keys()))
                                first_val = obj[first_key]
                                logger.info(f"  First value type: {type(first_val)}")
                                
                                # If the value is a dict, check its structure
                                if isinstance(first_val, dict):
                                    logger.info(f"  First value is a dict with keys: {list(first_val.keys())}")
                        elif isinstance(obj, list):
                            logger.info(f"  List with {len(obj)} elements")
                            if obj:
                                logger.info(f"  First element type: {type(obj[0])}")
                    except Exception as e:
                        logger.warning(f"  Error converting array.item(): {e}")
                
                # If it's a 1D array, show a sample of values
                elif array.ndim == 1:
                    sample_values = array[:3].tolist()
                    logger.info(f"  Sample values: {sample_values}")
            
            return True
    except Exception as e:
        logger.error(f"Error reading NPZ file: {e}")
        return False

def check_pkl_file(file_path):
    """Check the contents of a pickle file."""
    try:
        logger.info(f"Loading pickle file: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded pickle object of type: {type(data)}")
        
        if isinstance(data, dict):
            logger.info(f"Dictionary with {len(data)} keys")
            sample_keys = list(data.keys())[:5]
            logger.info(f"Sample keys: {sample_keys}")
            
            # Check a few values to understand their structure
            if data:
                key = next(iter(data.keys()))
                value = data[key]
                logger.info(f"Value for key '{key}' is of type {type(value)}")
                
                # If it's a dictionary, check its keys
                if isinstance(value, dict):
                    logger.info(f"Fields available for this record: {list(value.keys())}")
                    
                    # Check one field's vector if available
                    for field in ['person', 'record', 'title']:
                        if field in value:
                            logger.info(f"Field '{field}' has type: {type(value[field])}")
                            
                            # If it's a list/array, check its size
                            if hasattr(value[field], '__len__'):
                                logger.info(f"Field '{field}' has length: {len(value[field])}")
                            
                            break
                
                # Display some stats on values
                field_counts = {}
                empty_values = 0
                for record_id, record_data in data.items():
                    if not record_data:  # Empty record
                        empty_values += 1
                        continue
                        
                    # Count available fields
                    for field in record_data:
                        field_counts[field] = field_counts.get(field, 0) + 1
                
                logger.info(f"Field distribution across {len(data)} records:")
                for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {field}: {count} records ({count/len(data)*100:.1f}%)")
                
                if empty_values > 0:
                    logger.warning(f"Found {empty_values} empty records ({empty_values/len(data)*100:.1f}%)")
        
        elif isinstance(data, list):
            logger.info(f"List with {len(data)} elements")
            if data:
                logger.info(f"First element type: {type(data[0])}")
        
        return True
    except Exception as e:
        logger.error(f"Error reading pickle file: {e}")
        return False

def check_json_file(file_path):
    """Check the contents of a JSON file."""
    try:
        logger.info(f"Loading JSON file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON object of type: {type(data)}")
        
        if isinstance(data, dict):
            logger.info(f"Dictionary with {len(data)} keys")
            sample_keys = list(data.keys())[:5]
            logger.info(f"Sample keys: {sample_keys}")
            
            # Check a few values to understand their structure
            if data:
                key = next(iter(data.keys()))
                value = data[key]
                logger.info(f"Value for key '{key}' is of type {type(value)}")
        
        elif isinstance(data, list):
            logger.info(f"List with {len(data)} elements")
            if data:
                logger.info(f"First element type: {type(data[0])}")
        
        return True
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        return False

def check_checkpoint_file(file_path):
    """Dispatch to appropriate handler based on file extension."""
    ext = inspect_file(file_path)
    
    if ext == '.npz':
        return check_npz_file(file_path)
    elif ext == '.pkl':
        return check_pkl_file(file_path)
    elif ext == '.json':
        return check_json_file(file_path)
    else:
        logger.warning(f"Unsupported file extension: {ext}")
        return False

def check_embedding_files(directory):
    """Check all potential embedding files in a directory."""
    logger.info(f"Checking directory: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return False
    
    # Look for common embedding file patterns
    embedding_files = []
    
    # Check in output dir
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if "embedding" in filename.lower() or filename == "record_embeddings.pkl":
                embedding_files.append(filepath)
    
    # Also check in checkpoint dir if different
    checkpoint_dir = os.path.join(directory, "checkpoints")
    if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            filepath = os.path.join(checkpoint_dir, filename)
            if os.path.isfile(filepath):
                if "embedding" in filename.lower() or filename == "record_embeddings.pkl":
                    embedding_files.append(filepath)
    
    if not embedding_files:
        logger.warning("No embedding files found")
        return False
    
    logger.info(f"Found {len(embedding_files)} potential embedding files:")
    for filepath in embedding_files:
        logger.info(f"  {filepath}")
    
    # Check each file
    for filepath in embedding_files:
        logger.info("\n" + "="*60)
        check_checkpoint_file(filepath)
    
    return True

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Check embedding files for debugging")
    parser.add_argument("--file", help="Specific file to check")
    parser.add_argument("--dir", default="output", help="Directory to search for embedding files")
    args = parser.parse_args()
    
    if args.file:
        check_checkpoint_file(args.file)
    else:
        check_embedding_files(args.dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())