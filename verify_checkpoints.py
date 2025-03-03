#!/usr/bin/env python
"""
Script to manually verify checkpoint data and identify potential issues
between the preprocessing and embedding stages.
"""

import os
import json
import argparse
from collections import defaultdict

def check_checkpoint(checkpoint_dir):
    """Check checkpoint files for potential issues."""
    print(f"\n=== Checking checkpoint data in: {checkpoint_dir} ===\n")
    
    # Check if directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    # Check for required files
    required_files = [
        "unique_strings.json",
        "string_counts.json",
        "field_types.json",
        "record_field_hashes.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(checkpoint_dir, f))]
    
    if missing_files:
        print(f"Warning: Missing checkpoint files: {', '.join(missing_files)}")
    
    # Load and check unique strings
    unique_strings_path = os.path.join(checkpoint_dir, "unique_strings.json")
    if os.path.exists(unique_strings_path):
        try:
            with open(unique_strings_path, 'r') as f:
                unique_strings = json.load(f)
            print(f"✓ Unique strings: {len(unique_strings)} items")
            
            # Check a sample
            if unique_strings:
                print("\nSample unique strings (first 3):")
                sample_keys = list(unique_strings.keys())[:3]
                for key in sample_keys:
                    print(f"  - {key}: {unique_strings[key][:50]}...")
            else:
                print("❌ No unique strings found!")
        except Exception as e:
            print(f"❌ Error loading unique strings: {e}")
            return False
    else:
        print("❌ Unique strings file not found")
        return False
    
    # More verification checks...
    
    print("\n=== Checkpoint verification complete ===\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify checkpoint data for entity resolution pipeline")
    parser.add_argument('--checkpoint_dir', type=str, default='output/checkpoints',
                        help='Path to checkpoint directory')
    args = parser.parse_args()
    
    check_checkpoint(args.checkpoint_dir)

if __name__ == "__main__":
    main()