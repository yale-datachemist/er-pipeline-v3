import argparse
import os
import sys
import json
import signal
import atexit
import gc
import pickle
from datetime import datetime
from main_pipeline import EntityResolutionPipeline
from logging_setup import setup_logging, _log_handlers  # Import the new logging setup

# Set up logging with proper handler management
logger = setup_logging()

# Global reference to pipeline for cleanup handlers
pipeline_instance = None

def force_close_logging():
    """Force close all logging handlers to prevent ResourceWarnings."""
    import logging
    
    # Flush and close all handlers in the root logger
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        try:
            handler.flush()
            handler.close()
            root_logger.removeHandler(handler)
        except Exception:
            pass
    
    # Also close our tracked handlers
    for handler in _log_handlers:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    
    # Clear the handlers list
    _log_handlers.clear()
    
    # Force garbage collection to release file handles
    gc.collect()

def cleanup_handler():
    """Handle cleanup when the program exits."""
    global pipeline_instance
    if pipeline_instance:
        logger.info("Running cleanup during program exit")
        try:
            # Call the cleanup method on the pipeline
            if hasattr(pipeline_instance, '_cleanup'):
                pipeline_instance._cleanup()
            
            # Explicitly close Weaviate connection
            if hasattr(pipeline_instance, 'weaviate_manager') and pipeline_instance.weaviate_manager:
                pipeline_instance.weaviate_manager.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    # Force close all logging handlers at the end
    force_close_logging()

def signal_handler(sig, frame):
    """Handle interruption signals to ensure clean exit."""
    logger.info(f"Received signal {sig}, initiating clean shutdown")
    cleanup_handler()
    sys.exit(0)

# Register the cleanup and signal handlers
atexit.register(cleanup_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline for Yale University Library Catalog')
    
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    
    parser.add_argument('--mode', type=str, choices=['dev', 'production'], default=None,
                        help='Override mode in config file (dev or production)')
    
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'preprocess', 'embed', 'index', 'train', 'cluster'],
                        help='Pipeline stage to run')
    
    parser.add_argument('--weaviate_url', type=str, default=None,
                        help='Weaviate endpoint URL')
    
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Skip preprocessing stage and use cached data')
                        
    parser.add_argument('--skip_embedding', action='store_true',
                        help='Skip embedding generation and only use cached embeddings')
    
    parser.add_argument('--max_records', type=int, default=None,
                        help='Maximum number of records to process')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for output files')
    
    parser.add_argument('--verify_data', action='store_true',
                        help='Verify data files before running the pipeline')
    
    parser.add_argument('--force_rerun', action='store_true',
                        help='Force rerun of all stages, ignoring checkpoints')
                        
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key to use for embeddings')
    
    parser.add_argument('--disable_cleanup', action='store_true',
                        help='Disable automatic cleanup of resources on exit')
    
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for log files')
    
    return parser.parse_args()

def verify_data_files(config):
    """Check data files exist and contain expected content."""
    data_dir = config.get("data_dir")
    ground_truth_file = config.get("ground_truth_file")
    
    status = {}
    
    # Check data directory
    if not os.path.exists(data_dir):
        status["data_dir"] = {"exists": False, "error": f"Directory not found: {data_dir}"}
    else:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        status["data_dir"] = {
            "exists": True, 
            "csv_files_count": len(csv_files),
            "csv_files": csv_files[:5]  # Show first 5 files
        }
        
        # Check content of first CSV file
        if csv_files:
            first_file = os.path.join(data_dir, csv_files[0])
            try:
                with open(first_file, 'r', encoding='utf-8') as f:
                    header = f.readline().strip()
                    first_row = f.readline().strip() if f.readline() else ""
                
                status["sample_csv"] = {
                    "file": csv_files[0],
                    "header": header,
                    "first_row": first_row,
                    "delimiter": "," if "," in header else "tab" if "\t" in header else "unknown"
                }
            except Exception as e:
                status["sample_csv"] = {"error": f"Failed to read file: {str(e)}"}
    
    # Check ground truth file
    if not os.path.exists(ground_truth_file):
        status["ground_truth"] = {"exists": False, "error": f"File not found: {ground_truth_file}"}
    else:
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                first_row = f.readline().strip() if f.readline() else ""
            
            required_cols = ["left", "right", "match"]
            has_required = all(col in header.lower() for col in required_cols)
            
            status["ground_truth"] = {
                "exists": True,
                "header": header,
                "first_row": first_row,
                "has_required_columns": has_required,
                "delimiter": "," if "," in header else "tab" if "\t" in header else "unknown"
            }
        except Exception as e:
            status["ground_truth"] = {"exists": True, "error": f"Failed to read file: {str(e)}"}
    
    # Check output directory
    output_dir = config.get("output_dir")
    if not os.path.exists(output_dir):
        status["output_dir"] = {"exists": False, "note": f"Directory will be created: {output_dir}"}
    else:
        status["output_dir"] = {"exists": True}
    
    # Save verification results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "data_verification.json"), 'w') as f:
        json.dump(status, f, indent=4)
    
    # Print summary
    print("\n--- Data Verification Summary ---")
    all_ok = True
    
    if status.get("data_dir", {}).get("exists", False):
        csv_count = status.get("data_dir", {}).get("csv_files_count", 0)
        if csv_count > 0:
            print(f"✓ Data directory: Found {csv_count} CSV files")
        else:
            print("✗ Data directory: No CSV files found")
            all_ok = False
    else:
        print(f"✗ Data directory: {status.get('data_dir', {}).get('error')}")
        all_ok = False
        
    if status.get("ground_truth", {}).get("exists", False):
        if status.get("ground_truth", {}).get("has_required_columns", False):
            print("✓ Ground truth file: Valid format with required columns")
        else:
            print("✗ Ground truth file: Missing required columns (left, right, match)")
            all_ok = False
    else:
        print(f"✗ Ground truth file: {status.get('ground_truth', {}).get('error')}")
        all_ok = False
    
    sample_csv = status.get("sample_csv", {})
    if "error" not in sample_csv:
        delimiter = sample_csv.get("delimiter", "unknown")
        config_delimiter = config.get("csv_delimiter", ",")
        
        if delimiter == config_delimiter:
            print(f"✓ CSV format: Files use {delimiter} delimiter as configured")
        else:
            print(f"✗ CSV format: Files appear to use {delimiter} delimiter but config specifies {config_delimiter}")
            all_ok = False
    
    return all_ok, status

def main():
    """Main entry point."""
    global pipeline_instance
    
    args = parse_args()
    
    # If log_dir is specified, reconfigure logging to use it
    if args.log_dir:
        global logger
        logger = setup_logging(args.log_dir)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline_instance = EntityResolutionPipeline(args.config)
        
        # Override configuration settings from command line
        if args.mode:
            pipeline_instance.config['mode'] = args.mode
            logger.info(f"Using mode: {args.mode}")
        
        if args.weaviate_url:
            pipeline_instance.config['weaviate_url'] = args.weaviate_url
        
        if args.max_records:
            pipeline_instance.config['max_records'] = args.max_records
            logger.info(f"Using max records: {args.max_records}")
        
        if args.output_dir:
            pipeline_instance.config['output_dir'] = args.output_dir
            pipeline_instance.output_dir = args.output_dir
            os.makedirs(pipeline_instance.output_dir, exist_ok=True)
            logger.info(f"Using output directory: {args.output_dir}")
        
        if args.force_rerun:
            pipeline_instance.config['force_rerun'] = True
            logger.info("Force rerun enabled")
        
        if args.openai_api_key:
            pipeline_instance.config['openai_api_key'] = args.openai_api_key
            logger.info("Using OpenAI API key from command line")
        
        # For embedding stage, check API key
        if args.stage == 'embed' or args.stage == 'all':
            # Check if embedding is actually needed 
            if not args.skip_embedding:
                if not pipeline_instance.check_openai_api_key():
                    logger.error("Cannot proceed with embedding without a valid OpenAI API key")
                    print("\nError: No valid OpenAI API key found!")
                    print("Please provide your key in one of these ways:")
                    print("1. Set it in your config.yml file")
                    print("2. Use the --openai_api_key parameter")
                    print("3. Set the OPENAI_API_KEY environment variable")
                    sys.exit(1)
        
        # Check disable_cleanup flag
        if args.disable_cleanup:
            logger.info("Automatic cleanup on exit has been disabled")
            # Deregister cleanup handler if cleanup is disabled
            atexit.unregister(cleanup_handler)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        
        # Verify data files if requested
        if args.verify_data:
            verify_data_files(pipeline_instance.config)
        
        # Run the selected pipeline stage
        if args.stage == 'all':
            # Run full pipeline
            if args.skip_preprocessing:
                logger.info("Skipping preprocessing, using cached data")
            else:
                pipeline_instance.preprocess_data()
            
            if args.skip_embedding:
                logger.info("Skipping embedding generation, using cached data")
            else:
                pipeline_instance.generate_embeddings()
            
            pipeline_instance.setup_weaviate()
            pipeline_instance.index_strings()
            
            train_records, test_records, ground_truth_pairs = pipeline_instance.prepare_training_data()
            
            pipeline_instance.train_classifier(train_records, test_records, ground_truth_pairs)
            
            pipeline_instance.initialize_imputer_clusterer()
            pipeline_instance.run_clustering()
            
        elif args.stage == 'preprocess':
            # Run only preprocessing
            pipeline_instance.preprocess_data()
            
        elif args.stage == 'embed':
            # Run only embedding generation
            if args.skip_embedding:
                logger.info("Skipping embedding generation, using cached data")
            else:
                pipeline_instance.generate_embeddings()
            
        elif args.stage == 'index':
            # Run setup and indexing
            pipeline_instance.setup_weaviate()
            pipeline_instance.index_strings()
            
        elif args.stage == 'train':
            # Run classifier training
            pipeline_instance.setup_weaviate()
            train_records, test_records, ground_truth_pairs = pipeline_instance.prepare_training_data()
            pipeline_instance.train_classifier(train_records, test_records, ground_truth_pairs)
            
        elif args.stage == 'cluster':
            # Run clustering
            pipeline_instance.setup_weaviate()
            pipeline_instance.initialize_imputer_clusterer()
            pipeline_instance.run_clustering()
        
        logger.info("Pipeline execution completed")
    
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        # Cleanup will be handled by registered handlers
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # If cleanup is not disabled, ensure it runs even if there was an error
        if not args.disable_cleanup:
            cleanup_handler()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure logging handlers are properly closed even if there's an exception
        force_close_logging()