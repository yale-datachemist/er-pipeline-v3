#!/usr/bin/env python
import argparse
import logging
import os
import sys
from main_pipeline import EntityResolutionPipeline

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline for Yale University Library Catalog')
    
    parser.add_argument('--config', type=str, default='config.yaml',
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
    
    parser.add_argument('--max_records', type=int, default=None,
                        help='Maximum number of records to process')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for output files')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = EntityResolutionPipeline(args.config)
    
    # Override configuration settings from command line
    if args.mode:
        pipeline.config['mode'] = args.mode
    
    if args.weaviate_url:
        pipeline.config['weaviate_url'] = args.weaviate_url
    
    if args.max_records:
        pipeline.config['max_records'] = args.max_records
    
    if args.output_dir:
        pipeline.config['output_dir'] = args.output_dir
        pipeline.output_dir = args.output_dir
        os.makedirs(pipeline.output_dir, exist_ok=True)
    
    # Run specific pipeline stages based on arguments
    if args.stage == 'all':
        # Run full pipeline
        if args.skip_preprocessing:
            logger.info("Skipping preprocessing, using cached data")
        else:
            pipeline.preprocess_data()
        
        pipeline.generate_embeddings()
        pipeline.setup_weaviate()
        pipeline.index_records()
        
        train_records, test_records, ground_truth_pairs = pipeline.prepare_training_data()
        pipeline.train_classifier(train_records, test_records, ground_truth_pairs)
        
        pipeline.initialize_imputer_clusterer()
        pipeline.run_clustering()
        
    elif args.stage == 'preprocess':
        # Run only preprocessing
        pipeline.preprocess_data()
        
    elif args.stage == 'embed':
        # Run only embedding generation
        pipeline.generate_embeddings()
        
    elif args.stage == 'index':
        # Run setup and indexing
        pipeline.setup_weaviate()
        pipeline.index_records()
        
    elif args.stage == 'train':
        # Run classifier training
        pipeline.setup_weaviate()
        train_records, test_records, ground_truth_pairs = pipeline.prepare_training_data()
        pipeline.train_classifier(train_records, test_records, ground_truth_pairs)
        
    elif args.stage == 'cluster':
        # Run clustering
        pipeline.setup_weaviate()
        pipeline.initialize_imputer_clusterer()
        pipeline.run_clustering()
    
    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()
