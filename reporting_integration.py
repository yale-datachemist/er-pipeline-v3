"""
Reporting integration module for the entity resolution pipeline.
Provides functions to integrate classification reporting with the main pipeline.
"""

import os
import logging
import numpy as np
from typing import Any, Dict, List, Tuple

# Import the reporting functionality
from classification_reporting import generate_classification_report, extract_misclassified_pairs

# Set up logging
logger = logging.getLogger(__name__)

def integrate_classification_reporting(pipeline: Any) -> None:
    """
    Integrate classification reporting with the entity resolution pipeline.
    
    Args:
        pipeline: EntityResolutionPipeline instance to integrate with
    """
    logger.info("Integrating classification reporting with pipeline")
    
    # Store original train_classifier method
    original_train_classifier = pipeline.train_classifier
    
    # Define wrapper function with reporting
    def train_classifier_with_reporting(train_records: Dict, test_records: Dict, 
                                        ground_truth_pairs: List[Tuple]) -> None:
        # Call original method
        original_train_classifier(train_records, test_records, ground_truth_pairs)
        
        try:
            # Generate reports after training and evaluation
            # At this point, pipeline.stats should have the training and evaluation stats
            output_dir = os.path.join(pipeline.output_dir, "reports", "classification")
            os.makedirs(output_dir, exist_ok=True)
            
            # Build test pairs from test_records and ground_truth for record pairs
            from sklearn.model_selection import train_test_split
            
            # Split ground truth into train/test same way as in original method
            gt_train_pairs, gt_test_pairs = train_test_split(
                ground_truth_pairs,
                test_size=0.2,
                random_state=42,
                stratify=[int(is_match) for _, _, is_match in ground_truth_pairs]
            )
            
            # Create test pairs - ONLY using test records and test ground truth pairs
            test_pairs = []
            for left_id, right_id, is_match in gt_test_pairs:
                if left_id in test_records and right_id in test_records:
                    test_pairs.append((test_records[left_id], test_records[right_id], is_match))
            
            if not test_pairs:
                logger.warning("No valid test pairs available for reporting, skipping report generation")
                return
                
            logger.info(f"Preparing classification reports using {len(test_pairs)} test pairs")
            
            # Prepare test data again
            X_test, y_test, feature_names = pipeline.feature_engineer.prepare_training_data(
                test_pairs, pipeline.record_embeddings
            )
            
            if len(X_test) == 0 or len(y_test) == 0:
                logger.warning("Empty test data, skipping report generation")
                return
            
            # Get predictions
            predictions = pipeline.classifier.predict(X_test)
            prediction_probs = pipeline.classifier.predict_proba(X_test)
            
            # Extract model info from pipeline.stats
            classifier_stats = pipeline.stats.get('classifier', {})
            training_stats = classifier_stats.get('training', {})
            
            model_info = {
                "feature_importance": training_stats.get('feature_importance', {}),
                "iterations": training_stats.get('iterations', 0),
                "final_loss": training_stats.get('final_loss', 0),
                "final_accuracy": training_stats.get('final_accuracy', 0)
            }
            
            # Generate comprehensive report
            reporter = generate_classification_report(
                pipeline.config,
                X_test=X_test,
                y_test=y_test,
                predictions=predictions,
                prediction_probs=prediction_probs,
                feature_names=feature_names,
                record_pairs=test_pairs,
                model_info=model_info,
                output_dir=output_dir
            )
            
            # Generate misclassified pairs report
            misclassified_output = os.path.join(output_dir, "misclassified_pairs.csv")
            misclassified_df = extract_misclassified_pairs(
                X_test, y_test, predictions, feature_names, test_pairs, misclassified_output
            )
            
            logger.info(f"Generated classification reports in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating classification reports: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Replace original method with wrapper
    pipeline.train_classifier = train_classifier_with_reporting
    logger.info("Successfully integrated classification reporting")