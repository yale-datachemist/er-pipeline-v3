"""
Integration module for the classification reporting system.
This module provides functions to integrate reporting capabilities into the entity resolution pipeline.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import your classification reporting module
from classification_reporting import ClassificationReporter

logger = logging.getLogger(__name__)

def integrate_classification_reporting(pipeline_instance):
    """
    Patches the necessary methods in the pipeline to add reporting functionality.
    
    Args:
        pipeline_instance: Instance of EntityResolutionPipeline to patch
    
    Returns:
        The patched pipeline instance
    """
    # Create a reporter instance if it doesn't exist
    if not hasattr(pipeline_instance, 'reporter'):
        reports_dir = os.path.join(pipeline_instance.output_dir, 'reports', 'classification')
        os.makedirs(reports_dir, exist_ok=True)
        pipeline_instance.reporter = ClassificationReporter(pipeline_instance.config, reports_dir)
        logger.info(f"Created classification reporter at {reports_dir}")
    
    # Patch the train_classifier method
    original_train_classifier = pipeline_instance.train_classifier
    
    def patched_train_classifier(self, train_records, test_records, ground_truth_pairs):
        """
        Patched version of train_classifier that adds reporting capabilities.
        """
        logger.info("Running patched train_classifier with reporting")
        
        # Call the original method to ensure normal operation
        result = original_train_classifier(train_records, test_records, ground_truth_pairs)
        
        # Now add reporting functionality
        try:
            # Generate feature vectors for reporting (this won't have been done in the original method)
            feature_engineering = self.feature_engineer
            classifier = self.classifier
            
            # Get the train and test pairs used in the original method
            if hasattr(self, '_train_pairs'):
                train_pairs = self._train_pairs
            else:
                # Need to recreate train pairs if not stored
                train_pairs = []
                for left_id, right_id, is_match in ground_truth_pairs:
                    if left_id in train_records and right_id in train_records:
                        train_pairs.append((train_records[left_id], train_records[right_id], is_match))
            
            if hasattr(self, '_test_pairs'):
                test_pairs = self._test_pairs
            else:
                # Need to recreate test pairs if not stored
                test_pairs = []
                for left_id, right_id, is_match in ground_truth_pairs:
                    if left_id in test_records and right_id in test_records:
                        test_pairs.append((test_records[left_id], test_records[right_id], is_match))
                        
            # Focus on the test set for reporting
            test_record_pairs = [(pair[0], pair[1]) for pair in test_pairs]
            test_labels = [1 if pair[2] else 0 for pair in test_pairs]
            
            # Generate feature vectors for test pairs
            # This is a simplified version - your actual feature engineering might be different
            test_feature_dicts = []
            for record1, record2, _ in test_pairs:
                record1_id = record1.get("id")
                record2_id = record2.get("id")
                
                embeddings1 = self.record_embeddings.get(record1_id, {})
                embeddings2 = self.record_embeddings.get(record2_id, {})
                
                features = feature_engineering.generate_features(record1, record2, embeddings1, embeddings2)
                test_feature_dicts.append(features)
            
            # Convert feature dictionaries to a feature matrix
            X_test, feature_names = feature_engineering.prepare_test_features(test_feature_dicts)
            
            # Get predictions on test set
            test_probs = classifier.predict_proba(X_test)
            test_preds = classifier.predict(X_test)
            
            # Add model info for reporting
            model_info = {
                "feature_importance": getattr(classifier, 'feature_importance', {}),
                "training_stats": getattr(classifier, 'training_stats', {}),
                "evaluation_stats": getattr(classifier, 'evaluation_stats', {})
            }
            
            # Send data to reporter
            self.reporter.clear_data()
            self.reporter.add_feature_vectors(
                feature_vectors=X_test,
                feature_names=feature_names,
                record_pairs=test_record_pairs,
                labels=test_labels,
                predictions=test_preds,
                prediction_probs=test_probs
            )
            self.reporter.add_model_info(model_info)
            
            # Generate all reports
            self.reporter.generate_feature_vector_report()
            self.reporter.generate_feature_statistics()
            self.reporter.generate_performance_metrics()
            self.reporter.generate_error_analysis()
            
            # Generate plots
            self.reporter.plot_feature_distributions()
            self.reporter.plot_confusion_matrix()
            self.reporter.plot_feature_importance()
            self.reporter.plot_roc_curve()
            
            logger.info("Generated comprehensive classification reports")
            
        except Exception as e:
            logger.error(f"Error generating classification reports: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
    
    # Replace the original method with the patched version
    pipeline_instance.train_classifier = patched_train_classifier.__get__(pipeline_instance)
    
    # Add a prepare_test_features method to FeatureEngineer if needed
    if not hasattr(pipeline_instance.feature_engineer, 'prepare_test_features'):
        def prepare_test_features(self, feature_dicts):
            """
            Prepare test features from feature dictionaries.
            
            Args:
                feature_dicts: List of feature dictionaries
                
            Returns:
                Tuple of (feature_matrix, feature_names)
            """
            import numpy as np
            
            # Get all feature names from dictionaries
            all_feature_names = set()
            for feature_dict in feature_dicts:
                all_feature_names.update(feature_dict.keys())
            
            feature_names = sorted(all_feature_names)
            
            # Create feature matrix
            X = np.zeros((len(feature_dicts), len(feature_names)))
            
            for i, feature_dict in enumerate(feature_dicts):
                for j, feature_name in enumerate(feature_names):
                    X[i, j] = feature_dict.get(feature_name, 0.0)
            
            # Scale features if a scaler is available
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'transform'):
                X_scaled = self.scaler.transform(X)
                return X_scaled, feature_names
            else:
                return X, feature_names
        
        pipeline_instance.feature_engineer.prepare_test_features = prepare_test_features.__get__(pipeline_instance.feature_engineer)
    
    logger.info("Successfully integrated classification reporting into the pipeline")
    return pipeline_instance

# Additional helper function to store intermediate data during training
def patch_feature_engineering(pipeline_instance):
    """
    Patches the feature engineering component to store more data for reporting.
    
    Args:
        pipeline_instance: Instance of EntityResolutionPipeline
    """
    feature_engineer = pipeline_instance.feature_engineer
    
    # Patch the prepare_training_data method to store pairs
    original_prepare_training_data = feature_engineer.prepare_training_data
    
    def patched_prepare_training_data(self, train_pairs, record_embeddings):
        # Store the train pairs for later use in reporting
        if hasattr(pipeline_instance, '_train_pairs'):
            pipeline_instance._train_pairs = train_pairs
            
        # Call the original method
        return original_prepare_training_data(train_pairs, record_embeddings)
    
    feature_engineer.prepare_training_data = patched_prepare_training_data.__get__(feature_engineer)
    
    return pipeline_instance

def integrate_reporting_to_main_pipeline():
    """
    Function to be called from run_pipeline.py to integrate reporting.
    
    Example usage:
        import reporting_integration
        reporting_integration.integrate_reporting_to_main_pipeline()
    """
    try:
        # Import the pipeline module - assuming it's imported in the main script
        import main_pipeline
        
        # The main pipeline's train_classifier method needs to store its train/test pairs
        original_train_classifier = main_pipeline.EntityResolutionPipeline.train_classifier
        
        def patched_train_classifier(self, train_records, test_records, ground_truth_pairs):
            # Create train and test pairs and store them
            train_pairs = []
            for left_id, right_id, is_match in ground_truth_pairs:
                if left_id in train_records and right_id in train_records:
                    train_pairs.append((train_records[left_id], train_records[right_id], is_match))
            
            self._train_pairs = train_pairs
            
            test_pairs = []
            for left_id, right_id, is_match in ground_truth_pairs:
                if left_id in test_records and right_id in test_records:
                    test_pairs.append((test_records[left_id], test_records[right_id], is_match))
            
            self._test_pairs = test_pairs
            
            # Call the original method
            return original_train_classifier(self, train_records, test_records, ground_truth_pairs)
        
        # Replace the original method
        main_pipeline.EntityResolutionPipeline.train_classifier = patched_train_classifier
        
        # Modify the run method to integrate reporting for each pipeline instance
        original_run_pipeline = main_pipeline.EntityResolutionPipeline.run_pipeline
        
        def patched_run_pipeline(self):
            # Integrate reporting before running
            integrate_classification_reporting(self)
            patch_feature_engineering(self)
            
            # Call the original method
            return original_run_pipeline(self)
        
        # Replace the original method
        main_pipeline.EntityResolutionPipeline.run_pipeline = patched_run_pipeline
        
        logger.info("Successfully integrated reporting into main pipeline module")
        return True
    except Exception as e:
        logger.error(f"Error integrating reporting: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False