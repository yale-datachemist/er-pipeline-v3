#!/usr/bin/env python3
"""
Script to install the classification reporting module into the entity resolution pipeline.
This script makes the necessary changes to integrate comprehensive reporting.
"""

import os
import sys
import logging
from importlib import util
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_file_exists(filename):
    """Check if a file exists and is readable."""
    return os.path.exists(filename) and os.access(filename, os.R_OK)

def create_backup(filename):
    """Create a backup of a file."""
    backup_file = f"{filename}.bak"
    try:
        shutil.copy2(filename, backup_file)
        logger.info(f"Created backup: {backup_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False

def modify_run_pipeline_script():
    """Modify run_pipeline.py to include reporting functionality."""
    run_pipeline_path = "run_pipeline.py"
    
    if not check_file_exists(run_pipeline_path):
        logger.error(f"Could not find run_pipeline.py")
        return False
    
    # Create backup
    if not create_backup(run_pipeline_path):
        return False
    
    try:
        # Read the file
        with open(run_pipeline_path, 'r') as f:
            content = f.read()
        
        # Check if already modified
        if "import reporting_integration" in content:
            logger.info("run_pipeline.py already contains reporting integration")
            return True
        
        # Find the import section
        import_section_end = content.find("# Global reference to pipeline for cleanup handlers")
        if import_section_end == -1:
            # Try another marker
            import_section_end = content.find("def force_close_logging")
            if import_section_end == -1:
                logger.warning("Could not identify import section in run_pipeline.py")
                import_section_end = 0
        
        # Insert our import
        import_statement = "\n# Import reporting integration\nimport reporting_integration\n"
        new_content = content[:import_section_end] + import_statement + content[import_section_end:]
        
        # Find the main function
        main_function_start = new_content.find("def main():")
        if main_function_start == -1:
            logger.error("Could not find main function in run_pipeline.py")
            return False
        
        # Find where to insert the integration code
        initialization_marker = new_content.find("pipeline_instance = EntityResolutionPipeline(", main_function_start)
        if initialization_marker == -1:
            logger.error("Could not find pipeline initialization in main function")
            return False
        
        # Find the line end
        line_end = new_content.find("\n", initialization_marker)
        if line_end == -1:
            logger.error("Could not find end of initialization line")
            return False
        
        # Insert the integration code after pipeline initialization
        integration_code = "\n    # Integrate reporting capabilities\n    reporting_integration.integrate_classification_reporting(pipeline_instance)\n"
        new_content = new_content[:line_end+1] + integration_code + new_content[line_end+1:]
        
        # Write the modified file
        with open(run_pipeline_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Successfully modified {run_pipeline_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error modifying run_pipeline.py: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to install the reporting module."""
    logger.info("Installing classification reporting module")
    
    # Check if classification_reporting.py exists
    if not check_file_exists("classification_reporting.py"):
        logger.error("classification_reporting.py not found. Please make sure it's in the current directory.")
        return 1
    
    # Modify run_pipeline.py
    if not modify_run_pipeline_script():
        logger.error("Failed to modify run_pipeline.py, aborting installation")
        return 1
    
    logger.info("""
Classification reporting module has been successfully integrated!

To use the module:
1. Run your pipeline as usual: python run_pipeline.py --config config.yml --stage train
2. Reports will be generated in: output/reports/classification/

The reports include:
- feature_vectors.csv: Complete feature vectors for each pair
- feature_statistics.csv: Statistics for each feature
- performance_metrics.csv: Performance at different thresholds
- error_analysis.csv: Analysis of misclassified pairs
- Various plots: Feature distributions, confusion matrix, ROC curve, etc.
""")
    return 0

if __name__ == "__main__":
    sys.exit(main())