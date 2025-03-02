Suggested steps to test the fixes to your entity resolution pipeline:

1. First, check that your configuration file is properly set up:
   - Ensure the data_dir points to where your CSV files are located
   - Make sure ground_truth_file points to a valid CSV file
   - Verify that csv_delimiter is set to "," in the config

2. Incremental testing approach:
   
   a. Start by testing just the preprocessing stage:
   
   ```bash
   python run_pipeline.py --stage preprocess --config config.yml
   ```
   
   Check the logs to ensure:
   - CSV files are being found in the data directory
   - Files are being read with the comma delimiter
   - Records are being processed
   - Unique strings are being extracted
   
   b. Next, test the embedding generation:
   
   ```bash
   python run_pipeline.py --stage embed --config config.yml
   ```
   
   Check that embeddings are being generated and saved.
   
   c. Test the Weaviate setup and indexing:
   
   ```bash
   python run_pipeline.py --stage index --config config.yml
   ```
   
   Verify that Weaviate is connecting and data is being indexed.

   d. Test the classifier training:
   
   ```bash
   python run_pipeline.py --stage train --config config.yml
   ```
   
   Check that training and test data are being prepared correctly.