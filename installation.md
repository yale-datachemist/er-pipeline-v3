# Entity Resolution Pipeline: Installation and Running Guide

This guide will help you set up and run the entity resolution pipeline for Yale University Library catalog data.

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Weaviate)
- OpenAI API key (for generating embeddings)

## 1. Installation

### 1.1 Clone the Repository (if applicable)
```bash
git clone https://github.com/your-repo/yale-entity-resolution.git
cd yale-entity-resolution
```

### 1.2 Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Set Up Weaviate
Start Weaviate using Docker Compose:
```bash
docker-compose up -d
```

Verify Weaviate is running:
```bash
docker ps | grep weaviate
```

## 2. Configuration

### 2.1 Edit configuration file
Update the `config.yml` file with your settings:

- Set `data_dir` to point to your CSV data files
- Set `ground_truth_file` to point to your ground truth file
- Ensure `csv_delimiter` is set to "," for comma-delimited files
- Set your OpenAI API key in `openai_api_key`

### 2.2 OpenAI API Key
You can either:
- Set it in the config file as mentioned above, OR
- Use an environment variable:
  ```bash
  export OPENAI_API_KEY=your-api-key-here
  ```
- Pass it as a command-line argument:
  ```bash
  python run_pipeline.py --openai_api_key your-api-key-here
  ```

## 3. Data Preparation

Ensure your data is properly formatted:
- CSV files should be comma-delimited
- Ground truth file should have `left`, `right`, and `match` columns

Verify your data files:
```bash
python run_pipeline.py --verify_data --config config.yml
```

This will check that:
- Data directories exist
- CSV files are found and properly formatted
- Ground truth file exists with required columns

## 4. Running the Pipeline

### 4.1 Run the Complete Pipeline
```bash
python run_pipeline.py --config config.yml
```

### 4.2 Run Individual Stages
You can run specific stages of the pipeline for testing:

```bash
# Preprocessing stage
python run_pipeline.py --stage preprocess --config config.yml

# Embedding generation
python run_pipeline.py --stage embed --config config.yml

# Weaviate indexing
python run_pipeline.py --stage index --config config.yml

# Classifier training
python run_pipeline.py --stage train --config config.yml

# Entity clustering
python run_pipeline.py --stage cluster --config config.yml
```

### 4.3 Additional Options

```bash
# Force rerun all stages (ignore checkpoints)
python run_pipeline.py --force_rerun --config config.yml

# Limit number of records for faster testing
python run_pipeline.py --max_records 1000 --config config.yml

# Skip preprocessing (use cached data)
python run_pipeline.py --skip_preprocessing --config config.yml

# Specify output directory
python run_pipeline.py --output_dir ./my_results --config config.yml

# Run in development mode for faster testing
python run_pipeline.py --mode dev --config config.yml
```

## 5. Output Files

After successful execution, the pipeline generates:

- `entity_clusters.jsonl`: The entity clusters identified in the data
- `pipeline_stats.json`: Statistics and metrics about the pipeline execution
- `classifier_model.pkl`: The trained classifier model
- `feature_scaler.pkl`: The feature scaler used for normalization
- Detailed logs with timestamps in the root directory

## 6. Troubleshooting

### 6.1 View Detailed Logs
The pipeline creates detailed log files with timestamps in the root directory:
```bash
cat entity_resolution_YYYYMMDD_HHMMSS.log
```

### 6.2 Common Issues

- **CSV parsing errors**: Check that your files are correctly formatted with commas as delimiters
- **OpenAI API errors**: Verify your API key and check rate limits
- **Weaviate connection errors**: Make sure Weaviate is running and accessible
- **Ground truth parsing errors**: Ensure your ground truth file has the required columns

### 6.3 Restart Weaviate
If you encounter Weaviate connection issues, restart it:
```bash
docker-compose down
docker-compose up -d
```

## 7. Pipeline Performance Tuning

For larger datasets or improved performance:

- Adjust batch sizes in config.yml
- Increase/decrease parallelism based on your hardware
- Tune match threshold for clustering
- Adjust field weights for different entity matching criteria

## 8. Cleaning Up

If you need to start fresh:
```bash
# Clear Weaviate data
docker-compose down -v

# Remove checkpoints and outputs
rm -rf output/*
rm -rf checkpoints/*
```