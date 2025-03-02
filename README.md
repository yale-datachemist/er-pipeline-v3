# Entity Resolution for Yale University Library Catalog

This system performs entity resolution on personal names in the Yale University Library catalog using vector embeddings, approximate nearest neighbor search, and machine learning classification.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Development Mode](#development-mode)
- [Production Mode](#production-mode)
- [Output Format](#output-format)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Overview

The entity resolution pipeline identifies and clusters personal name entities that refer to the same real-world individuals across catalog records. It accomplishes this through:

1. Vector representation of entities using 1,536-dimensional embeddings
2. Approximate nearest neighbor (ANN) search using Weaviate
3. Feature engineering with name analysis, temporal reasoning, and vector similarities
4. Classification using logistic regression
5. Graph-based entity clustering

## System Architecture

The system consists of the following components:

- **Data Preprocessing**: Normalizes and deduplicates text fields to optimize processing
- **Vector Embedding**: Generates embeddings using OpenAI's `text-embedding-3-small` model
- **Weaviate Integration**: Indexes embeddings for efficient similarity search
- **Feature Engineering**: Constructs feature vectors for record pairs
- **Classification**: Trains and applies a logistic regression classifier
- **Null Value Imputation**: Handles missing fields using vector-based hot deck imputation
- **Entity Clustering**: Forms identity clusters using graph-based methods

## Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Weaviate)
- 32GB+ RAM (recommended)
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/yale-entity-resolution.git
   cd yale-entity-resolution
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Weaviate using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Configure the system (see Configuration section)

## Configuration

Edit the `config.yaml` file to configure the pipeline:

- **General settings**: Mode, directories, API keys
- **Dataset limits**: For development vs. production
- **Embedding settings**: Model, batch sizes
- **Weaviate settings**: URL, connection parameters
- **Classification settings**: Learning rate, regularization
- **Clustering settings**: Match threshold, neighbor limits

### Environment Variables

You can also set environment variables to override configuration:

- `OPENAI_API_KEY`: Your OpenAI API key
- `WEAVIATE_URL`: URL for the Weaviate instance
- `MEM_LIMIT`: Memory limit for Weaviate (e.g., "8Gi")

## Usage

### Basic Usage

Run the complete pipeline:

```bash
python run_pipeline.py --config config.yaml
```

### Command Line Options

```
usage: run_pipeline.py [-h] [--config CONFIG] [--mode {dev,production}]
                      [--stage {all,preprocess,embed,index,train,cluster}]
                      [--weaviate_url WEAVIATE_URL] [--skip_preprocessing]
                      [--max_records MAX_RECORDS] [--output_dir OUTPUT_DIR]

arguments:
  --config CONFIG       Path to configuration file
  --mode {dev,production}
                        Override mode in config file
  --stage {all,preprocess,embed,index,train,cluster}
                        Pipeline stage to run
  --weaviate_url WEAVIATE_URL
                        Weaviate endpoint URL
  --skip_preprocessing  Skip preprocessing and use cached data
  --max_records MAX_RECORDS
                        Maximum number of records to process
  --output_dir OUTPUT_DIR
                        Directory for output files
```

## Pipeline Stages

The pipeline can be run in stages for more control:

1. **Preprocessing**:
   ```bash
   python run_pipeline.py --stage preprocess
   ```

2. **Embedding**:
   ```bash
   python run_pipeline.py --stage embed
   ```

3. **Indexing**:
   ```bash
   python run_pipeline.py --stage index
   ```

4. **Training**:
   ```bash
   python run_pipeline.py --stage train
   ```

5. **Clustering**:
   ```bash
   python run_pipeline.py --stage cluster
   ```

### Execution Order

When running individual stages, follow this order:
1. Preprocessing
2. Embedding
3. Indexing
4. Training
5. Clustering

## Development Mode

Development mode processes a subset of the data for rapid iteration:

```bash
python run_pipeline.py --mode dev --max_records 10000
```

This mode:
- Uses a smaller sample size
- Reduces batch sizes
- Limits LLM API calls
- Increases checkpoint frequency

## Production Mode

Production mode processes the entire dataset with optimal settings:

```bash
python run_pipeline.py --mode production
```

This mode:
- Processes all records
- Uses larger batch sizes
- Allocates more resources
- Optimizes for throughput

## Output Format

The pipeline generates several outputs:

1. **Entity clusters** (`entity_clusters.jsonl`):
   - JSON Lines format with one cluster per line
   - Contains canonical names, confidence scores, and record IDs

2. **Statistics** (`pipeline_stats.json`):
   - Performance metrics for each pipeline stage
   - Timing information

3. **Model files**:
   - Trained classifier (`classifier_model.pkl`)
   - Feature scaler (`feature_scaler.pkl`)

4. **Logs** (`entity_resolution.log`):
   - Detailed logging information

### Cluster Format Example

```json
{
  "cluster_id": "cluster_42",
  "canonical_name": "Smith, John",
  "life_dates": {"birth_year": "1856", "death_year": "1915"},
  "size": 8,
  "confidence": 0.92,
  "records": ["123456#Agent700-32", "789012#Agent100-11", ...],
  "record_details": [...]
}
```

## Performance Tuning

### Memory Management

- **Development**: 32GB RAM minimum recommended
- **Production**: 256GB RAM recommended for full dataset

Adjust in `docker-compose.yml`:
```bash
MEM_LIMIT=16Gi docker-compose up -d
```

### Parallel Processing

Modify `config.yaml` to adjust:
- `embedding_workers`: Number of parallel embedding workers
- `weaviate_batch_size`: Records per batch for indexing

### Embedding Cache

The system caches embeddings to avoid redundant API calls:
- `use_cached_embeddings: true` in config
- Stored in `output/embeddings.npz`

## Troubleshooting

### Common Issues

1. **Weaviate Connection Error**:
   - Check that Docker is running
   - Verify port 8080 is accessible
   - Check container logs: `docker-compose logs weaviate`

2. **Out of Memory**:
   - Reduce `max_records` or `batch_size` values
   - Increase Docker container memory limit
   - Use development mode

3. **OpenAI API Rate Limits**:
   - Adjust rate limiting parameters in config
   - Implement exponential backoff (built-in)

### Logs

Check the log file for detailed information:
```bash
tail -f entity_resolution.log
```

## File Structure

```
.
├── README.md                     # This documentation
├── config.yaml                   # Configuration file
├── docker-compose.yml            # Weaviate deployment
├── run_pipeline.py               # Main entry point
├── main_pipeline.py              # Pipeline orchestration
├── preprocessing.py              # Data preprocessing
├── embedding.py                  # Vector embedding
├── weaviate_integration.py       # Weaviate connector
├── feature_engineering.py        # Feature engineering
├── imputation_clustering.py      # Imputation and clustering
├── requirements.txt              # Dependencies
├── data/                         # Input data directory
│   └── ground_truth.csv         # Ground truth labels
├── output/                       # Output directory
│   ├── entity_clusters.jsonl    # Entity clusters
│   ├── pipeline_stats.json      # Performance stats
│   ├── classifier_model.pkl     # Trained classifier
│   └── embeddings.npz           # Cached embeddings
└── checkpoints/                  # Checkpoints directory
```

## Requirements

The system requires the following Python packages:

```
pandas==2.0.0
numpy==1.24.3
scikit-learn==1.2.2
openai==1.7.0
weaviate-client==4.0.0
PyYAML==6.0
networkx==3.1
tqdm==4.65.0
python-Levenshtein==0.21.0
```

See `requirements.txt` for a complete list.
