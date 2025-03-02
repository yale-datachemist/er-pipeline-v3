# Debugging Guide for Entity Resolution Pipeline

This guide provides comprehensive instructions for testing and debugging the entity resolution pipeline after applying the fixes.

## 1. Configuration and Setup

### 1.1 First, check your configuration file
Ensure your `config.yml` has the correct settings:
- `data_dir`: Points to where your CSV files are located
- `ground_truth_file`: Points to a valid CSV file
- `csv_delimiter`: Set to "," (since your files are comma-delimited)

### 1.2 Check file permissions
Make sure your Python process has read/write permissions for:
- Input data directory
- Output directory
- Checkpoint directory

### 1.3 Weaviate setup
Verify Weaviate is running:
```bash
docker ps | grep weaviate
```

If not running, start it with:
```bash
docker-compose up -d
```

## 2. Incremental Testing

Test each stage individually to identify where issues occur.

### 2.1 Preprocessing Stage
```bash
python run_pipeline.py --stage preprocess --config config.yml
```

Check logs for:
- ✓ Collections created successfully
- ✓ String data indexed properly
- ✓ Entity maps created and indexed
- ✓ Field vectors retrieved for records

### 2.4 Training Data Preparation and Classifier Training
```bash
python run_pipeline.py --stage train --config config.yml
```

Check:
- ✓ Ground truth file is parsed correctly
- ✓ Training and test data are prepared properly
- ✓ Feature engineering works as expected
- ✓ Classifier is trained successfully
- ✓ Evaluation metrics are reasonable

### 2.5 Entity Clustering
```bash
python run_pipeline.py --stage cluster --config config.yml
```

Check:
- ✓ Components are initialized properly
- ✓ Match graph is built successfully
- ✓ Clusters are formed and saved
- ✓ `entity_clusters.jsonl` is created in the output directory

## 3. Common Error Scenarios and Solutions

### 3.1 No CSV Files Found
**Symptoms:**
- Error: "No CSV files found in directory"

**Solutions:**
- Verify the path in `data_dir` is correct
- Check file extensions (should be `.csv`)
- Ensure directory exists and has read permissions

### 3.2 CSV Parsing Errors
**Symptoms:**
- Error: "CSV parsing error"
- Few or no records processed

**Solutions:**
- Verify files are comma-delimited
- Check for malformed CSV data (unescaped quotes, etc.)
- Try inspecting a sample file:
  ```bash
  head -n 5 data/yourfile.csv
  ```

### 3.3 Ground Truth File Issues
**Symptoms:**
- Error: "No ground truth pairs found"
- Error: "Missing required columns"

**Solutions:**
- Ensure the ground truth file has columns: `left`, `right`, and `match`
- Check that match values are in a format that can be interpreted as boolean
- Verify record IDs in ground truth match those in your dataset

### 3.4 Weaviate Connection Issues
**Symptoms:**
- Error: "Failed to connect to Weaviate"

**Solutions:**
- Ensure Weaviate is running
- Check the URL in config
- Try restarting Weaviate:
  ```bash
  docker-compose down && docker-compose up -d
  ```

### 3.5 OpenAI API Issues
**Symptoms:**
- Error: "Error in API request"
- No embeddings generated

**Solutions:**
- Verify API key is correct
- Check API rate limits and quotas
- Reduce batch size to avoid rate limiting

### 3.6 Missing Records in Training
**Symptoms:**
- Warning: "Could not find X records from ground truth in processed data"

**Solutions:**
- Ensure record IDs in ground truth match the format in your dataset
- Check for preprocessing issues that might skip certain records

## 4. Analyzing Pipeline Results

### 4.1 Check Statistics
Examine `pipeline_stats.json` in the output directory to see:
- Number of records processed
- Number of unique strings
- Number of clusters formed
- Classification and clustering metrics

### 4.2 Evaluate Clusters
Review the clusters in `entity_clusters.jsonl`:
- Are the sizes reasonable?
- Check a few clusters manually to verify correctness
- Look at confidence scores - low confidence clusters may need refinement

### 4.3 Refine Parameters
If results aren't satisfactory, try adjusting these parameters:
- `match_threshold` - Lower for more clusters, higher for fewer larger clusters
- `max_neighbors` - Increase to consider more candidates
- `field_weights` - Adjust to prioritize certain fields CSV files found in data directory
- ✓ Files read with comma delimiter
- ✓ Records processed successfully
- ✓ No errors in string deduplication

Look at these files in the checkpoint directory:
- `unique_strings.json`
- `record_field_hashes.json`