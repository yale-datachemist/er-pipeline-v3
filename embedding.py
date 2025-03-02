import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates vector embeddings for text using OpenAI's API.
    """
    def __init__(self, config: Dict):
        """
        Initialize the embedding generator with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.client = OpenAI(api_key=config.get("openai_api_key"))
        self.model = config.get("embedding_model", "text-embedding-3-small")
        self.batch_size = config.get("embedding_batch_size", 100)
        self.max_retries = config.get("max_retries", 5)
        self.retry_delay = config.get("retry_delay", 1)
        
        # Storage for generated embeddings
        self.embeddings: Dict[str, List[float]] = {}
        
        # Rate limiting parameters
        self.requests_per_minute = config.get("requests_per_minute", 10000)
        self.tokens_per_minute = config.get("tokens_per_minute", 5000000)
        self.daily_token_limit = config.get("daily_token_limit", 500000000)
        
        # Tracking for rate limiting
        self.request_timestamps = []
        self.token_counts = []
        self.daily_token_count = 0
        self.daily_reset_time = time.time()
    
    def _check_rate_limits(self, estimated_tokens: int) -> bool:
        """
        Check if the request would exceed rate limits and wait if necessary.
        
        Args:
            estimated_tokens: Estimated number of tokens in the request
            
        Returns:
            Boolean indicating whether the request can proceed
        """
        current_time = time.time()
        
        # Check daily token limit
        if current_time - self.daily_reset_time > 86400:  # 24 hours
            self.daily_token_count = 0
            self.daily_reset_time = current_time
            
        if self.daily_token_count + estimated_tokens > self.daily_token_limit:
            logger.warning("Daily token limit reached, cannot proceed")
            return False
            
        # Clean up old timestamps (older than 1 minute)
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if current_time - ts < 60]
        self.token_counts = [count for count, ts in zip(self.token_counts, self.request_timestamps)
                            if current_time - ts < 60]
        
        # Check requests per minute
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - min(self.request_timestamps))
            if sleep_time > 0:
                logger.info(f"Rate limit approaching, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                
        # Check tokens per minute
        current_tokens_per_minute = sum(self.token_counts)
        if current_tokens_per_minute + estimated_tokens > self.tokens_per_minute:
            sleep_time = 60 - (current_time - min(self.request_timestamps))
            if sleep_time > 0:
                logger.info(f"Token limit approaching, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                
        return True
    
    def _update_rate_tracking(self, tokens_used: int) -> None:
        """
        Update rate tracking after a successful API call.
        
        Args:
            tokens_used: Number of tokens used in the request
        """
        current_time = time.time()
        self.request_timestamps.append(current_time)
        self.token_counts.append(tokens_used)
        self.daily_token_count += tokens_used
    
    def _generate_embedding_batch(self, texts: List[str]) -> Tuple[List[str], List[List[float]]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tuple of (text_hashes, embeddings)
        """
        # Estimate token count (rough approximation)
        estimated_tokens = sum(len(text.split()) * 1.3 for text in texts)
        
        # Check rate limits
        if not self._check_rate_limits(estimated_tokens):
            return [], []
            
        # Generate text hashes
        text_hashes = []
        for text in texts:
            import hashlib
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            text_hashes.append(text_hash)
        
        # Retry logic for API calls
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float"
                )
                
                # Extract embeddings
                embeddings = [data.embedding for data in response.data]
                
                # Verify dimensions
                expected_dim = 1536
                for i, emb in enumerate(embeddings):
                    if len(emb) != expected_dim:
                        logger.error(f"Embedding dimension mismatch: got {len(emb)}, expected {expected_dim}")
                        return [], []
                
                # Update rate tracking with actual token usage
                tokens_used = response.usage.total_tokens
                self._update_rate_tracking(tokens_used)
                
                return text_hashes, embeddings
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit exceeded: {str(e)}. Waiting before retry.")
                import random
                # Add jitter to avoid thundering herd problem
                time.sleep(self.retry_delay * (2 ** attempt) + random.uniform(0, 1))
                
            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt+1} failed: {str(e)}")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
        logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
        return [], []
    
    def generate_embeddings(self, texts: List[str], text_ids: List[str] = None) -> Dict[str, List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            text_ids: Optional list of IDs for the texts
            
        Returns:
            Dictionary mapping text IDs (or hashes) to embeddings
        """
        if not texts:
            return {}
            
        # If no text_ids provided, use indices as IDs
        if text_ids is None:
            text_ids = [str(i) for i in range(len(texts))]
        
        # Create batches
        batches = []
        batch_ids = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            ids = text_ids[i:i+self.batch_size]
            batches.append(batch)
            batch_ids.append(ids)
        
        # Process batches
        result_dict = {}
        
        for batch, ids in tqdm(zip(batches, batch_ids), total=len(batches), desc="Generating embeddings"):
            text_hashes, embeddings = self._generate_embedding_batch(batch)
            
            if len(text_hashes) != len(embeddings):
                logger.warning("Mismatched lengths between hashes and embeddings")
                continue
                
            # Store embeddings with their IDs
            for text_id, embedding in zip(ids, embeddings):
                result_dict[text_id] = embedding
                # Also store in instance variable for persistence
                self.embeddings[text_id] = embedding
        
        return result_dict
    
    def generate_embeddings_for_unique_strings(self, unique_strings: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Generate embeddings only for unique strings.
        
        Args:
            unique_strings: Dictionary mapping hash -> original string
            
        Returns:
            Dictionary mapping hash -> embedding
        """
        logger.info(f"Generating embeddings for {len(unique_strings)} unique strings")
        
        # Prepare inputs
        hashes = list(unique_strings.keys())
        texts = list(unique_strings.values())
        
        # Process in parallel batches
        max_workers = self.config.get("embedding_workers", 4)
        chunk_size = max(1, len(texts) // max_workers)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i:i+chunk_size]
                chunk_hashes = hashes[i:i+chunk_size]
                
                future = executor.submit(self.generate_embeddings, chunk_texts, chunk_hashes)
                futures.append(future)
            
            # Collect results
            embeddings_dict = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing embedding chunks"):
                chunk_results = future.result()
                embeddings_dict.update(chunk_results)
        
        logger.info(f"Generated {len(embeddings_dict)} embeddings")
        return embeddings_dict
    
    def save_embeddings(self, output_file: str) -> None:
        """
        Save the generated embeddings to a file.
        
        Args:
            output_file: Path to the output file
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save as NumPy array for efficiency
        embeddings_array = {}
        for text_id, embedding in self.embeddings.items():
            embeddings_array[text_id] = np.array(embedding, dtype=np.float32)
        
        np.savez_compressed(output_file, **embeddings_array)
        logger.info(f"Saved {len(self.embeddings)} embeddings to {output_file}")
    
    def load_embeddings(self, input_file: str) -> bool:
        """
        Load embeddings from a file.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            Boolean indicating success
        """
        try:
            with np.load(input_file, allow_pickle=True) as data:
                for key in data.files:
                    self.embeddings[key] = data[key].tolist()
                    
            logger.info(f"Loaded {len(self.embeddings)} embeddings from {input_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return False
