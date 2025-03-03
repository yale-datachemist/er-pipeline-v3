import logging
import numpy as np
import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional, Set, Any, Union
import Levenshtein
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles feature engineering for entity resolution.
    """
    def __init__(self, config: Dict):
        """
        Initialize the feature engineer with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.scaler = StandardScaler()
        self.field_weights = {
            "person": 1.0,
            "roles": 0.7,
            "title": 0.9,
            "attribution": 0.8,
            "provision": 0.6,
            "subjects": 0.7,
            "genres": 0.5,
            "relatedWork": 0.6,
            "record": 0.8
        }
    
    def extract_name_components(self, name: str) -> Dict[str, str]:
        """
        Extract components from a person name.
        
        Args:
            name: Person name string
            
        Returns:
            Dictionary of name components
        """
        components = {
            "full_name": name.strip(),
            "last_name": "",
            "first_name": "",
            "middle_name": "",
            "birth_year": None,
            "death_year": None
        }
        
        # Make a copy of the name for processing
        name_for_processing = name
        
        # Expanded life dates patterns to catch more formats
        life_dates_patterns = [
            r'\((\d{4})-(\d{4})?\)',              # (1856-1915)
            r',\s*(\d{4})-(\d{4})?',              # , 1856-1915
            r',\s*b\.?\s*(\d{4})',                # , b. 1856
            r',\s*d\.?\s*(\d{4})',                # , d. 1915
            r'\(b\.?\s*(\d{4})\)',                # (b. 1856)
            r'\(d\.?\s*(\d{4})\)',                # (d. 1915)
            r',\s*\(?born\s+(\d{4})\)?',          # , born 1856 or (born 1856)
            r',\s*\(?died\s+(\d{4})\)?',          # , died 1915 or (died 1915)
            r',\s*fl\.?\s*(\d{4})(?:-(\d{4}))?'   # , fl. 1856 or , fl. 1856-1870
        ]
        
        # Try each pattern
        for pattern in life_dates_patterns:
            match = re.search(pattern, name_for_processing)
            if match:
                # Different patterns have different group structures
                if 'b' in pattern or 'born' in pattern:
                    components["birth_year"] = match.group(1)
                elif 'd' in pattern or 'died' in pattern:
                    components["death_year"] = match.group(1)
                elif 'fl' in pattern:
                    components["birth_year"] = match.group(1)  # Use floruit as approximate birth
                    if match.lastindex >= 2 and match.group(2):
                        components["death_year"] = match.group(2)
                else:
                    components["birth_year"] = match.group(1)
                    if match.lastindex >= 2 and match.group(2):
                        components["death_year"] = match.group(2)
                
                # Remove the matched date portion for name parsing
                name_for_processing = re.sub(pattern, '', name_for_processing).strip()
        
        # Check for common name format: Last, First Middle
        if ',' in name_for_processing:
            parts = name_for_processing.split(',', 1)
            components["last_name"] = parts[0].strip()
            
            # Process first and middle names
            if len(parts) > 1:
                first_middle = parts[1].strip().split()
                if first_middle:
                    components["first_name"] = first_middle[0]
                    components["middle_name"] = ' '.join(first_middle[1:]) if len(first_middle) > 1 else ""
        else:
            # Assume format: First Middle Last
            parts = name_for_processing.split()
            if len(parts) > 0:
                components["last_name"] = parts[-1]
                components["first_name"] = parts[0] if len(parts) > 1 else ""
                components["middle_name"] = ' '.join(parts[1:-1]) if len(parts) > 2 else ""
        
        return components
    
    def compute_name_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """
        Compute similarity features between two person names.
        
        Args:
            name1: First person name
            name2: Second person name
            
        Returns:
            Dictionary of similarity features
        """
        # Extract name components
        comp1 = self.extract_name_components(name1)
        comp2 = self.extract_name_components(name2)
        
        # Compute similarities
        similarities = {}
        
        # Full name Levenshtein similarity
        similarities["full_name_levenshtein"] = 1.0 - (Levenshtein.distance(comp1["full_name"], comp2["full_name"]) / 
                                               max(len(comp1["full_name"]), len(comp2["full_name"]), 1))
        
        # Last name similarity
        if comp1["last_name"] and comp2["last_name"]:
            similarities["last_name_levenshtein"] = 1.0 - (Levenshtein.distance(comp1["last_name"], comp2["last_name"]) / 
                                                   max(len(comp1["last_name"]), len(comp2["last_name"]), 1))
        else:
            similarities["last_name_levenshtein"] = 0.0
        
        # First name similarity
        if comp1["first_name"] and comp2["first_name"]:
            similarities["first_name_levenshtein"] = 1.0 - (Levenshtein.distance(comp1["first_name"], comp2["first_name"]) / 
                                                    max(len(comp1["first_name"]), len(comp2["first_name"]), 1))
            
            # First initial match
            similarities["first_initial_match"] = 1.0 if comp1["first_name"][0] == comp2["first_name"][0] else 0.0
        else:
            similarities["first_name_levenshtein"] = 0.0
            similarities["first_initial_match"] = 0.0
        
        # Middle name/initial similarity
        if comp1["middle_name"] and comp2["middle_name"]:
            similarities["middle_name_levenshtein"] = 1.0 - (Levenshtein.distance(comp1["middle_name"], comp2["middle_name"]) / 
                                                     max(len(comp1["middle_name"]), len(comp2["middle_name"]), 1))
            
            # Middle initial match
            similarities["middle_initial_match"] = 1.0 if comp1["middle_name"][0] == comp2["middle_name"][0] else 0.0
        else:
            similarities["middle_name_levenshtein"] = 0.0
            similarities["middle_initial_match"] = 0.0
        
        # Birth/death year match
        similarities["birth_year_match"] = 1.0 if (comp1["birth_year"] and comp2["birth_year"] and 
                                             comp1["birth_year"] == comp2["birth_year"]) else 0.0
        similarities["death_year_match"] = 1.0 if (comp1["death_year"] and comp2["death_year"] and 
                                             comp1["death_year"] == comp2["death_year"]) else 0.0
        
        # Life dates present indicator
        similarities["life_dates_present1"] = 1.0 if comp1["birth_year"] or comp1["death_year"] else 0.0
        similarities["life_dates_present2"] = 1.0 if comp2["birth_year"] or comp2["death_year"] else 0.0
        
        return similarities
    
    def extract_publication_years(self, provision: str) -> List[int]:
        """
        Extract publication years from provision information.
        
        Args:
            provision: Provision text
            
        Returns:
            List of years (as integers)
        """
        if not provision or pd.isna(provision):
            return []
            
        # Look for 4-digit years
        years = [int(y) for y in re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', provision)]
        return sorted(years)
    
    def compute_temporal_features(self, provision1: str, provision2: str) -> Dict[str, float]:
        """
        Compute temporal features between two provision strings.
        
        Args:
            provision1: First provision text
            provision2: Second provision text
            
        Returns:
            Dictionary of temporal features
        """
        years1 = self.extract_publication_years(provision1)
        years2 = self.extract_publication_years(provision2)
        
        features = {}
        
        # No years found
        if not years1 or not years2:
            features["years_available"] = 0.0
            features["min_year_diff"] = -1.0
            features["max_year_diff"] = -1.0
            features["mean_year_diff"] = -1.0
            features["temporal_overlap"] = 0.0
            return features
        
        features["years_available"] = 1.0
        
        # Year differences
        all_diffs = []
        for y1 in years1:
            for y2 in years2:
                all_diffs.append(abs(y1 - y2))
        
        features["min_year_diff"] = min(all_diffs) if all_diffs else -1.0
        features["max_year_diff"] = max(all_diffs) if all_diffs else -1.0
        features["mean_year_diff"] = sum(all_diffs) / len(all_diffs) if all_diffs else -1.0
        
        # Temporal overlap
        min_year1, max_year1 = min(years1), max(years1)
        min_year2, max_year2 = min(years2), max(years2)
        
        # Check for overlap
        if max_year1 < min_year2 or max_year2 < min_year1:
            features["temporal_overlap"] = 0.0
        else:
            overlap_start = max(min_year1, min_year2)
            overlap_end = min(max_year1, max_year2)
            overlap_length = overlap_end - overlap_start + 1
            
            range1 = max_year1 - min_year1 + 1
            range2 = max_year2 - min_year2 + 1
            
            # Normalize by the smaller range
            features["temporal_overlap"] = overlap_length / min(range1, range2)
        
        return features
    
    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        if not vec1 or not vec2:
            return 0.0
            
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Compute cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def compute_vector_similarities(self, fields1: Dict[str, List[float]], fields2: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute vector similarities between corresponding fields with support for both list and dict formats.
        
        Args:
            fields1: Dictionary of field vectors for first record
            fields2: Dictionary of field vectors for second record
            
        Returns:
            Dictionary of field similarity features
        """
        similarities = {}
        
        # Compute similarities for each field
        for field, weight in self.field_weights.items():
            vec1 = fields1.get(field, [])
            vec2 = fields2.get(field, [])
            
            # Skip fields with missing vectors
            if not vec1 or not vec2:
                similarities[f"{field}_cosine"] = 0.0
                continue
            
            # Handle different vector formats (dict or list)
            # The diagnostic showed vectors are stored as dicts of length 1
            if isinstance(vec1, dict) and 'default' in vec1:
                vec1 = vec1.get('default', [])
            elif isinstance(vec1, dict) and len(vec1) == 1:
                # If it's a dict with one key (but not 'default'), take the first value
                vec1 = next(iter(vec1.values()), [])
                
            if isinstance(vec2, dict) and 'default' in vec2:
                vec2 = vec2.get('default', [])
            elif isinstance(vec2, dict) and len(vec2) == 1:
                vec2 = next(iter(vec2.values()), [])
            
            # Skip if either vector is empty after conversion
            if not vec1 or not vec2:
                similarities[f"{field}_cosine"] = 0.0
                continue
            
            # Ensure vectors have same dimensionality
            if len(vec1) != len(vec2):
                similarities[f"{field}_cosine"] = 0.0
                continue
            
            # Compute cosine similarity
            try:
                similarity = self.compute_cosine_similarity(vec1, vec2)
                similarities[f"{field}_cosine"] = similarity
            except Exception as e:
                logger.debug(f"Error computing cosine similarity for field {field}: {str(e)}")
                similarities[f"{field}_cosine"] = 0.0
        
        return similarities

    
    def generate_features(self, record1: Dict, record2: Dict, 
                     embeddings1: Dict[str, List[float]], embeddings2: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Generate features for a pair of records with improved error handling and logging.
        
        Args:
            record1: First record dictionary
            record2: Second record dictionary
            embeddings1: Field embeddings for first record
            embeddings2: Field embeddings for second record
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Debug checks to help diagnose issues
        record1_id = record1.get("id", "unknown")
        record2_id = record2.get("id", "unknown")
        
        # Check if both records have person names
        if not record1.get("person") or not record2.get("person"):
            # Skip pairs where person name is missing
            logger.debug(f"Missing person in one of the records: {record1_id} or {record2_id}")
            return features

        try:
            # Name-based features
            name_features = self.compute_name_similarity(
                record1.get("person", ""), 
                record2.get("person", "")
            )
            features.update(name_features)
            
            # Temporal features from provision
            temporal_features = self.compute_temporal_features(
                record1.get("provision", ""), 
                record2.get("provision", "")
            )
            features.update(temporal_features)
            
            # Vector-based similarity features - only if embeddings are available
            if embeddings1 and embeddings2:
                vector_features = self.compute_vector_similarities(embeddings1, embeddings2)
                features.update(vector_features)
            else:
                # Add placeholder features if vectors aren't available
                for field in self.field_weights.keys():
                    features[f"{field}_cosine"] = 0.0
            
            # Text length features
            for field in ["person", "roles", "title", "record"]:
                len1 = len(record1.get(field, "")) if record1.get(field) else 0
                len2 = len(record2.get(field, "")) if record2.get(field) else 0
                
                features[f"{field}_len_ratio"] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
            
            # Field presence indicators
            for field in ["attribution", "provision", "subjects", "genres", "relatedWork"]:
                features[f"{field}_present_both"] = 1.0 if (record1.get(field) and record2.get(field)) else 0.0
                features[f"{field}_present_none"] = 1.0 if (not record1.get(field) and not record2.get(field)) else 0.0
                features[f"{field}_present_one"] = 1.0 if (bool(record1.get(field)) != bool(record2.get(field))) else 0.0
            
            # Check if we have enough features
            if len(features) < 5:  # Arbitrary minimum threshold
                logger.warning(f"Too few features generated for records {record1_id} and {record2_id}: {len(features)}")
        
        except Exception as e:
            logger.warning(f"Error generating features for records {record1_id} and {record2_id}: {str(e)}")
            # Return empty features dict which will be skipped
        
        return features
    
    def vectorize_features(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert a feature dictionary to a feature vector.
        
        Args:
            feature_dict: Dictionary of features
            
        Returns:
            Feature vector as numpy array
        """
        # Define the feature order (for consistency)
        feature_names = sorted(feature_dict.keys())
        
        # Create the feature vector
        feature_vector = np.array([feature_dict[name] for name in feature_names])
        
        return feature_vector, feature_names
    
    def prepare_training_data(self, train_pairs: List[Tuple], record_embeddings: Dict[str, Dict[str, List[float]]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for the classifier with improved embeddings access.
        
        Args:
            train_pairs: List of (record1, record2, is_match) tuples
            record_embeddings: Dictionary mapping record IDs to field embeddings
            
        Returns:
            Tuple of (feature_matrix, labels, feature_names)
        """
        logger.info(f"Preparing training data with {len(train_pairs)} pairs")
        
        if not train_pairs:
            logger.error("No training pairs provided")
            return np.array([]), np.array([]), []
        
        # Generate features for each pair
        feature_dicts = []
        labels = []
        error_count = 0
        missing_embeddings = 0
        missing_ids = set()
        
        for record1, record2, is_match in tqdm(train_pairs, desc="Generating features"):
            record1_id = record1.get("id")
            record2_id = record2.get("id")
            
            if not record1_id or not record2_id:
                error_count += 1
                continue
                
            # Get embeddings for the records - add detailed logging
            embeddings1 = record_embeddings.get(record1_id, {})
            embeddings2 = record_embeddings.get(record2_id, {})
            
            if not embeddings1:
                missing_ids.add(record1_id)
            if not embeddings2:
                missing_ids.add(record2_id)
                
            if not embeddings1 or not embeddings2:
                # Skip pairs without embeddings but count them
                missing_embeddings += 1
                if missing_embeddings <= 5:  # Only log the first few to avoid flood
                    logger.debug(f"Missing embeddings for records: {record1_id if not embeddings1 else ''}, {record2_id if not embeddings2 else ''}")
                continue
                
            # Generate features
            features = self.generate_features(record1, record2, embeddings1, embeddings2)
            
            # Only add if features were actually generated
            if features:
                feature_dicts.append(features)
                labels.append(1 if is_match else 0)
        
        # Log diagnostics
        if missing_embeddings > 0:
            logger.warning(f"Skipped {missing_embeddings} pairs due to missing embeddings")
            # Log some of the missing IDs
            if missing_ids:
                sample_missing = list(missing_ids)[:10]
                logger.warning(f"Sample missing embedding IDs: {sample_missing}")
        
        if error_count > 0:
            logger.warning(f"Skipped {error_count} pairs due to missing record IDs")
        
        if not feature_dicts:
            logger.error("No valid feature dictionaries generated")
            # Provide more diagnostics
            if train_pairs:
                sample_pair = train_pairs[0]
                sample_record1 = sample_pair[0]
                sample_record2 = sample_pair[1]
                sample_id1 = sample_record1.get("id", "unknown")
                sample_id2 = sample_record2.get("id", "unknown")
                
                # Check embeddings
                has_embedding1 = sample_id1 in record_embeddings
                has_embedding2 = sample_id2 in record_embeddings
                
                logger.error(f"Sample pair diagnostic - Record IDs: {sample_id1}, {sample_id2}")
                logger.error(f"Embeddings available: {has_embedding1}, {has_embedding2}")
                
                if has_embedding1:
                    emb1 = record_embeddings.get(sample_id1, {})
                    logger.error(f"Sample embedding keys for {sample_id1}: {list(emb1.keys())}")
                    
                    # Check field value for a specific field
                    if 'person' in emb1:
                        person_vec = emb1['person']
                        logger.error(f"Person vector type: {type(person_vec)}, {'length=1 dict' if isinstance(person_vec, dict) and len(person_vec)==1 else 'not length=1 dict'}")
                        if isinstance(person_vec, dict) and len(person_vec) == 1:
                            first_key = next(iter(person_vec.keys()))
                            logger.error(f"Person vector dict first key: {first_key}")
                    
            return np.array([]), np.array([]), []
        
        # Collect all feature names from all dictionaries
        all_feature_names = set()
        for feat_dict in feature_dicts:
            all_feature_names.update(feat_dict.keys())
        
        feature_names = sorted(all_feature_names)
        
        # Create feature matrix
        X = np.zeros((len(feature_dicts), len(feature_names)))
        
        for i, feature_dict in enumerate(feature_dicts):
            for j, feature_name in enumerate(feature_names):
                X[i, j] = feature_dict.get(feature_name, 0.0)
        
        # Convert labels to numpy array
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared training data with {X_scaled.shape[0]} samples and {X_scaled.shape[1]} features")
        
        # Debug info about class balance
        positive_count = np.sum(y)
        negative_count = len(y) - positive_count
        logger.info(f"Class balance: {positive_count} positives, {negative_count} negatives")
        
        return X_scaled, y, feature_names
    
    def save_scaler(self, file_path: str) -> None:
        """
        Save the fitted scaler to a file.
        
        Args:
            file_path: Path to save the scaler
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved scaler to {file_path}")
    
    def load_scaler(self, file_path: str) -> bool:
        """
        Load a fitted scaler from a file.
        
        Args:
            file_path: Path to the scaler file
            
        Returns:
            Boolean indicating success
        """
        try:
            with open(file_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            return False


class LogisticRegressionClassifier:
    """
    Logistic regression classifier implemented with gradient descent.
    """
    def __init__(self, config: Dict):
        """
        Initialize the classifier with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.learning_rate = config.get("learning_rate", 0.01)
        self.max_iterations = config.get("max_iterations", 1000)
        self.tolerance = config.get("tolerance", 1e-4)
        self.l2_reg = config.get("l2_regularization", 0.01)
        self.grad_clip_threshold = config.get("grad_clip_threshold", 5.0)
        
        self.weights = None
        self.bias = None
        self.feature_names = None
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with improved numerical stability.
        
        Args:
            z: Input array
            
        Returns:
            Sigmoid of input
        """
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model not trained yet")
            
        z = X.dot(self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Array of predicted labels
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict:
        """
        Train the model using gradient descent with improved stability.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            
        Returns:
            Dictionary of training metrics
        """
        n_samples, n_features = X.shape
        
        # Check class balance
        class_balance = np.mean(y)
        logger.info(f"Class balance (proportion of positive examples): {class_balance:.3f}")
        
        # Adjust weight initialization for class imbalance
        self.weights = np.zeros(n_features)
        # Initialize bias based on class distribution
        self.bias = np.log(class_balance / (1 - class_balance)) if 0 < class_balance < 1 else 0.0
        self.feature_names = feature_names
        
        # Tracking for convergence
        prev_loss = float('inf')
        
        # Training history
        history = {
            "loss": [],
            "accuracy": []
        }
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict_proba(X)
            
            # Compute loss
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            loss += (self.l2_reg / (2 * n_samples)) * np.sum(self.weights ** 2)  # L2 regularization
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.l2_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Apply gradient clipping
            dw_norm = np.linalg.norm(dw)
            if dw_norm > self.grad_clip_threshold:
                dw = self.grad_clip_threshold * dw / dw_norm
            
            if abs(db) > self.grad_clip_threshold:
                db = self.grad_clip_threshold * np.sign(db)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate accuracy
            y_pred_labels = self.predict(X)
            accuracy = np.mean(y_pred_labels == y)
            
            # Store metrics
            history["loss"].append(loss)
            history["accuracy"].append(accuracy)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                logger.info(f"Converged at iteration {i}")
                break
                
            prev_loss = loss
            
            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Iteration {i+1}: Loss = {loss:.6f}, Accuracy = {accuracy:.6f}")
        
        # Feature importance
        feature_importance = {}
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                feature_importance[name] = abs(self.weights[i])
        
        return {
            "iterations": i + 1,
            "final_loss": loss,
            "final_accuracy": accuracy,
            "feature_importance": feature_importance,
            "history": history
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Evaluate the model on a test set.
        
        Args:
            X: Feature matrix
            y: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        tn = np.sum((y_pred == 0) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        
        accuracy = (tp + tn) / len(y) if len(y) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        }
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            file_path: Path to save the model
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model_data = {
            "weights": self.weights,
            "bias": self.bias,
            "feature_names": self.feature_names
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Saved model to {file_path}")
    
    def load_model(self, file_path: str) -> bool:
        """
        Load a trained model from a file.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Boolean indicating success
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.weights = model_data["weights"]
            self.bias = model_data["bias"]
            self.feature_names = model_data["feature_names"]
            
            logger.info(f"Loaded model from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False