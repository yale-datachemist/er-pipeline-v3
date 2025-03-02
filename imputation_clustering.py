import logging
import numpy as np
import pandas as pd
import json
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import re
from datetime import datetime
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NullValueImputer:
    """
    Handles imputation of null values using vector-based methods.
    """
    def __init__(self, config: Dict, weaviate_manager):
        """
        Initialize the imputer with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
            weaviate_manager: Weaviate manager instance for vector search
        """
        self.config = config
        self.weaviate_manager = weaviate_manager
        self.imputation_cache = {}  # Cache for imputed values
        self.imputable_fields = ["attribution", "provision", "subjects", "genres", "relatedWork"]
        self.num_neighbors = config.get("imputation_neighbors", 10)
    
    def impute_null_fields(self, record: Dict, embeddings: Dict[str, List[float]]) -> Dict:
        """
        Impute null fields in a record using vector-based imputation, tracking original presence.
        
        Args:
            record: Record dictionary
            embeddings: Field embeddings for the record
            
        Returns:
            Record with imputed values and presence tracking
        """
        imputed_record = record.copy()
        
        # Track which fields were originally present
        original_presence = {}
        for field in self.imputable_fields:
            has_value = bool(record.get(field)) and not pd.isna(record.get(field))
            original_presence[field] = has_value
        
        # Store original presence in the record
        imputed_record["_original_presence"] = original_presence
        
        # Check each field for null values
        for field in self.imputable_fields:
            if not record.get(field) or pd.isna(record.get(field)):
                imputed_value = self.impute_field(record, field, embeddings.get("record", []))
                if imputed_value:
                    imputed_record[field] = imputed_value
        
        return imputed_record
    
    def impute_field(self, record: Dict, field: str, record_vector: List[float]) -> Optional[str]:
        """
        Impute a null field in a record using vector-based hot-deck imputation.
        
        Args:
            record: Record dictionary
            field: Field name to impute
            record_vector: Vector embedding of the record
            
        Returns:
            Imputed value or None if imputation failed
        """
        record_id = record.get("id", "unknown")
        
        # Check cache first
        cache_key = f"{record_id}_{field}"
        if cache_key in self.imputation_cache:
            return self.imputation_cache[cache_key]
            
        if not record_vector:
            logger.warning(f"No record vector available for imputation: {record_id}")
            return None
        
        # Get candidates from vector search
        candidates = self.weaviate_manager.get_imputation_candidates(
            query_vector=record_vector,
            field_type=field,
            limit=self.num_neighbors
        )
        
        if not candidates:
            logger.warning(f"No imputation candidates found for {record_id}, field {field}")
            return None
        
        # Extract text values from candidates
        candidate_texts = [c.get("text", "") for c in candidates]
        
        # Simple method: use most common value
        from collections import Counter
        value_counts = Counter(candidate_texts)
        if not value_counts:
            return None
            
        most_common = value_counts.most_common(1)[0][0]
        
        # Store in cache
        self.imputation_cache[cache_key] = most_common
        
        return most_common


class EntityClusterer:
    """
    Clusters entity records into identity groups.
    """
    def __init__(self, config: Dict, classifier, weaviate_manager, imputer, feature_engineer):
        """
        Initialize the clusterer with configuration settings.
        
        Args:
            config: Dictionary containing configuration settings
            classifier: Trained classifier instance
            weaviate_manager: Weaviate manager instance
            imputer: Null value imputer instance
            feature_engineer: Feature engineer instance
        """
        self.config = config
        self.classifier = classifier
        self.weaviate_manager = weaviate_manager
        self.imputer = imputer
        self.feature_engineer = feature_engineer
        
        self.match_threshold = config.get("match_threshold", 0.7)
        self.min_cluster_size = config.get("min_cluster_size", 2)
        self.max_neighbors = config.get("max_neighbors", 50)
        self.record_embeddings = {}
        
        # LLM client for fallback disambiguation
        self.llm_client = openai.OpenAI(api_key=config.get("openai_api_key"))
        self.llm_model = config.get("llm_model", "gpt-4o")
        self.max_llm_requests = config.get("max_llm_requests", 1000)
        self.llm_requests_made = 0
    
    def get_query_candidates(self, person_vector: List[float], person_name: str, record_id: str) -> List[Dict]:
        """
        Get candidate matches for a query record using ANN search.
        
        Args:
            person_vector: Vector embedding of the person name
            person_name: Person name string
            record_id: Record ID
            
        Returns:
            List of candidate records
        """
        # First, do vector search for candidates
        neighbors = self.weaviate_manager.search_neighbors(
            query_vector=person_vector,
            limit=self.max_neighbors,
            distance_threshold=0.3  # Can be adjusted based on validation
        )
        
        # Filter out the query record itself
        candidates = [n for n in neighbors if n.get("id") != record_id]
        
        # Additional filtering for name components
        filtered_candidates = []
        query_name_parts = self.feature_engineer.extract_name_components(person_name)
        
        for candidate in candidates:
            candidate_name = candidate.get("person", "")
            candidate_parts = self.feature_engineer.extract_name_components(candidate_name)
            
            # Strong signal: exact match with life dates
            if (query_name_parts["birth_year"] and query_name_parts["birth_year"] == candidate_parts["birth_year"]):
                filtered_candidates.append(candidate)
                continue
                
            # Last name match
            if (query_name_parts["last_name"] and candidate_parts["last_name"] and 
                query_name_parts["last_name"].lower() == candidate_parts["last_name"].lower()):
                
                # First initial match
                if (query_name_parts["first_name"] and candidate_parts["first_name"] and
                    query_name_parts["first_name"][0].lower() == candidate_parts["first_name"][0].lower()):
                    filtered_candidates.append(candidate)
                    continue
            
            # Include near matches
            filtered_candidates.append(candidate)
        
        return filtered_candidates[:self.max_neighbors]  # Limit to max neighbors
    
    def classify_record_pair(self, record1: Dict, record2: Dict, 
                             embeddings1: Dict[str, List[float]], embeddings2: Dict[str, List[float]]) -> Tuple[bool, float]:
        """
        Classify whether two records refer to the same entity.
        
        Args:
            record1: First record dictionary
            record2: Second record dictionary
            embeddings1: Field embeddings for first record
            embeddings2: Field embeddings for second record
            
        Returns:
            Tuple of (is_match, confidence)
        """
        # First, check for exact name match with life dates (very strong signal)
        name1 = record1.get("person", "")
        name2 = record2.get("person", "")
        
        name_parts1 = self.feature_engineer.extract_name_components(name1)
        name_parts2 = self.feature_engineer.extract_name_components(name2)
        
        # Strong signal: Exact match with birth/death years
        # This overrides other signals unless there's strong evidence against it
        exact_name_match = self._check_exact_name_match_with_dates(name_parts1, name_parts2)
        if exact_name_match:
            logger.info(f"Strong signal: Exact name match with life dates between '{name1}' and '{name2}'")
            return True, 0.98
        
        # Extract features
        features = self.feature_engineer.generate_features(record1, record2, embeddings1, embeddings2)
        
        # Vectorize features
        feature_vector, _ = self.feature_engineer.vectorize_features(features)
        
        # Scale features (single sample reshape)
        scaled_vector = self.feature_engineer.scaler.transform(feature_vector.reshape(1, -1))
        
        # Get prediction probability
        probability = float(self.classifier.predict_proba(scaled_vector)[0])
        
        # Handle life date signals that weren't exact matches
        if (name_parts1["birth_year"] and name_parts2["birth_year"] and 
            name_parts1["birth_year"] == name_parts2["birth_year"]):
            
            # Boost probability for matching birth years
            probability = max(probability, 0.9)
            
            # If death years also match, even stronger signal
            if (name_parts1["death_year"] and name_parts2["death_year"] and 
                name_parts1["death_year"] == name_parts2["death_year"]):
                probability = max(probability, 0.95)
        
        # Special case: Check for conflicting life dates - strong negative signal
        if self._has_conflicting_life_dates(name_parts1, name_parts2):
            logger.info(f"Negative signal: Conflicting life dates between '{name1}' and '{name2}'")
            probability = min(probability, 0.3)
            
        # Make decision
        is_match = probability >= self.match_threshold
        
        # Check for temporal impossibility
        if is_match:
            temporal_conflict = self._check_temporal_conflict(record1, record2)
            if temporal_conflict:
                logger.info(f"Temporal impossibility detected between records")
                is_match = False
                probability = min(probability, 0.2)
        
        return is_match, probability
    
    def _check_exact_name_match_with_dates(self, name_parts1: Dict, name_parts2: Dict) -> bool:
        """
        Check if two names are exact matches with life dates.
        
        Args:
            name_parts1: Components of first name
            name_parts2: Components of second name
            
        Returns:
            Boolean indicating exact match with dates
        """
        # Both must have at least birth year
        if not (name_parts1["birth_year"] and name_parts2["birth_year"]):
            return False
            
        # Birth years must match
        if name_parts1["birth_year"] != name_parts2["birth_year"]:
            return False
            
        # If both have death years, they must match
        if (name_parts1["death_year"] and name_parts2["death_year"] and 
            name_parts1["death_year"] != name_parts2["death_year"]):
            return False
            
        # Last names must match
        if (not name_parts1["last_name"] or not name_parts2["last_name"] or 
            name_parts1["last_name"].lower() != name_parts2["last_name"].lower()):
            return False
            
        # First names must match if both present
        if (name_parts1["first_name"] and name_parts2["first_name"] and 
            name_parts1["first_name"].lower() != name_parts2["first_name"].lower()):
            return False
            
        # If we got here, we have matching names with matching birth years
        return True
    
    def _has_conflicting_life_dates(self, name_parts1: Dict, name_parts2: Dict) -> bool:
        """
        Check if two name entries have conflicting life dates.
        
        Args:
            name_parts1: Components of first name
            name_parts2: Components of second name
            
        Returns:
            Boolean indicating conflict
        """
        # If both have birth years, they must match
        if (name_parts1["birth_year"] and name_parts2["birth_year"] and
            name_parts1["birth_year"] != name_parts2["birth_year"]):
            
            # Try to convert to integers for comparison
            try:
                birth1 = int(name_parts1["birth_year"])
                birth2 = int(name_parts2["birth_year"])
                
                # Allow small differences (data errors)
                if abs(birth1 - birth2) <= 1:
                    return False
                    
                return True
            except ValueError:
                # Non-numeric years, consider as conflict
                return True
        
        # Same for death years
        if (name_parts1["death_year"] and name_parts2["death_year"] and
            name_parts1["death_year"] != name_parts2["death_year"]):
            
            try:
                death1 = int(name_parts1["death_year"])
                death2 = int(name_parts2["death_year"])
                
                # Allow small differences
                if abs(death1 - death2) <= 1:
                    return False
                    
                return True
            except ValueError:
                return True
                
        return False
    
    def _check_temporal_conflict(self, record1: Dict, record2: Dict) -> bool:
        """
        Check for temporal impossibilities between records.
        
        Args:
            record1: First record dictionary
            record2: Second record dictionary
            
        Returns:
            Boolean indicating temporal conflict
        """
        # Extract birth years
        name_parts1 = self.feature_engineer.extract_name_components(record1.get("person", ""))
        name_parts2 = self.feature_engineer.extract_name_components(record2.get("person", ""))
        
        birth_year1 = name_parts1.get("birth_year")
        birth_year2 = name_parts2.get("birth_year")
        
        # Extract publication years
        pub_years1 = self.feature_engineer.extract_publication_years(record1.get("provision", ""))
        pub_years2 = self.feature_engineer.extract_publication_years(record2.get("provision", ""))
        
        # If no publication years, no conflict
        if not pub_years1 or not pub_years2:
            return False
            
        # If no birth years, no way to check conflict
        if not birth_year1 and not birth_year2:
            return False
            
        birth_year = birth_year1 or birth_year2
        if not birth_year:
            return False
            
        try:
            birth = int(birth_year)
            
            # Check if publication is too early
            # (Allow for publication up to 15 years before birth for data errors)
            earliest_pub1 = min(pub_years1) if pub_years1 else 9999
            earliest_pub2 = min(pub_years2) if pub_years2 else 9999
            
            if (earliest_pub1 < birth - 15) or (earliest_pub2 < birth - 15):
                # Publication too early for author's birth
                return True
                
            # Check posthumous publications if death dates exist
            death_year1 = name_parts1.get("death_year")
            death_year2 = name_parts2.get("death_year")
            
            death_year = death_year1 or death_year2
            if death_year:
                try:
                    death = int(death_year)
                    
                    # Allow posthumous works up to 100 years after death
                    latest_pub1 = max(pub_years1) if pub_years1 else 0
                    latest_pub2 = max(pub_years2) if pub_years2 else 0
                    
                    if (latest_pub1 > death + 100) or (latest_pub2 > death + 100):
                        # Publication too late after death
                        return True
                        
                except ValueError:
                    pass
                    
        except ValueError:
            pass
            
        return False
    
    def llm_disambiguation(self, record1: Dict, record2: Dict) -> Tuple[bool, str]:
        """
        Use LLM for disambiguation in borderline cases.
        
        Args:
            record1: First record dictionary
            record2: Second record dictionary
            
        Returns:
            Tuple of (is_match, explanation)
        """
        # Check if we've hit the limit
        if self.llm_requests_made >= self.max_llm_requests:
            logger.warning("LLM request limit reached, skipping disambiguation")
            return None, "LLM request limit reached"
        
        try:
            # Prepare prompt
            system_prompt = "You are a classifier deciding whether two records refer to the same person."
            
            user_prompt = (
                "Tell me whether the following two records are referring to the same person or a different person "
                "using a chain of reasoning followed by a single yes or no answer on a single line, without any formatting.\n\n"
                f"1. Subject: {record1.get('person', '')}\n"
                f"Title: {record1.get('title', '')}\n"
                f"Subjects: {record1.get('subjects', '')}\n"
                f"Genres: {record1.get('genres', '')}\n"
                f"Provision information: {record1.get('provision', '')}\n\n"
                f"2. Subject: {record2.get('person', '')}\n"
                f"Title: {record2.get('title', '')}\n"
                f"Subjects: {record2.get('subjects', '')}\n"
                f"Genres: {record2.get('genres', '')}\n"
                f"Provision information: {record2.get('provision', '')}"
            )
            
            # Make API call
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            
            # Increment request counter
            self.llm_requests_made += 1
            
            # Extract reasoning and decision
            content = response.choices[0].message.content
            
            # Find the yes/no answer
            match = re.search(r'(?i)\b(yes|no)\b', content.split('\n')[-1])
            is_match = match and match.group(1).lower() == 'yes'
            
            return is_match, content
            
        except Exception as e:
            logger.error(f"Error using LLM for disambiguation: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def build_match_graph(self, records: List[Dict], embeddings_map: Dict[str, Dict[str, List[float]]]) -> nx.Graph:
        """
        Build a graph of entity matches with improved memory efficiency.
        
        Args:
            records: List of record dictionaries
            embeddings_map: Map of record IDs to field embeddings
            
        Returns:
            NetworkX graph of matches
        """
        graph = nx.Graph()
        
        # Process in smaller batches to reduce memory usage
        batch_size = self.config.get("graph_batch_size", 1000)
        
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i+batch_size]
            
            # Add batch records as nodes
            for record in batch_records:
                record_id = record.get("id")
                if record_id:
                    graph.add_node(record_id, **record)
            
            # Process each record in batch for matches
            for record in tqdm(batch_records, desc=f"Building match graph (batch {i//batch_size + 1})"):
                record_id = record.get("id")
                person_name = record.get("person", "")
                
                if not record_id or not person_name or record_id not in embeddings_map:
                    continue
                    
                # Get person vector for ANN search
                person_vector = embeddings_map[record_id].get("person", [])
                if not person_vector:
                    continue
                    
                # Get candidate matches
                candidates = self.get_query_candidates(person_vector, person_name, record_id)
                
                # Classify each candidate pair
                for candidate in candidates:
                    candidate_id = candidate.get("id")
                    
                    if not candidate_id or candidate_id not in embeddings_map:
                        continue
                        
                    # Skip if edge already exists
                    if graph.has_edge(record_id, candidate_id):
                        continue
                        
                    # Get full records from graph
                    candidate_record = graph.nodes[candidate_id]
                    
                    # Handle null values through imputation
                    if any(pd.isna(record.get(field)) for field in self.imputer.imputable_fields):
                        record = self.imputer.impute_null_fields(record, embeddings_map[record_id])
                        
                    if any(pd.isna(candidate_record.get(field)) for field in self.imputer.imputable_fields):
                        candidate_record = self.imputer.impute_null_fields(candidate_record, embeddings_map[candidate_id])
                    
                    # Classify the pair
                    is_match, confidence = self.classify_record_pair(
                        record, candidate_record,
                        embeddings_map[record_id], embeddings_map[candidate_id]
                    )
                    
                    # Handle borderline cases with LLM if configured
                    llm_result = None
                    llm_explanation = None
                    
                    if 0.4 <= confidence <= 0.8 and self.config.get("use_llm_fallback", True):
                        llm_result, llm_explanation = self.llm_disambiguation(record, candidate_record)
                        
                        # Override classifier if LLM provided a result
                        if llm_result is not None:
                            is_match = llm_result
                            # Adjust confidence based on LLM decision
                            confidence = 0.85 if is_match else 0.15
                    
                    # Add edge if it's a match
                    if is_match:
                        graph.add_edge(
                            record_id, 
                            candidate_id, 
                            confidence=confidence,
                            llm_used=(llm_result is not None),
                            llm_explanation=llm_explanation
                        )
        
        return graph
    
    def extract_clusters(self, graph: nx.Graph) -> List[Dict]:
        """
        Extract entity clusters from the match graph with enhanced graph-based refinement.
        
        Args:
            graph: NetworkX graph of matches
            
        Returns:
            List of cluster dictionaries
        """
        # Apply edge weight refinement to prune weak connections
        self._refine_graph_edges(graph)
        
        # Find connected components (initial clusters)
        connected_components = list(nx.connected_components(graph))
        
        # Apply community detection for large components
        refined_components = []
        for component in connected_components:
            if len(component) > 20:  # Only refine larger clusters
                # Extract subgraph for this component
                subgraph = graph.subgraph(component)
                
                # Apply Louvain community detection
                try:
                    import community as community_louvain
                    partition = community_louvain.best_partition(subgraph)
                    
                    # Group nodes by community
                    communities = defaultdict(set)
                    for node, community_id in partition.items():
                        communities[community_id].add(node)
                    
                    # Add the refined communities
                    for community in communities.values():
                        refined_components.append(community)
                except ImportError:
                    # Fallback if community detection package not available
                    logger.warning("Community detection package not available, using connected component")
                    refined_components.append(component)
            else:
                # Keep smaller components as is
                refined_components.append(component)
        
        clusters = []
        
        for i, component in enumerate(tqdm(refined_components, desc="Extracting clusters")):
            # Skip single-node components
            if len(component) < self.min_cluster_size:
                continue
                
            # Get records in this cluster
            cluster_records = []
            for record_id in component:
                record_data = graph.nodes[record_id]
                cluster_records.append(record_data)
            
            # Compute node centrality to identify central nodes
            subgraph = graph.subgraph(component)
            centrality = nx.degree_centrality(subgraph)
            
            # Sort records by centrality
            central_records = sorted([(record_id, centrality.get(record_id, 0)) 
                                     for record_id in component], 
                                    key=lambda x: x[1], reverse=True)
            
            # Use most central node as primary record
            primary_record_id = central_records[0][0] if central_records else None
            
            # Determine canonical name using life dates and frequency
            canonical_name, life_dates = self._get_canonical_name(cluster_records)
            
            # Calculate cluster confidence from edge confidences and structure
            cluster_confidence = self._calculate_cluster_confidence(subgraph)
            
            # Generate explanation for the clustering
            explanation = self._generate_cluster_explanation(subgraph, canonical_name, life_dates)
            
            # Collect disambiguated identity information
            identity = {
                "cluster_id": f"cluster_{i}",
                "canonical_name": canonical_name,
                "life_dates": life_dates,
                "size": len(component),
                "confidence": cluster_confidence,
                "primary_record_id": primary_record_id,
                "centrality_scores": dict(central_records),
                "records": [record.get("id") for record in cluster_records],
                "record_details": cluster_records,
                "explanation": explanation
            }
            
            clusters.append(identity)
        
        return clusters
    
    def _refine_graph_edges(self, graph: nx.Graph) -> None:
        """
        Refine graph edges by pruning weak connections.
        
        Args:
            graph: NetworkX graph to refine
        """
        edges_to_remove = []
        
        # Identify transitivity issues: A-B-C connected but A-C has low similarity
        for node_a in graph.nodes():
            neighbors_a = list(graph.neighbors(node_a))
            
            for i, node_b in enumerate(neighbors_a):
                edge_ab = graph.get_edge_data(node_a, node_b)
                conf_ab = edge_ab.get('confidence', 0)
                
                # Skip low-confidence edges
                if conf_ab < 0.7:
                    continue
                    
                for node_c in neighbors_a[i+1:]:
                    edge_bc = graph.get_edge_data(node_b, node_c)
                    
                    # If B-C exists, check if A-C should exist
                    if edge_bc:
                        conf_bc = edge_bc.get('confidence', 0)
                        
                        # If both A-B and B-C are strong but A-C doesn't exist or is weak
                        if conf_bc >= 0.7:
                            if not graph.has_edge(node_a, node_c):
                                # Infer low confidence A-C edge
                                pass
                            else:
                                edge_ac = graph.get_edge_data(node_a, node_c)
                                conf_ac = edge_ac.get('confidence', 0)
                                
                                # If A-C has much lower confidence, might be a false positive
                                if conf_ac < 0.4 and (conf_ab + conf_bc) / 2 > 0.8:
                                    edges_to_remove.append((node_a, node_c))
        
        # Remove problematic edges
        for u, v in edges_to_remove:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
                logger.info(f"Removed inconsistent edge: {u} - {v}")
    
    def _calculate_cluster_confidence(self, graph: nx.Graph) -> float:
        """
        Calculate cluster confidence based on graph structure and edge weights.
        
        Args:
            graph: NetworkX graph of the cluster
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        if len(graph) <= 1:
            return 1.0  # Single node is perfectly confident
            
        # Edge density (how complete is the graph)
        n_nodes = len(graph)
        max_edges = n_nodes * (n_nodes - 1) / 2
        actual_edges = len(graph.edges())
        density = actual_edges / max_edges if max_edges > 0 else 0
        
        # Average edge confidence
        edge_confidences = [data["confidence"] for _, _, data in graph.edges(data=True)]
        avg_confidence = sum(edge_confidences) / len(edge_confidences) if edge_confidences else 0.0
        
        # Calculate clustering coefficient (local transitivity)
        try:
            clustering_coef = nx.average_clustering(graph)
        except:
            clustering_coef = 0.0
        
        # Weighted confidence score
        confidence = 0.4 * avg_confidence + 0.3 * density + 0.3 * clustering_coef
        
        return confidence
    
    def _generate_cluster_explanation(self, graph: nx.Graph, canonical_name: str, life_dates: Dict) -> str:
        """
        Generate an explanation for the clustering decision.
        
        Args:
            graph: NetworkX graph of the cluster
            canonical_name: Canonical name of the entity
            life_dates: Dictionary of life dates
            
        Returns:
            Explanation string
        """
        n_nodes = len(graph)
        n_edges = len(graph.edges())
        
        # Get edge confidence stats
        edge_confidences = [data["confidence"] for _, _, data in graph.edges(data=True)]
        avg_confidence = sum(edge_confidences) / len(edge_confidences) if edge_confidences else 0.0
        min_confidence = min(edge_confidences) if edge_confidences else 0.0
        max_confidence = max(edge_confidences) if edge_confidences else 0.0
        
        # Count how many edges used LLM
        llm_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get("llm_used", False))
        
        # Build explanation
        explanation = f"Cluster represents entity '{canonical_name}'"
        
        if life_dates:
            birth = life_dates.get("birth_year", "unknown")
            death = life_dates.get("death_year", "unknown")
            explanation += f" ({birth}-{death})"
        
        explanation += f" with {n_nodes} records connected by {n_edges} matches."
        explanation += f" Average match confidence: {avg_confidence:.2f} (range: {min_confidence:.2f}-{max_confidence:.2f})."
        
        if llm_edges > 0:
            explanation += f" {llm_edges} matches were resolved using LLM disambiguation."
        
        # Add record type information if available
        roles = Counter()
        for _, data in graph.nodes(data=True):
            role = data.get("roles", "")
            if role:
                roles[role] += 1
        
        if roles:
            top_roles = roles.most_common(3)
            role_str = ", ".join(f"{role} ({count})" for role, count in top_roles)
            explanation += f" Most common roles: {role_str}."
        
        return explanation
    
    def _get_canonical_name(self, records: List[Dict]) -> Tuple[str, Dict]:
        """
        Determine the canonical name for a cluster with improved logic.
        
        Args:
            records: List of record dictionaries in the cluster
            
        Returns:
            Tuple of (canonical_name, life_dates)
        """
        name_parts_list = []
        
        # Extract name components from each record
        for record in records:
            person_name = record.get("person", "")
            if person_name:
                name_parts = self.feature_engineer.extract_name_components(person_name)
                name_parts_list.append(name_parts)
        
        if not name_parts_list:
            return "Unknown Person", {}
        
        # Prioritize names with more complete information
        # Score each name by completeness
        name_scores = []
        for i, parts in enumerate(name_parts_list):
            score = 0
            if parts["last_name"]: score += 10
            if parts["first_name"]: score += 5
            if parts["middle_name"]: score += 3
            if parts["birth_year"]: score += 8
            if parts["death_year"]: score += 7
            
            # Prefer names that match the most frequent components
            last_names = Counter(p["last_name"] for p in name_parts_list if p["last_name"])
            most_common_last = last_names.most_common(1)[0][0] if last_names else ""
            if parts["last_name"] == most_common_last: score += 5
            
            name_scores.append((i, score))
        
        # Use the highest scoring name as the base for canonical form
        best_name_index = max(name_scores, key=lambda x: x[1])[0]
        best_name_parts = name_parts_list[best_name_index]
        
        # Use the most complete name components
        last_name = best_name_parts["last_name"]
        first_name = best_name_parts["first_name"]
        
        # Check for birth/death years
        birth_years = [parts["birth_year"] for parts in name_parts_list if parts["birth_year"]]
        death_years = [parts["death_year"] for parts in name_parts_list if parts["death_year"]]
        
        # Use most frequent birth/death years, prioritizing consistency
        birth_year_counts = Counter(birth_years)
        death_year_counts = Counter(death_years)
        
        birth_year = birth_year_counts.most_common(1)[0][0] if birth_years else None
        death_year = death_year_counts.most_common(1)[0][0] if death_years else None
        
        # Check for logical consistency in life dates
        if birth_year and death_year:
            try:
                birth_int = int(birth_year)
                death_int = int(death_year)
                if death_int < birth_int:
                    # Inconsistent dates - keep the more frequent one
                    birth_count = birth_year_counts[birth_year]
                    death_count = death_year_counts[death_year]
                    if birth_count > death_count:
                        death_year = None
                    else:
                        birth_year = None
            except ValueError:
                # Not numeric years, keep both
                pass
        
        # Construct canonical name
        if first_name and last_name:
            canonical_name = f"{last_name}, {first_name}"
        else:
            canonical_name = last_name or first_name or "Unknown Person"
        
        # Add life dates if available
        life_dates = {}
        if birth_year:
            life_dates["birth_year"] = birth_year
        if death_year:
            life_dates["death_year"] = death_year
            
        return canonical_name, life_dates
    
    def save_clusters(self, clusters: List[Dict], output_file: str) -> None:
        """
        Save entity clusters to a file with enhanced information.
        
        Args:
            clusters: List of cluster dictionaries
            output_file: Path to the output file
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert to JSON-Lines format with enhanced information
        with open(output_file, 'w') as f:
            for cluster in clusters:
                # Enhance the cluster with analysis metadata
                output_cluster = self._prepare_cluster_for_output(cluster)
                f.write(json.dumps(output_cluster) + '\n')
        
        logger.info(f"Saved {len(clusters)} clusters to {output_file}")
        
        # Save a summary file
        summary_file = os.path.splitext(output_file)[0] + "_summary.json"
        self._save_cluster_summary(clusters, summary_file)
    
    def _prepare_cluster_for_output(self, cluster: Dict) -> Dict:
        """
        Prepare a cluster for output with enhanced information.
        
        Args:
            cluster: Cluster dictionary
            
        Returns:
            Enhanced cluster dictionary for output
        """
        # Create a copy to avoid modifying the original
        output = cluster.copy()
        
        # Add timestamp
        output["generated_at"] = datetime.now().isoformat()
        
        # Simplify record details for output
        simplified_records = []
        for record in cluster.get("record_details", []):
            simplified_records.append({
                "id": record.get("id"),
                "person": record.get("person"),
                "title": record.get("title"),
                "provision": record.get("provision"),
                "roles": record.get("roles"),
                "subjects": record.get("subjects", ""),
                "genres": record.get("genres", "")
            })
        
        # Replace full details with simplified version
        output["record_details"] = simplified_records
        
        # Add provenance information
        output["provenance"] = {
            "method": "entity_resolution_pipeline",
            "version": "1.0",
            "clustering_threshold": self.match_threshold,
            "llm_assisted": any(record.get("llm_used", False) for record in cluster.get("record_details", [])),
        }
        
        # Add confidence details
        if "confidence" in cluster:
            confidence_category = "high" if cluster["confidence"] > 0.8 else \
                                 "medium" if cluster["confidence"] > 0.6 else "low"
            output["confidence_category"] = confidence_category
        
        # Add canonical record info
        if "primary_record_id" in cluster:
            primary_id = cluster["primary_record_id"]
            primary_record = next((r for r in cluster.get("record_details", []) 
                                  if r.get("id") == primary_id), None)
            if primary_record:
                output["canonical_record"] = {
                    "id": primary_id,
                    "person": primary_record.get("person"),
                    "title": primary_record.get("title"),
                    "provision": primary_record.get("provision")
                }
        
        return output
    
    def _save_cluster_summary(self, clusters: List[Dict], summary_file: str) -> None:
        """
        Save a summary of all clusters.
        
        Args:
            clusters: List of cluster dictionaries
            summary_file: Path to the summary file
        """
        # Compute cluster statistics
        cluster_sizes = [cluster["size"] for cluster in clusters]
        confidence_values = [cluster["confidence"] for cluster in clusters]
        
        # Count clusters by size range
        size_ranges = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, "21-50": 0, "51+": 0}
        for size in cluster_sizes:
            if size == 1:
                size_ranges["1"] += 1
            elif 2 <= size <= 5:
                size_ranges["2-5"] += 1
            elif 6 <= size <= 10:
                size_ranges["6-10"] += 1
            elif 11 <= size <= 20:
                size_ranges["11-20"] += 1
            elif 21 <= size <= 50:
                size_ranges["21-50"] += 1
            else:
                size_ranges["51+"] += 1
        
        # Count clusters by confidence range
        confidence_ranges = {"0.9-1.0": 0, "0.8-0.9": 0, "0.7-0.8": 0, "0.6-0.7": 0, "0.0-0.6": 0}
        for conf in confidence_values:
            if 0.9 <= conf <= 1.0:
                confidence_ranges["0.9-1.0"] += 1
            elif 0.8 <= conf < 0.9:
                confidence_ranges["0.8-0.9"] += 1
            elif 0.7 <= conf < 0.8:
                confidence_ranges["0.7-0.8"] += 1
            elif 0.6 <= conf < 0.7:
                confidence_ranges["0.6-0.7"] += 1
            else:
                confidence_ranges["0.0-0.6"] += 1
        
        # Top 10 largest clusters
        largest_clusters = sorted(clusters, key=lambda c: c["size"], reverse=True)[:10]
        largest_info = [{"id": c["cluster_id"], "name": c["canonical_name"], "size": c["size"]} 
                       for c in largest_clusters]
        
        # Prepare summary
        summary = {
            "total_clusters": len(clusters),
            "total_records_clustered": sum(cluster_sizes),
            "cluster_size_distribution": size_ranges,
            "cluster_confidence_distribution": confidence_ranges,
            "largest_clusters": largest_info,
            "average_cluster_size": sum(cluster_sizes) / len(clusters) if clusters else 0,
            "average_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved cluster summary to {summary_file}")
