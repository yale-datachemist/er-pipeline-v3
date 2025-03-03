"""
Comprehensive reporting module for the classification stage of entity resolution.
Provides detailed analysis of feature vectors, model performance, and visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
import json

logger = logging.getLogger(__name__)

class ClassificationReporter:
    """
    Comprehensive reporting system for the classification stage of entity resolution.
    Provides detailed analysis of feature vectors, model performance, and visualization.
    """
    
    def __init__(self, config: Dict, output_dir: str = "output/reports"):
        """
        Initialize the reporter with configuration settings.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for saving reports
        """
        self.config = config
        self.output_dir = output_dir
        self.report_data = {
            "feature_vectors": [],
            "predictions": [],
            "true_labels": [],
            "record_pairs": [],
            "feature_names": [],
            "model_info": {}
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def clear_data(self):
        """Reset the report data."""
        self.report_data = {
            "feature_vectors": [],
            "predictions": [],
            "true_labels": [],
            "record_pairs": [],
            "feature_names": [],
            "model_info": {}
        }
    
    def add_feature_vectors(self, feature_vectors: np.ndarray, 
                          feature_names: List[str],
                          record_pairs: List[Tuple[Dict, Dict]],
                          labels: np.ndarray,
                          predictions: Optional[np.ndarray] = None,
                          prediction_probs: Optional[np.ndarray] = None):
        """
        Add feature vectors and related data for analysis.
        
        Args:
            feature_vectors: Feature vectors for pairs (n_samples, n_features)
            feature_names: Names of the features
            record_pairs: List of record pairs corresponding to feature vectors
            labels: True labels for the pairs
            predictions: Optional predicted labels
            prediction_probs: Optional prediction probabilities
        """
        self.report_data["feature_vectors"] = feature_vectors
        self.report_data["feature_names"] = feature_names
        self.report_data["record_pairs"] = record_pairs
        self.report_data["true_labels"] = labels
        
        if predictions is not None:
            self.report_data["predictions"] = predictions
        
        if prediction_probs is not None:
            self.report_data["prediction_probs"] = prediction_probs
    
    def add_model_info(self, model_info: Dict):
        """
        Add model information to the report.
        
        Args:
            model_info: Dictionary containing model information
        """
        self.report_data["model_info"] = model_info
    
    def generate_feature_vector_report(self, save_csv: bool = True) -> pd.DataFrame:
        """
        Generate a report of all feature vectors.
        
        Args:
            save_csv: Whether to save as CSV
            
        Returns:
            DataFrame containing feature vectors and metadata
        """
        if not self.report_data["feature_vectors"] or not self.report_data["feature_names"]:
            logger.warning("No feature vectors or names available for reporting")
            return pd.DataFrame()
        
        # Create DataFrame with feature vectors
        feature_df = pd.DataFrame(
            self.report_data["feature_vectors"],
            columns=self.report_data["feature_names"]
        )
        
        # Add metadata
        feature_df["true_label"] = self.report_data["true_labels"]
        
        if len(self.report_data["predictions"]) > 0:
            feature_df["predicted_label"] = self.report_data["predictions"]
            
        if "prediction_probs" in self.report_data and len(self.report_data["prediction_probs"]) > 0:
            feature_df["confidence"] = self.report_data["prediction_probs"]
        
        # Add record pair information
        if self.report_data["record_pairs"]:
            feature_df["left_id"] = [pair[0].get("id", "unknown") for pair in self.report_data["record_pairs"]]
            feature_df["right_id"] = [pair[1].get("id", "unknown") for pair in self.report_data["record_pairs"]]
            feature_df["left_person"] = [pair[0].get("person", "unknown") for pair in self.report_data["record_pairs"]]
            feature_df["right_person"] = [pair[1].get("person", "unknown") for pair in self.report_data["record_pairs"]]
        
        # Save to CSV if requested
        if save_csv:
            output_path = os.path.join(self.output_dir, "feature_vectors.csv")
            feature_df.to_csv(output_path, index=False)
            logger.info(f"Saved feature vectors to {output_path}")
        
        return feature_df
    
    def generate_feature_statistics(self, save_csv: bool = True) -> pd.DataFrame:
        """
        Generate statistics for each feature.
        
        Args:
            save_csv: Whether to save as CSV
            
        Returns:
            DataFrame with feature statistics
        """
        if not self.report_data["feature_vectors"] or not self.report_data["feature_names"]:
            logger.warning("No feature vectors or names available for statistics")
            return pd.DataFrame()
        
        # Create DataFrame with feature vectors
        feature_df = pd.DataFrame(
            self.report_data["feature_vectors"],
            columns=self.report_data["feature_names"]
        )
        
        # Add true labels
        feature_df["true_label"] = self.report_data["true_labels"]
        
        # Calculate statistics
        stats = []
        for feature in self.report_data["feature_names"]:
            # Overall statistics
            overall_stats = feature_df[feature].describe()
            
            # Statistics by class
            positive_stats = feature_df[feature_df["true_label"] == 1][feature].describe()
            negative_stats = feature_df[feature_df["true_label"] == 0][feature].describe()
            
            # Feature importance if available
            importance = "N/A"
            if "model_info" in self.report_data and "feature_importance" in self.report_data["model_info"]:
                importance = self.report_data["model_info"]["feature_importance"].get(feature, "N/A")
            
            stats.append({
                "feature": feature,
                "mean": overall_stats["mean"],
                "std": overall_stats["std"],
                "min": overall_stats["min"],
                "max": overall_stats["max"],
                "positive_mean": positive_stats["mean"],
                "negative_mean": negative_stats["mean"],
                "class_difference": positive_stats["mean"] - negative_stats["mean"],
                "importance": importance
            })
        
        stats_df = pd.DataFrame(stats)
        
        # Sort by importance if available, otherwise by class difference
        if "importance" in stats_df and stats_df["importance"].dtype != object:
            stats_df = stats_df.sort_values("importance", ascending=False)
        else:
            stats_df = stats_df.sort_values("class_difference", ascending=False)
        
        # Save to CSV if requested
        if save_csv:
            output_path = os.path.join(self.output_dir, "feature_statistics.csv")
            stats_df.to_csv(output_path, index=False)
            logger.info(f"Saved feature statistics to {output_path}")
        
        return stats_df
    
    def generate_performance_metrics(self, thresholds: List[float] = None) -> Dict:
        """
        Generate detailed performance metrics at different thresholds.
        
        Args:
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.report_data["true_labels"]:
            logger.warning("No labels available for performance metrics")
            return {}
            
        if "prediction_probs" not in self.report_data or not self.report_data["prediction_probs"]:
            logger.warning("No prediction probabilities available for threshold analysis")
            if not self.report_data["predictions"]:
                return {}
            # Use predictions at default threshold only
            metrics = self._calculate_metrics_at_threshold(
                self.report_data["true_labels"],
                self.report_data["predictions"]
            )
            return {"default": metrics}
        
        # If no thresholds provided, use default set
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Calculate metrics at each threshold
        metrics_by_threshold = {}
        for threshold in thresholds:
            predictions = (self.report_data["prediction_probs"] >= threshold).astype(int)
            metrics = self._calculate_metrics_at_threshold(
                self.report_data["true_labels"],
                predictions
            )
            metrics_by_threshold[threshold] = metrics
        
        # Save metrics to CSV
        metrics_list = []
        for threshold, metrics in metrics_by_threshold.items():
            metrics_dict = metrics.copy()
            metrics_dict["threshold"] = threshold
            metrics_list.append(metrics_dict)
        
        metrics_df = pd.DataFrame(metrics_list)
        output_path = os.path.join(self.output_dir, "performance_metrics.csv")
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"Saved performance metrics to {output_path}")
        
        return metrics_by_threshold
    
    def _calculate_metrics_at_threshold(self, true_labels: np.ndarray, predictions: np.ndarray) -> Dict:
        """
        Calculate classification metrics at a specific threshold.
        
        Args:
            true_labels: True class labels
            predictions: Predicted class labels
            
        Returns:
            Dictionary of metrics
        """
        # Calculate base metrics
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        
        # Handle division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        }
    
    def generate_error_analysis(self, save_csv: bool = True) -> pd.DataFrame:
        """
        Generate analysis of misclassified pairs.
        
        Args:
            save_csv: Whether to save as CSV
            
        Returns:
            DataFrame containing error analysis
        """
        if not self.report_data["true_labels"] or not self.report_data["predictions"]:
            logger.warning("Missing true labels or predictions for error analysis")
            return pd.DataFrame()
        
        # Create DataFrame with feature vectors
        feature_df = pd.DataFrame(
            self.report_data["feature_vectors"],
            columns=self.report_data["feature_names"]
        )
        
        # Add metadata
        feature_df["true_label"] = self.report_data["true_labels"]
        feature_df["predicted_label"] = self.report_data["predictions"]
        
        if "prediction_probs" in self.report_data and len(self.report_data["prediction_probs"]) > 0:
            feature_df["confidence"] = self.report_data["prediction_probs"]
        
        # Add record pair information
        if self.report_data["record_pairs"]:
            feature_df["left_id"] = [pair[0].get("id", "unknown") for pair in self.report_data["record_pairs"]]
            feature_df["right_id"] = [pair[1].get("id", "unknown") for pair in self.report_data["record_pairs"]]
            feature_df["left_person"] = [pair[0].get("person", "unknown") for pair in self.report_data["record_pairs"]]
            feature_df["right_person"] = [pair[1].get("person", "unknown") for pair in self.report_data["record_pairs"]]
        
        # Filter misclassified pairs
        misclassified = feature_df[feature_df["true_label"] != feature_df["predicted_label"]]
        
        # Add error type
        misclassified["error_type"] = misclassified.apply(
            lambda x: "False Positive" if x["true_label"] == 0 else "False Negative", 
            axis=1
        )
        
        # Save to CSV if requested
        if save_csv and not misclassified.empty:
            output_path = os.path.join(self.output_dir, "error_analysis.csv")
            misclassified.to_csv(output_path, index=False)
            logger.info(f"Saved error analysis to {output_path}")
        
        return misclassified
    
    def plot_feature_distributions(self, top_n: int = 10) -> None:
        """
        Plot distributions of top features by importance.
        
        Args:
            top_n: Number of top features to plot
        """
        if not self.report_data["feature_vectors"] or not self.report_data["feature_names"]:
            logger.warning("No feature vectors or names available for plotting")
            return
        
        # Create DataFrame with feature vectors
        feature_df = pd.DataFrame(
            self.report_data["feature_vectors"],
            columns=self.report_data["feature_names"]
        )
        
        # Add true labels
        feature_df["true_label"] = self.report_data["true_labels"]
        
        # Get top features by importance or class difference
        stats_df = self.generate_feature_statistics(save_csv=False)
        if "importance" in stats_df and stats_df["importance"].dtype != object:
            top_features = stats_df.nlargest(top_n, "importance")["feature"].tolist()
        else:
            top_features = stats_df.nlargest(top_n, "class_difference")["feature"].tolist()
        
        # Plot distributions for top features
        plt.figure(figsize=(15, top_n * 3))
        
        for i, feature in enumerate(top_features):
            plt.subplot(top_n, 1, i + 1)
            sns.histplot(
                data=feature_df, x=feature, hue="true_label",
                element="step", stat="density", common_norm=False
            )
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend(["Non-match", "Match"])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "feature_distributions.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved feature distributions plot to {output_path}")
    
    def plot_confusion_matrix(self, normalize: bool = True) -> None:
        """
        Plot confusion matrix for the classification results.
        
        Args:
            normalize: Whether to normalize by row
        """
        if not self.report_data["true_labels"] or not self.report_data["predictions"]:
            logger.warning("Missing true labels or predictions for confusion matrix")
            return
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.report_data["true_labels"], self.report_data["predictions"])
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {output_path}")
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
        """
        if "model_info" not in self.report_data or "feature_importance" not in self.report_data["model_info"]:
            logger.warning("No feature importance available for plotting")
            return
        
        # Get feature importance
        importance = self.report_data["model_info"]["feature_importance"]
        
        if not importance:
            logger.warning("Empty feature importance dictionary")
            return
        
        # Convert to DataFrame
        imp_df = pd.DataFrame(
            {"feature": list(importance.keys()), "importance": list(importance.values())}
        )
        
        # Sort and take top N
        imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=imp_df, y="feature", x="importance", palette="viridis")
        plt.title(f"Top {top_n} Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {output_path}")
    
    def plot_roc_curve(self) -> None:
        """
        Plot ROC curve.
        """
        if not self.report_data["true_labels"] or "prediction_probs" not in self.report_data:
            logger.warning("Missing true labels or prediction probabilities for ROC curve")
            return
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(self.report_data["true_labels"], self.report_data["prediction_probs"])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved ROC curve plot to {output_path}")
    
    def plot_precision_recall_curve(self) -> None:
        """
        Plot precision-recall curve.
        """
        if not self.report_data["true_labels"] or "prediction_probs" not in self.report_data:
            logger.warning("Missing true labels or prediction probabilities for PR curve")
            return
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(
            self.report_data["true_labels"],
            self.report_data["prediction_probs"]
        )
        pr_auc = auc(recall, precision)
        
        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC = {pr_auc:.3f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "precision_recall_curve.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved precision-recall curve plot to {output_path}")
    
    def plot_feature_correlation(self, top_n: int = 20) -> None:
        """
        Plot correlation between top features.
        
        Args:
            top_n: Number of top features to include
        """
        if not self.report_data["feature_vectors"] or not self.report_data["feature_names"]:
            logger.warning("No feature vectors or names available for correlation plot")
            return
        
        # Create DataFrame with feature vectors
        feature_df = pd.DataFrame(
            self.report_data["feature_vectors"],
            columns=self.report_data["feature_names"]
        )
        
        # Get top features by importance or class difference
        stats_df = self.generate_feature_statistics(save_csv=False)
        if "importance" in stats_df and stats_df["importance"].dtype != object:
            top_features = stats_df.nlargest(top_n, "importance")["feature"].tolist()
        else:
            top_features = stats_df.nlargest(top_n, "class_difference")["feature"].tolist()
        
        # Calculate correlation matrix for top features
        corr_matrix = feature_df[top_features].corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                   annot=True, fmt=".2f", square=True)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "feature_correlation.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved feature correlation plot to {output_path}")
    
    def plot_vector_space_visualization(self, method: str = 'pca', n_components: int = 2) -> None:
        """
        Plot visualization of feature vectors in reduced dimensionality.
        
        Args:
            method: Dimensionality reduction method ('pca' or 'tsne')
            n_components: Number of components for reduction
        """
        if not self.report_data["feature_vectors"] or not self.report_data["true_labels"]:
            logger.warning("No feature vectors or labels available for visualization")
            return
        
        # Apply dimensionality reduction
        feature_vectors = self.report_data["feature_vectors"]
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            logger.warning(f"Unknown method: {method}, using PCA instead")
            reducer = PCA(n_components=n_components)
        
        # Reduce dimensionality
        reduced_vectors = reducer.fit_transform(feature_vectors)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if n_components == 2:
            scatter = plt.scatter(
                reduced_vectors[:, 0], reduced_vectors[:, 1],
                c=self.report_data["true_labels"],
                cmap='viridis', alpha=0.5, s=30
            )
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
        elif n_components == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2],
                c=self.report_data["true_labels"],
                cmap='viridis', alpha=0.5, s=30
            )
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        
        plt.colorbar(scatter, label='Class')
        plt.title(f'Feature Vector Visualization using {method.upper()}')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"{method.lower()}_visualization.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved vector space visualization plot to {output_path}")
    
    def generate_comprehensive_report(self, output_format: str = 'html') -> None:
        """
        Generate a comprehensive report including all analyses and visualizations.
        
        Args:
            output_format: Output format ('html' or 'markdown')
        """
        # First, generate all analyses
        feature_stats = self.generate_feature_statistics()
        feature_vectors = self.generate_feature_vector_report()
        error_analysis = self.generate_error_analysis()
        performance_metrics = self.generate_performance_metrics()
        
        # Generate all plots
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_feature_importance()
        self.plot_feature_distributions()
        self.plot_feature_correlation()
        self.plot_vector_space_visualization('pca')
        self.plot_vector_space_visualization('tsne')
        
        # Prepare summary data
        summary = {
            "dataset_size": len(self.report_data["true_labels"]) if self.report_data["true_labels"] else 0,
            "feature_count": len(self.report_data["feature_names"]) if self.report_data["feature_names"] else 0,
            "positive_count": sum(self.report_data["true_labels"]) if self.report_data["true_labels"] else 0,
            "top_features": feature_stats.head(10).to_dict(orient='records') if not feature_stats.empty else [],
            "performance": performance_metrics.get(0.5, {}) if isinstance(performance_metrics, dict) else {},
            "error_count": len(error_analysis) if not error_analysis.empty else 0,
            "report_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary as JSON
        with open(os.path.join(self.output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # If HTML format requested, generate an HTML report
        if output_format.lower() == 'html':
            self._generate_html_report(summary)
        else:
            self._generate_markdown_report(summary)
            
    def _generate_html_report(self, summary: Dict) -> None:
        """
        Generate HTML report.
        
        Args:
            summary: Summary dictionary
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Entity Resolution Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .stats-table th {{ background-color: #4CAF50; color: white; }}
                .image-gallery {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
                .image-container {{ margin: 10px; max-width: 45%; }}
                .image-container img {{ max-width: 100%; }}
                .summary-box {{ background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Entity Resolution Classification Report</h1>
            <p>Generated on: {summary.get('report_date', '')}</p>
            
            <div class="section summary-box">
                <h2>Summary</h2>
                <p>Dataset size: {summary.get('dataset_size', 0)} record pairs</p>
                <p>Number of features: {summary.get('feature_count', 0)}</p>
                <p>Class distribution: {summary.get('positive_count', 0)} matches, {summary.get('dataset_size', 0) - summary.get('positive_count', 0)} non-matches</p>
                <p>Error count: {summary.get('error_count', 0)}</p>
                
                <h3>Performance Metrics</h3>
                <p>Accuracy: {summary.get('performance', {}).get('accuracy', 0):.4f}</p>
                <p>Precision: {summary.get('performance', {}).get('precision', 0):.4f}</p>
                <p>Recall: {summary.get('performance', {}).get('recall', 0):.4f}</p>
                <p>F1 Score: {summary.get('performance', {}).get('f1_score', 0):.4f}</p>
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                <div class="image-container">
                    <img src="feature_importance.png" alt="Feature Importance">
                </div>
            </div>
            
            <div class="section">
                <h2>Classification Performance</h2>
                <div class="image-gallery">
                    <div class="image-container">
                        <h3>Confusion Matrix</h3>
                        <img src="confusion_matrix.png" alt="Confusion Matrix">
                    </div>
                    <div class="image-container">
                        <h3>ROC Curve</h3>
                        <img src="roc_curve.png" alt="ROC Curve">
                    </div>
                    <div class="image-container">
                        <h3>Precision-Recall Curve</h3>
                        <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Feature Distributions and Correlations</h2>
                <div class="image-gallery">
                    <div class="image-container">
                        <h3>Feature Distributions</h3>
                        <img src="feature_distributions.png" alt="Feature Distributions">
                    </div>
                    <div class="image-container">
                        <h3>Feature Correlations</h3>
                        <img src="feature_correlation.png" alt="Feature Correlations">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Vector Space Visualization</h2>
                <div class="image-gallery">
                    <div class="image-container">
                        <h3>PCA Visualization</h3>
                        <img src="pca_visualization.png" alt="PCA Visualization">
                    </div>
                    <div class="image-container">
                        <h3>t-SNE Visualization</h3>
                        <img src="tsne_visualization.png" alt="t-SNE Visualization">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Top Features</h2>
                <table class="stats-table">
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                        <th>Class Difference</th>
                        <th>Positive Mean</th>
                        <th>Negative Mean</th>
                    </tr>
        """
        
        # Add rows for top features
        for feature in summary.get('top_features', []):
            html_content += f"""
                    <tr>
                        <td>{feature.get('feature', '')}</td>
                        <td>{feature.get('importance', 'N/A')}</td>
                        <td>{feature.get('class_difference', 0):.4f}</td>
                        <td>{feature.get('positive_mean', 0):.4f}</td>
                        <td>{feature.get('negative_mean', 0):.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <p>For more detailed analysis, please refer to the CSV files in the report directory.</p>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(self.output_dir, "classification_report.html"), 'w') as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML report to {os.path.join(self.output_dir, 'classification_report.html')}")
    
    def _generate_markdown_report(self, summary: Dict) -> None:
        """
        Generate Markdown report.
        
        Args:
            summary: Summary dictionary
        """
        markdown_content = f"""
# Entity Resolution Classification Report

Generated on: {summary.get('report_date', '')}

## Summary

- Dataset size: {summary.get('dataset_size', 0)} record pairs
- Number of features: {summary.get('feature_count', 0)}
- Class distribution: {summary.get('positive_count', 0)} matches, {summary.get('dataset_size', 0) - summary.get('positive_count', 0)} non-matches
- Error count: {summary.get('error_count', 0)}

### Performance Metrics

- Accuracy: {summary.get('performance', {}).get('accuracy', 0):.4f}
- Precision: {summary.get('performance', {}).get('precision', 0):.4f}
- Recall: {summary.get('performance', {}).get('recall', 0):.4f}
- F1 Score: {summary.get('performance', {}).get('f1_score', 0):.4f}

## Feature Importance

![Feature Importance](feature_importance.png)

## Classification Performance

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### ROC Curve

![ROC Curve](roc_curve.png)

### Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)

## Feature Distributions and Correlations

### Feature Distributions

![Feature Distributions](feature_distributions.png)

### Feature Correlations

![Feature Correlations](feature_correlation.png)

## Vector Space Visualization

### PCA Visualization

![PCA Visualization](pca_visualization.png)

### t-SNE Visualization

![t-SNE Visualization](tsne_visualization.png)

## Top Features

| Feature | Importance | Class Difference | Positive Mean | Negative Mean |
|---------|------------|------------------|---------------|---------------|
"""
        
        # Add rows for top features
        for feature in summary.get('top_features', []):
            markdown_content += f"| {feature.get('feature', '')} | {feature.get('importance', 'N/A')} | {feature.get('class_difference', 0):.4f} | {feature.get('positive_mean', 0):.4f} | {feature.get('negative_mean', 0):.4f} |\n"
        
        markdown_content += """

For more detailed analysis, please refer to the CSV files in the report directory.
"""
        
        # Save Markdown report
        with open(os.path.join(self.output_dir, "classification_report.md"), 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Saved Markdown report to {os.path.join(self.output_dir, 'classification_report.md')}")


def generate_classification_report(config: Dict, X_test: np.ndarray, y_test: np.ndarray, 
                                  predictions: np.ndarray, prediction_probs: np.ndarray,
                                  feature_names: List[str], record_pairs: List[Tuple[Dict, Dict]],
                                  model_info: Dict, output_dir: str = None) -> None:
    """
    Generate a comprehensive classification report from test data.
    
    Args:
        config: Configuration dictionary
        X_test: Test feature vectors
        y_test: True labels for test data
        predictions: Predicted labels
        prediction_probs: Prediction probabilities
        feature_names: Names of features
        record_pairs: List of record pairs corresponding to test data
        model_info: Dictionary of model information
        output_dir: Directory for saving reports
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(config.get("output_dir", "output"), "reports")
    
    # Create reporter
    reporter = ClassificationReporter(config, output_dir)
    
    # Add data
    reporter.add_feature_vectors(
        feature_vectors=X_test,
        feature_names=feature_names,
        record_pairs=record_pairs,
        labels=y_test,
        predictions=predictions,
        prediction_probs=prediction_probs
    )
    
    # Add model info
    reporter.add_model_info(model_info)
    
    # Generate comprehensive report
    reporter.generate_comprehensive_report()
    
    logger.info(f"Generated comprehensive classification report in {output_dir}")
    
    return reporter


def extract_misclassified_pairs(feature_vectors: np.ndarray, true_labels: np.ndarray,
                             predictions: np.ndarray, feature_names: List[str],
                             record_pairs: List[Tuple[Dict, Dict]], output_file: str = None) -> pd.DataFrame:
    """
    Extract and analyze misclassified pairs.
    
    Args:
        feature_vectors: Feature vectors
        true_labels: True labels
        predictions: Predicted labels
        feature_names: Names of features
        record_pairs: List of record pairs
        output_file: File to save results
        
    Returns:
        DataFrame of misclassified pairs with features
    """
    # Create DataFrame of feature vectors
    feature_df = pd.DataFrame(feature_vectors, columns=feature_names)
    
    # Add labels and predictions
    feature_df["true_label"] = true_labels
    feature_df["predicted_label"] = predictions
    
    # Add record info
    feature_df["left_id"] = [pair[0].get("id", "unknown") for pair in record_pairs]
    feature_df["right_id"] = [pair[1].get("id", "unknown") for pair in record_pairs]
    feature_df["left_person"] = [pair[0].get("person", "unknown") for pair in record_pairs]
    feature_df["right_person"] = [pair[1].get("person", "unknown") for pair in record_pairs]
    
    # Find misclassified pairs
    misclassified = feature_df[feature_df["true_label"] != feature_df["predicted_label"]]
    
    # Add error type
    misclassified["error_type"] = misclassified.apply(
        lambda x: "False Positive" if x["true_label"] == 0 else "False Negative", 
        axis=1
    )
    
    # Save to file if specified
    if output_file and not misclassified.empty:
        misclassified.to_csv(output_file, index=False)
        logger.info(f"Saved {len(misclassified)} misclassified pairs to {output_file}")
    
    return misclassified