"""
ResultsManager - Handles storage, retrieval, and analysis of optimization results.

This module extracts results management logic from EnhancedOptimizer to provide
a clean interface for handling optimization outputs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import hashlib


class ResultsManager:
    """
    Manages optimization results including storage, retrieval, and analysis.
    
    This class provides methods for:
    - Saving optimization results with versioning
    - Loading previous results
    - Analyzing and comparing results
    - Generating result summaries
    """
    
    def __init__(self, results_dir: str = "optimization_results"):
        """
        Initialize the ResultsManager.
        
        Args:
            results_dir: Directory to store optimization results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def save_results(
        self, 
        results: Dict[str, Any], 
        optimization_type: str = "grid_search",
        version_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save optimization results with versioning support.
        
        Args:
            results: Optimization results dictionary
            optimization_type: Type of optimization performed
            version_metadata: Additional metadata for versioning
            
        Returns:
            Path to saved results file
        """
        # Generate version ID based on parameters and timestamp
        version_id = self._generate_version_id(results, version_metadata)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with version
        filename = f"{optimization_type}_{timestamp}_{version_id[:8]}.json"
        filepath = self.results_dir / filename
        
        # Add metadata to results
        results_with_metadata = {
            "version_id": version_id,
            "timestamp": timestamp,
            "optimization_type": optimization_type,
            "metadata": version_metadata or {},
            "results": results
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
            
        self.logger.info(f"Saved optimization results to {filepath}")
        
        # Also save regime-specific parameters separately if available
        if "best_parameters_per_regime" in results:
            self._save_regime_parameters(results["best_parameters_per_regime"], version_id)
            
        return str(filepath)
        
    def _generate_version_id(
        self, 
        results: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a unique version ID for the results."""
        # Create a string representation of key parameters
        version_data = {
            "parameters": results.get("best_parameters_on_train", {}),
            "metric": results.get("metric_optimized", ""),
            "dataset_info": metadata.get("dataset_info", {}) if metadata else {}
        }
        
        # Generate hash
        version_string = json.dumps(version_data, sort_keys=True)
        version_hash = hashlib.sha256(version_string.encode()).hexdigest()
        
        return version_hash
        
    def _save_regime_parameters(
        self, 
        regime_params: Dict[str, Any], 
        version_id: str
    ) -> None:
        """Save regime-specific parameters in the production format."""
        # Format for production use
        production_format = {}
        
        for regime, regime_data in regime_params.items():
            if isinstance(regime_data, dict) and 'parameters' in regime_data:
                params = regime_data['parameters']
            else:
                params = regime_data
                
            production_format[regime] = params
            
        # Save with version
        filename = f"regime_parameters_v{version_id[:8]}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(production_format, f, indent=2)
            
        # Also save to the standard location for backward compatibility
        standard_path = Path("regime_optimized_parameters.json")
        with open(standard_path, 'w') as f:
            json.dump(production_format, f, indent=2)
            
        self.logger.info(f"Saved regime parameters to {filepath} and {standard_path}")
        
    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load optimization results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Results dictionary or None if not found
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Handle both old and new formats
            if "results" in data:
                return data["results"]
            else:
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading results from {filepath}: {e}")
            return None
            
    def get_latest_results(
        self, 
        optimization_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent optimization results.
        
        Args:
            optimization_type: Filter by optimization type
            
        Returns:
            Latest results dictionary or None
        """
        pattern = f"{optimization_type}_*.json" if optimization_type else "*.json"
        files = list(self.results_dir.glob(pattern))
        
        if not files:
            return None
            
        # Sort by modification time
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return self.load_results(str(latest_file))
        
    def get_top_performers(
        self,
        results: List[Dict[str, Any]],
        metric_name: str,
        n: int = 10,
        higher_is_better: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get top N performing parameter sets from results.
        
        Args:
            results: List of result dictionaries
            metric_name: Name of metric to sort by
            n: Number of top results to return
            higher_is_better: Whether higher metric values are better
            
        Returns:
            List of (parameters, metric_value) tuples
        """
        # Filter results with valid metrics
        valid_results = []
        for result in results:
            metric_value = result.get('metrics', {}).get(metric_name)
            if metric_value is not None and metric_value != "N/A":
                valid_results.append((result['parameters'], metric_value))
                
        if not valid_results:
            self.logger.warning("No valid results with metrics found")
            return []
            
        # Sort by metric value
        sorted_results = sorted(
            valid_results,
            key=lambda x: x[1],
            reverse=higher_is_better
        )
        
        # Return top N
        return sorted_results[:min(n, len(sorted_results))]
        
    def generate_summary(
        self, 
        results: Dict[str, Any],
        output_format: str = "text"
    ) -> str:
        """
        Generate a summary of optimization results.
        
        Args:
            results: Optimization results dictionary
            output_format: Format for output ("text", "json", "html")
            
        Returns:
            Formatted summary string
        """
        if output_format == "json":
            return json.dumps(results, indent=2, default=str)
            
        # Text format
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("OPTIMIZATION RESULTS SUMMARY")
        summary_lines.append("=" * 80)
        
        # Best overall parameters
        if "best_parameters_on_train" in results and results["best_parameters_on_train"]:
            params_str = self._format_parameters(results["best_parameters_on_train"])
            summary_lines.append(f"Best overall parameters: {params_str}")
            
        # Metrics
        if "best_training_metric_value" in results:
            train_metric = results["best_training_metric_value"]
            test_metric = results.get("test_set_metric_value_for_best_params", "N/A")
            metric_name = results.get("metric_optimized", "metric")
            summary_lines.append(
                f"Training {metric_name}: {train_metric} | Test {metric_name}: {test_metric}"
            )
            
        # Regime-specific results
        if "best_parameters_per_regime" in results:
            summary_lines.append("\nREGIME-SPECIFIC OPTIMAL PARAMETERS:")
            for regime, regime_data in sorted(results["best_parameters_per_regime"].items()):
                regime_summary = self._format_regime_result(regime, regime_data)
                summary_lines.append(f"  {regime_summary}")
                
        # Adaptive test results
        if "regime_adaptive_test_results" in results:
            summary_lines.extend(self._format_adaptive_results(
                results["regime_adaptive_test_results"]
            ))
            
        return "\n".join(summary_lines)
        
    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """Format parameters dictionary for display."""
        if params is None:
            return "None"
        return ", ".join([
            f"{k.split('.')[-1]}: {v}" 
            for k, v in params.items()
        ])
        
    def _format_regime_result(self, regime: str, regime_data: Dict[str, Any]) -> str:
        """Format a single regime result for display."""
        if isinstance(regime_data, dict) and 'parameters' in regime_data:
            params = regime_data['parameters']
            metric_info = regime_data.get('metric', {})
            metric_name = metric_info.get('name', 'metric')
            metric_value = metric_info.get('value', 'N/A')
        else:
            params = regime_data
            metric_name = 'metric'
            metric_value = 'N/A'
            
        params_str = self._format_parameters(params)
        return f"{regime}: {metric_name}={metric_value} | Params: {params_str}"
        
    def _format_adaptive_results(self, adaptive_results: Dict[str, Any]) -> List[str]:
        """Format adaptive test results for display."""
        lines = []
        lines.append("\n" + "=" * 50)
        lines.append("ADAPTIVE TEST RESULTS")
        lines.append("=" * 50)
        
        if "adaptive_metric" in adaptive_results:
            metric = adaptive_results["adaptive_metric"]
            lines.append(f"Final portfolio value: {metric}")
            
        if "regimes_detected" in adaptive_results:
            regimes = adaptive_results["regimes_detected"]
            lines.append(f"Regimes detected: {', '.join(regimes)}")
            
        return lines
        
    def compare_results(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Compare two optimization results.
        
        Args:
            results1: First results dictionary
            results2: Second results dictionary  
            metric_name: Metric to compare
            
        Returns:
            Comparison summary
        """
        comparison = {
            "metric": metric_name,
            "result1_value": results1.get(metric_name, "N/A"),
            "result2_value": results2.get(metric_name, "N/A"),
            "improvement": None
        }
        
        # Calculate improvement if both values are numeric
        val1 = results1.get(metric_name)
        val2 = results2.get(metric_name)
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            comparison["improvement"] = ((val2 - val1) / val1) * 100
            
        return comparison