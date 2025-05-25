"""
ParameterManager - Basic parameter versioning and management system.

This module provides basic parameter versioning functionality as a quick win
towards implementing the full STRATEGY_LIFECYCLE_MANAGEMENT specification.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
from dataclasses import dataclass, asdict


@dataclass
class ParameterMetadata:
    """Metadata for a parameter set."""
    version_id: str
    created_at: str
    strategy_name: str
    optimization_method: str
    training_period: Dict[str, str]
    performance_metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    regime: Optional[str] = None
    parent_version: Optional[str] = None
    notes: Optional[str] = None


@dataclass 
class VersionedParameterSet:
    """A versioned set of strategy parameters."""
    parameters: Dict[str, Any]
    metadata: ParameterMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameters": self.parameters,
            "metadata": asdict(self.metadata)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionedParameterSet':
        """Create from dictionary."""
        metadata = ParameterMetadata(**data["metadata"])
        return cls(parameters=data["parameters"], metadata=metadata)


class ParameterManager:
    """
    Manages versioned parameter sets for strategies.
    
    This provides basic functionality for:
    - Creating versioned parameter sets
    - Storing and retrieving parameters
    - Tracking parameter lineage
    - Loading parameters for production use
    """
    
    def __init__(self, storage_dir: str = "parameter_versions"):
        """
        Initialize the ParameterManager.
        
        Args:
            storage_dir: Directory to store parameter versions
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for loaded parameters
        self._cache = {}
        
    def create_version(
        self,
        parameters: Dict[str, Any],
        strategy_name: str,
        optimization_method: str,
        training_period: Dict[str, str],
        performance_metrics: Dict[str, float],
        dataset_info: Dict[str, Any],
        regime: Optional[str] = None,
        parent_version: Optional[str] = None,
        notes: Optional[str] = None
    ) -> VersionedParameterSet:
        """
        Create a new versioned parameter set.
        
        Args:
            parameters: The strategy parameters
            strategy_name: Name of the strategy
            optimization_method: Method used for optimization
            training_period: Start and end dates of training
            performance_metrics: Performance metrics achieved
            dataset_info: Information about the dataset used
            regime: Market regime if regime-specific
            parent_version: Version ID of parent parameters
            notes: Optional notes about this version
            
        Returns:
            VersionedParameterSet with generated version ID
        """
        # Generate version ID
        version_id = self._generate_version_id(
            parameters, 
            strategy_name, 
            optimization_method,
            regime
        )
        
        # Create metadata
        metadata = ParameterMetadata(
            version_id=version_id,
            created_at=datetime.now().isoformat(),
            strategy_name=strategy_name,
            optimization_method=optimization_method,
            training_period=training_period,
            performance_metrics=performance_metrics,
            dataset_info=dataset_info,
            regime=regime,
            parent_version=parent_version,
            notes=notes
        )
        
        # Create versioned parameter set
        param_set = VersionedParameterSet(parameters=parameters, metadata=metadata)
        
        # Save to storage
        self._save_version(param_set)
        
        return param_set
        
    def _generate_version_id(
        self,
        parameters: Dict[str, Any],
        strategy_name: str,
        optimization_method: str,
        regime: Optional[str] = None
    ) -> str:
        """Generate a unique version ID."""
        version_data = {
            "parameters": parameters,
            "strategy": strategy_name,
            "method": optimization_method,
            "regime": regime,
            "timestamp": datetime.now().isoformat()
        }
        
        version_string = json.dumps(version_data, sort_keys=True)
        version_hash = hashlib.sha256(version_string.encode()).hexdigest()
        
        return version_hash[:16]  # Use first 16 chars for readability
        
    def _save_version(self, param_set: VersionedParameterSet) -> None:
        """Save a versioned parameter set to storage."""
        filename = f"{param_set.metadata.strategy_name}_{param_set.metadata.version_id}.json"
        filepath = self.storage_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(param_set.to_dict(), f, indent=2)
            
        self.logger.info(f"Saved parameter version {param_set.metadata.version_id} to {filepath}")
        
        # Update cache
        self._cache[param_set.metadata.version_id] = param_set
        
    def load_version(self, version_id: str) -> Optional[VersionedParameterSet]:
        """
        Load a specific parameter version.
        
        Args:
            version_id: Version ID to load
            
        Returns:
            VersionedParameterSet or None if not found
        """
        # Check cache first
        if version_id in self._cache:
            return self._cache[version_id]
            
        # Search in storage
        for filepath in self.storage_dir.glob("*.json"):
            if version_id in filepath.name:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    param_set = VersionedParameterSet.from_dict(data)
                    self._cache[version_id] = param_set
                    return param_set
                except Exception as e:
                    self.logger.error(f"Error loading version from {filepath}: {e}")
                    
        return None
        
    def load_latest(
        self, 
        strategy_name: str,
        regime: Optional[str] = None
    ) -> Optional[VersionedParameterSet]:
        """
        Load the latest parameter version for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            regime: Optional regime filter
            
        Returns:
            Latest VersionedParameterSet or None
        """
        # Find all versions for this strategy
        pattern = f"{strategy_name}_*.json"
        files = list(self.storage_dir.glob(pattern))
        
        if not files:
            return None
            
        # Load and filter by regime if specified
        candidates = []
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                param_set = VersionedParameterSet.from_dict(data)
                
                if regime is None or param_set.metadata.regime == regime:
                    candidates.append(param_set)
            except Exception as e:
                self.logger.error(f"Error loading {filepath}: {e}")
                
        if not candidates:
            return None
            
        # Return most recent
        return max(candidates, key=lambda ps: ps.metadata.created_at)
        
    def export_for_production(
        self,
        strategy_name: str,
        output_file: str = "production_parameters.json"
    ) -> str:
        """
        Export parameters in production-ready format.
        
        Args:
            strategy_name: Name of the strategy
            output_file: Output filename
            
        Returns:
            Path to exported file
        """
        # Collect all regime-specific parameters
        production_params = {}
        regime_versions = {}
        
        for filepath in self.storage_dir.glob(f"{strategy_name}_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                param_set = VersionedParameterSet.from_dict(data)
                
                if param_set.metadata.regime:
                    # Keep only the latest version for each regime
                    regime = param_set.metadata.regime
                    if (regime not in production_params or 
                        param_set.metadata.created_at > regime_versions[regime]):
                        production_params[regime] = param_set.parameters
                        regime_versions[regime] = param_set.metadata.created_at
                        
            except Exception as e:
                self.logger.error(f"Error processing {filepath}: {e}")
                
        # Save in production format
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(production_params, f, indent=2)
            
        self.logger.info(f"Exported production parameters to {output_path}")
        return str(output_path)
        
    def get_parameter_history(
        self,
        strategy_name: str,
        regime: Optional[str] = None
    ) -> List[VersionedParameterSet]:
        """
        Get the history of parameter versions.
        
        Args:
            strategy_name: Name of the strategy
            regime: Optional regime filter
            
        Returns:
            List of VersionedParameterSet ordered by creation time
        """
        history = []
        
        for filepath in self.storage_dir.glob(f"{strategy_name}_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                param_set = VersionedParameterSet.from_dict(data)
                
                if regime is None or param_set.metadata.regime == regime:
                    history.append(param_set)
                    
            except Exception as e:
                self.logger.error(f"Error loading {filepath}: {e}")
                
        # Sort by creation time
        history.sort(key=lambda ps: ps.metadata.created_at)
        return history
        
    def compare_versions(
        self,
        version1_id: str,
        version2_id: str
    ) -> Dict[str, Any]:
        """
        Compare two parameter versions.
        
        Args:
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Comparison summary
        """
        v1 = self.load_version(version1_id)
        v2 = self.load_version(version2_id)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
            
        # Find parameter differences
        param_diffs = {}
        all_keys = set(v1.parameters.keys()) | set(v2.parameters.keys())
        
        for key in all_keys:
            val1 = v1.parameters.get(key)
            val2 = v2.parameters.get(key)
            if val1 != val2:
                param_diffs[key] = {"v1": val1, "v2": val2}
                
        # Compare performance
        perf_comparison = {}
        for metric in set(v1.metadata.performance_metrics.keys()) | set(v2.metadata.performance_metrics.keys()):
            perf_comparison[metric] = {
                "v1": v1.metadata.performance_metrics.get(metric),
                "v2": v2.metadata.performance_metrics.get(metric)
            }
            
        return {
            "version1": version1_id,
            "version2": version2_id,
            "parameter_differences": param_diffs,
            "performance_comparison": perf_comparison,
            "metadata": {
                "v1": asdict(v1.metadata),
                "v2": asdict(v2.metadata)
            }
        }