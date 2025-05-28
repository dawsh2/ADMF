"""
Parameter management for strategies and optimization.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import copy


@dataclass
class Parameter:
    """Definition of a single parameter."""
    name: str
    param_type: str  # 'discrete', 'continuous', 'categorical'
    
    # For discrete parameters
    values: Optional[List[Any]] = None
    
    # For continuous parameters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    
    # Common attributes
    default: Optional[Any] = None
    description: Optional[str] = None
    constraints: Optional[List[str]] = None  # e.g., ["value > other_param"]
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a parameter value."""
        if self.param_type == 'discrete':
            if self.values and value not in self.values:
                return False, f"Value {value} not in allowed values: {self.values}"
                
        elif self.param_type == 'continuous':
            if self.min_value is not None and value < self.min_value:
                return False, f"Value {value} below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Value {value} above maximum {self.max_value}"
                
        return True, None
        
    def sample(self, method: str = 'grid', n_samples: Optional[int] = None) -> List[Any]:
        """Generate sample values for optimization."""
        if self.param_type == 'discrete':
            return self.values or []
            
        elif self.param_type == 'continuous':
            if method == 'grid':
                if self.step:
                    # Use step size
                    current = self.min_value or 0
                    max_val = self.max_value or current + 1
                    samples = []
                    while current <= max_val:
                        samples.append(current)
                        current += self.step
                    return samples
                elif n_samples:
                    # Use number of samples
                    import numpy as np
                    return np.linspace(
                        self.min_value or 0,
                        self.max_value or 1,
                        n_samples
                    ).tolist()
                    
        return [self.default] if self.default is not None else []


class ParameterSpace:
    """
    Defines the parameter space for optimization.
    
    Supports hierarchical namespacing for complex strategies.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._parameters: Dict[str, Parameter] = {}
        self._subspaces: Dict[str, 'ParameterSpace'] = {}
        
    def add_parameter(self, param: Union[Parameter, str], **kwargs) -> None:
        """Add a parameter to the space."""
        if isinstance(param, str):
            # Create parameter from kwargs
            param = Parameter(name=param, **kwargs)
            
        self._parameters[param.name] = param
        
    def add_subspace(self, name: str, subspace: 'ParameterSpace') -> None:
        """Add a sub-parameter space (for nested components)."""
        self._subspaces[name] = subspace
        
    def get_parameter(self, path: str) -> Optional[Parameter]:
        """Get parameter by path (supports dot notation)."""
        parts = path.split('.')
        
        if len(parts) == 1:
            return self._parameters.get(path)
        else:
            # Navigate subspaces
            subspace_name = parts[0]
            if subspace_name in self._subspaces:
                return self._subspaces[subspace_name].get_parameter('.'.join(parts[1:]))
                
        return None
        
    def update_parameter(self, name: str, param: Parameter) -> None:
        """Update an existing parameter."""
        self._parameters[name] = param
        
    def sample(self, method: str = 'grid', n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for optimization.
        
        Returns list of parameter dictionaries with full paths.
        """
        import logging
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # First, get all parameter paths and their samples
        param_samples = {}
        
        # Local parameters
        logger.info(f"[{self.name}] Sampling local parameters:")
        for name, param in self._parameters.items():
            values = param.sample(method, n_samples)
            param_samples[name] = values
            logger.info(f"  - {name}: {len(values)} values = {values}")
            
        # Subspace parameters
        subspace_combinations = {}
        if self._subspaces:
            logger.info(f"[{self.name}] Processing subspaces:")
        for subspace_name, subspace in self._subspaces.items():
            logger.info(f"  - Subspace '{subspace_name}':")
            subspace_samples = subspace.sample(method, n_samples)
            logger.info(f"    Generated {len(subspace_samples)} combinations from subspace")
            
            # Store complete subspace combinations
            subspace_combinations[subspace_name] = subspace_samples
                    
        # Log parameter expansion
        logger.info(f"[{self.name}] Parameter expansion summary:")
        for param_name, values in param_samples.items():
            logger.info(f"  - {param_name}: {len(values)} values")
        for subspace_name, combos in subspace_combinations.items():
            logger.info(f"  - {subspace_name} (subspace): {len(combos)} combinations")
                    
        # Generate all combinations
        if not param_samples and not subspace_combinations:
            return [{}]
            
        # Use itertools to create cartesian product
        import itertools
        
        # Start with local parameters
        if param_samples:
            keys = list(param_samples.keys())
            values = [param_samples[k] for k in keys]
            
            # Calculate combinations for local params
            local_combinations = []
            for combo in itertools.product(*values):
                local_combinations.append(dict(zip(keys, combo)))
        else:
            local_combinations = [{}]
            
        # Now combine with subspace combinations
        final_combinations = []
        
        if subspace_combinations:
            # Get all subspace combination lists
            subspace_lists = []
            subspace_names = []
            for name, combos in subspace_combinations.items():
                subspace_lists.append(combos)
                subspace_names.append(name)
            
            # Create cartesian product of local combinations with each subspace combination
            for local_combo in local_combinations:
                for subspace_combo_tuple in itertools.product(*subspace_lists):
                    # Merge local parameters with subspace parameters
                    combined = local_combo.copy()
                    
                    # Add namespaced subspace parameters
                    for i, subspace_combo in enumerate(subspace_combo_tuple):
                        subspace_name = subspace_names[i]
                        for key, value in subspace_combo.items():
                            combined[f"{subspace_name}.{key}"] = value
                    
                    final_combinations.append(combined)
        else:
            final_combinations = local_combinations
            
        # Calculate and log total combinations
        logger.warning(f"[{self.name}] Total parameter combinations: {len(final_combinations)}")
        if len(final_combinations) <= 5:
            for i, combo in enumerate(final_combinations):
                logger.info(f"  Combination {i+1}: {combo}")
            
        return final_combinations
        
    def validate(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a parameter set."""
        errors = []
        
        for path, value in params.items():
            param = self.get_parameter(path)
            if param:
                valid, error = param.validate(value)
                if not valid:
                    errors.append(f"{path}: {error}")
                    
        return len(errors) == 0, errors
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'parameters': {
                name: {
                    'type': p.param_type,
                    'default': p.default,
                    'values': p.values,
                    'min': p.min_value,
                    'max': p.max_value,
                    'step': p.step
                }
                for name, p in self._parameters.items()
            },
            'subspaces': {
                name: space.to_dict()
                for name, space in self._subspaces.items()
            }
        }


class ParameterSet:
    """
    Manages a set of parameters with versioning and metadata.
    
    This is an immutable snapshot of parameters for reproducibility.
    """
    
    def __init__(self, name: str, version: Optional[str] = None):
        self.name = name
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._parameters: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'name': name,
            'version': self.version
        }
        
    def set(self, path: str, value: Any) -> None:
        """Set a parameter value."""
        self._parameters[path] = value
        
    def get(self, path: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self._parameters.get(path, default)
        
    def update(self, params: Dict[str, Any]) -> None:
        """Update multiple parameters."""
        self._parameters.update(params)
        
    def get_namespaced_params(self) -> Dict[str, Dict[str, Any]]:
        """Get parameters organized by namespace."""
        namespaced = {}
        
        for path, value in self._parameters.items():
            parts = path.split('.')
            if len(parts) > 1:
                namespace = parts[0]
                param_name = '.'.join(parts[1:])
                
                if namespace not in namespaced:
                    namespaced[namespace] = {}
                    
                namespaced[namespace][param_name] = value
            else:
                # Top-level parameter
                if 'root' not in namespaced:
                    namespaced['root'] = {}
                namespaced['root'][path] = value
                
        return namespaced
        
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the parameter set."""
        self._metadata[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'parameters': copy.deepcopy(self._parameters),
            'metadata': copy.deepcopy(self._metadata)
        }
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSet':
        """Create from dictionary representation."""
        metadata = data.get('metadata', {})
        param_set = cls(
            name=metadata.get('name', 'unnamed'),
            version=metadata.get('version')
        )
        
        param_set._parameters = copy.deepcopy(data.get('parameters', {}))
        param_set._metadata = copy.deepcopy(metadata)
        
        return param_set
        
    @classmethod
    def from_json(cls, json_str: str) -> 'ParameterSet':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
        
    def clone(self, new_version: Optional[str] = None) -> 'ParameterSet':
        """Create a copy with a new version."""
        new_set = ParameterSet(
            name=self.name,
            version=new_version or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        new_set._parameters = copy.deepcopy(self._parameters)
        new_set._metadata = copy.deepcopy(self._metadata)
        new_set._metadata['version'] = new_set.version
        new_set._metadata['cloned_from'] = self.version
        new_set._metadata['cloned_at'] = datetime.now().isoformat()
        
        return new_set