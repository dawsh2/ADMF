"""
OptimizationTarget interface for optimizable components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

from ...base.parameter import ParameterSpace, ParameterSet


class OptimizationTarget(ABC):
    """
    Interface that makes a component optimizable.
    
    Any component that can be optimized (Strategy, Indicator, Rule, etc.)
    should implement this interface.
    """
    
    @abstractmethod
    def get_parameter_space(self) -> ParameterSpace:
        """
        Get the parameter space for optimization.
        
        Returns:
            ParameterSpace defining all optimizable parameters
        """
        pass
        
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameter values on the component.
        
        Args:
            params: Dictionary of parameter values
        """
        pass
        
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary of current parameter values
        """
        pass
        
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate parameter values.
        
        Args:
            params: Parameter values to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Default implementation uses parameter space validation
        space = self.get_parameter_space()
        valid, errors = space.validate(params)
        
        if not valid:
            return False, "; ".join(errors)
            
        return True, None
        
    def reset(self) -> None:
        """
        Reset component state.
        
        Called between optimization iterations to ensure clean state.
        """
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get component metadata for optimization tracking.
        
        Returns:
            Dictionary of metadata (name, version, etc.)
        """
        return {}
        

class OptimizationTargetWrapper:
    """
    Wrapper to make any component an OptimizationTarget.
    
    Useful for components that have the methods but don't
    explicitly implement the interface.
    """
    
    def __init__(self, component: Any, name: Optional[str] = None):
        self.component = component
        self.name = name or getattr(component, 'name', 'unnamed')
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space from wrapped component."""
        if hasattr(self.component, 'get_parameter_space'):
            return self.component.get_parameter_space()
        else:
            # Return empty space
            return ParameterSpace(f"{self.name}_params")
            
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters on wrapped component."""
        if hasattr(self.component, 'set_parameters'):
            self.component.set_parameters(params)
        else:
            # Try setting as attributes
            for key, value in params.items():
                if hasattr(self.component, key):
                    setattr(self.component, key, value)
                    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters from wrapped component."""
        if hasattr(self.component, 'get_parameters'):
            return self.component.get_parameters()
        else:
            # Return empty dict
            return {}
            
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters."""
        if hasattr(self.component, 'validate_parameters'):
            return self.component.validate_parameters(params)
        else:
            # Default to valid
            return True, None
            
    def reset(self) -> None:
        """Reset wrapped component."""
        if hasattr(self.component, 'reset'):
            self.component.reset()
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from wrapped component."""
        metadata = {
            'name': self.name,
            'type': type(self.component).__name__
        }
        
        if hasattr(self.component, 'get_metadata'):
            metadata.update(self.component.get_metadata())
            
        return metadata