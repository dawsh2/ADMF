# src/core/container.py
import logging
from typing import Callable, Type, Any, Dict, Tuple, Optional # Added Optional

from .exceptions import DependencyNotFoundError

logger = logging.getLogger(__name__)

class Container:
    def __init__(self, parent: Optional['Container'] = None):
        self._instances: Dict[str, Any] = {}
        self._providers: Dict[str, Dict[str, Any]] = {}
        self._parent = parent
        logger.info(f"DI Container initialized{' (with parent)' if parent else ''}.")

    def register_instance(self, name: str, instance: Any):
        if name in self._instances or name in self._providers:
            logger.info(f"Service '{name}' is already registered. Overwriting.")
        self._instances[name] = instance
        logger.debug(f"Instance registered for '{name}': {type(instance).__name__}")

    # MODIFIED register_type signature and logic
    def register_type(self, 
                      name: str, 
                      service_type: Type, 
                      singleton: bool = True, 
                      constructor_args: Optional[Tuple[Any, ...]] = None, 
                      constructor_kwargs: Optional[Dict[str, Any]] = None):
        """
        Registers a class type. The container will instantiate it when resolved.

        Args:
            name (str): The name to register the service type under.
            service_type (Type): The class to be instantiated.
            singleton (bool, optional): If True, instantiate once and cache. Defaults to True.
            constructor_args (Optional[Tuple[Any, ...]], optional): 
                Positional arguments for the class constructor. Defaults to None.
            constructor_kwargs (Optional[Dict[str, Any]], optional): 
                Keyword arguments for the class constructor. Defaults to None.
        """
        if name in self._instances or name in self._providers:
            logger.info(f"Service '{name}' is already registered. Overwriting.")
        
        self._providers[name] = {
            'provider': service_type,
            'args': constructor_args if constructor_args is not None else (),
            'kwargs': constructor_kwargs if constructor_kwargs is not None else {}, # Ensure it's a dict
            'singleton': singleton,
            'is_factory': False
        }
        logger.debug(f"Type '{service_type.__name__}' registered for service name '{name}' (singleton={singleton}). Constructor_kwargs: {self._providers[name]['kwargs']}")

    def register_factory(self, 
                         name: str, 
                         factory: Callable[..., Any], 
                         singleton: bool = True, 
                         factory_args: Optional[Tuple[Any, ...]] = None, 
                         factory_kwargs: Optional[Dict[str, Any]] = None):
        if name in self._instances or name in self._providers:
            logger.info(f"Service '{name}' is already registered. Overwriting.")
        self._providers[name] = {
            'provider': factory,
            'args': factory_args if factory_args is not None else (),
            'kwargs': factory_kwargs if factory_kwargs is not None else {},
            'singleton': singleton,
            'is_factory': True
        }
        logger.debug(f"Factory '{factory.__name__}' registered for '{name}' (singleton={singleton}).")
    
    def resolve(self, name: str) -> Any:
        # First, try to resolve locally
        if name in self._instances:
            logger.debug(f"Resolving '{name}' from cached instance.")
            return self._instances[name]

        if name in self._providers:
            provider_info = self._providers[name]
            provider_callable = provider_info['provider']
            args_for_provider = provider_info['args']
            kwargs_for_provider = provider_info['kwargs']
            is_singleton = provider_info['singleton']
            
            logger.debug(f"Resolving '{name}' by creating new instance via its provider (singleton={is_singleton}). Args: {args_for_provider}, Kwargs: {kwargs_for_provider}")
            
            try:
                instance = provider_callable(*args_for_provider, **kwargs_for_provider)
            except Exception as e:
                logger.error(f"Error creating instance for '{name}': {e}", exc_info=True)
                raise DependencyNotFoundError(f"Failed to create instance for service '{name}'. Error: {e}") from e
            
            if is_singleton:
                self._instances[name] = instance
            
            return instance

        # If not found locally, try parent container
        if self._parent:
            logger.debug(f"Service '{name}' not found locally, checking parent container.")
            try:
                return self._parent.resolve(name)
            except DependencyNotFoundError:
                pass  # Continue to throw our own error below

        logger.error(f"No service or provider found for name '{name}'.")
        raise DependencyNotFoundError(f"Service '{name}' not found in container or parent container.")

    def is_registered(self, name: str, check_parent: bool = True) -> bool:
        """Check if a service is registered in this container or parent."""
        # Check local registration
        if name in self._instances or name in self._providers:
            return True
            
        # Check parent if requested
        if check_parent and self._parent:
            return self._parent.is_registered(name)
            
        return False
    
    def reset(self) -> None:
        """Clear all registrations and instances from this container."""
        self._instances.clear()
        self._providers.clear()
        logger.info("Container reset - all registrations cleared.")
        
    def register(self, name: str, service: Any, singleton: bool = True) -> None:
        """
        Convenience method to register either an instance or a type.
        
        If service is already an instance, use register_instance.
        If service is a type, use register_type.
        """
        if isinstance(service, type):
            self.register_type(name, service, singleton=singleton)
        else:
            self.register_instance(name, service)
