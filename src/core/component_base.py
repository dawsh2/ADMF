#!/usr/bin/env python3
"""
Base component class for ADMF system.

This module defines the base class that all system components must inherit from.
It formalizes the component contract and lifecycle methods as defined in
COMPONENT_LIFECYCLE.md.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

from .subscription_manager import SubscriptionManager


class ComponentState(Enum):
    """Component lifecycle states per COMPONENT_LIFECYCLE.md"""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    DISPOSED = "disposed"


class ComponentBase(ABC):
    """
    Base class for all managed components in the ADMF system.
    
    This class defines the standard lifecycle and interface that all components
    must implement. The lifecycle follows these states:
    
    1. CREATED - After __init__, minimal setup only
    2. INITIALIZED - After initialize(), dependencies injected
    3. RUNNING - After start(), actively processing
    4. STOPPED - After stop(), halted but state preserved
    5. DISPOSED - After teardown(), resources released
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """
        Minimal constructor - no external dependencies.
        
        As per COMPONENT_LIFECYCLE.md, the constructor is responsible for:
        - Setting component name and parameters
        - Initializing internal state variables
        - NOT accessing external resources or services
        - NOT requiring dependencies
        
        Args:
            instance_name: Unique name for this component instance
            config_key: Key in configuration for this component
        """
        self.instance_name = instance_name
        self.config_key = config_key
        self._state = ComponentState.CREATED
        self.initialized = False
        self.running = False
        
        # These will be set during initialize()
        self.config_loader = None
        self.config = None
        self.event_bus = None
        self.container = None
        self.logger = None
        self.component_config = {}
        self.subscription_manager: Optional[SubscriptionManager] = None
        self._context = {}
        
    @property
    def state(self) -> ComponentState:
        """Get current component state."""
        return self._state
        
    def initialize(self, context: Dict[str, Any]) -> None:
        """
        Initialize component with dependencies from context.
        
        As per COMPONENT_LIFECYCLE.md, the initialize method is responsible for:
        - Extracting dependencies from the context
        - Setting up resources and connections
        - Initializing event subscriptions
        - Validating configuration
        
        Args:
            context: Dictionary containing:
                - config_loader: Configuration loader
                - config: Full configuration
                - event_bus: Event bus instance
                - container: DI container
                - logger: Logger instance
        """
        # Extract standard dependencies
        self.config_loader = context.get('config_loader')
        self.config = context.get('config')
        self.event_bus = context.get('event_bus')
        self.container = context.get('container')
        self.logger = context.get('logger') or logging.getLogger(self.instance_name)
        
        # Store context for subclasses
        self._context = context
        
        # Get component-specific config
        if self.config_key and self.config:
            self.component_config = self.config.get('components', {}).get(self.config_key, {})
        else:
            self.component_config = {}
            
        # Initialize subscription manager and event subscriptions if event bus available
        if self.event_bus:
            self.subscription_manager = SubscriptionManager(self.event_bus)
            self.initialize_event_subscriptions()
            
        # Call component-specific initialization
        self._initialize()
        
        # Validate configuration
        self._validate_configuration()
        
        self.initialized = True
        self._state = ComponentState.INITIALIZED
        self.logger.info(f"Component {self.instance_name} initialized")
        
    @abstractmethod
    def _initialize(self) -> None:
        """
        Component-specific initialization logic.
        
        Subclasses should implement this to perform their specific
        initialization tasks like:
        - Setting up internal data structures
        - Initializing indicators or models
        - Preparing resources
        """
        pass
        
    def _start(self) -> None:
        """
        Component-specific start logic.
        
        Subclasses can override this to perform specific actions when starting.
        Called after the component state is set to RUNNING.
        """
        pass
        
    def _stop(self) -> None:
        """
        Component-specific stop logic.
        
        Subclasses can override this to perform specific actions when stopping.
        Called before the component state is set to STOPPED.
        """
        pass
        
    def _cleanup(self) -> None:
        """
        Component-specific cleanup logic.
        
        Subclasses can override this to perform specific cleanup actions.
        Called during teardown before state is set to DISPOSED.
        """
        pass
        
    def initialize_event_subscriptions(self) -> None:
        """
        Set up event subscriptions.
        
        Subclasses should override this method to subscribe to events
        they're interested in. Called automatically during initialize()
        if an event bus is available.
        
        The subscription_manager is already initialized and available for use.
        
        Example:
            self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        """
        pass
        
    def _validate_configuration(self) -> None:
        """
        Validate component configuration.
        
        Subclasses can override this to validate their specific
        configuration requirements.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
        
    def start(self) -> None:
        """
        Begin component operation.
        
        As per COMPONENT_LIFECYCLE.md, the start method is responsible for:
        - Beginning active operations
        - Starting background threads or tasks
        - Initiating data processing
        """
        if not self.initialized:
            raise RuntimeError(
                f"Component {self.instance_name} must be initialized before starting"
            )
            
        self.running = True
        self._state = ComponentState.RUNNING
        self._start()
        self.logger.info(f"Component {self.instance_name} started")
        
    def stop(self) -> None:
        """
        End component operation.
        
        As per COMPONENT_LIFECYCLE.md, the stop method is responsible for:
        - Halting active operations
        - Stopping background threads or tasks
        - Preserving state for potential restart
        """
        self._stop()
        self.running = False
        self._state = ComponentState.STOPPED
        self.logger.info(f"Component {self.instance_name} stopped")
        
    def reset(self) -> None:
        """
        Reset component state.
        
        As per COMPONENT_LIFECYCLE.md, the reset method is responsible for:
        - Clearing internal state
        - Preserving configuration
        - Preparing for a new run
        - Ensuring clean state separation between runs
        
        Note: Configuration (self.component_config) is preserved
        """
        self.logger.debug(f"Component {self.instance_name} reset")
        
    def teardown(self) -> None:
        """
        Release resources and perform final cleanup.
        
        As per COMPONENT_LIFECYCLE.md, the teardown method is responsible for:
        - Releasing external resources
        - Closing connections
        - Unsubscribing from events
        - Final cleanup before destruction
        """
        # Unsubscribe from all events using subscription manager
        if self.subscription_manager:
            self.subscription_manager.unsubscribe_all()
            
        # Call component-specific cleanup
        self._cleanup()
            
        self.initialized = False
        self.running = False
        self._state = ComponentState.DISPOSED
        self.logger.info(f"Component {self.instance_name} teardown complete")
        
    def dispose(self) -> None:
        """Alias for teardown() for compatibility with test."""
        self.teardown()
        
    @property
    def name(self) -> str:
        """Get component name (for compatibility)."""
        return self.instance_name
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status information.
        
        Returns:
            Dictionary containing component status
        """
        return {
            'name': self.instance_name,
            'type': self.__class__.__name__,
            'initialized': self.initialized,
            'running': self.running,
            'config_key': self.config_key,
            'has_event_bus': self.event_bus is not None,
            'has_container': self.container is not None
        }
        
    def get_specific_config(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value for this component.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.component_config.get(key, default)
        
    def __repr__(self) -> str:
        """String representation of component."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.instance_name}', "
            f"initialized={self.initialized}, "
            f"running={self.running})"
        )