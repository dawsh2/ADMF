"""
Bootstrap System for ADMF
========================

The Bootstrap system provides a centralized initialization framework for the ADMF 
trading system, ensuring consistent component creation, dependency management, and
lifecycle control across all run modes (production, optimization, backtest, test).

Key responsibilities:
- Single source of truth for system initialization (BOOTSTRAP_SYSTEM.md)
- Component discovery and dynamic loading (DEPENDENCY_MANAGEMENT.md)
- Dependency graph validation and cycle detection
- Lifecycle state management (COMPONENT_LIFECYCLE.md)
- Clean state isolation between runs
- Hook system for extensibility

Component Lifecycle:
- CREATED: Component instance exists (minimal constructor)
- INITIALIZED: Dependencies injected, component ready
- RUNNING: Component actively processing
- STOPPED: Component halted but state preserved
- DISPOSED: Component cleaned up, resources released
"""

import os
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

# Import core system components
from .config import SimpleConfigLoader
from .container import Container
from .event_bus import EventBus
from .logging_setup import setup_logging

# Import required components based on COMPONENT_ARCHITECTURE.md
from ..data.csv_data_handler import CSVDataHandler
from ..execution.simulated_execution_handler import SimulatedExecutionHandler
from ..risk.basic_portfolio import BasicPortfolio
from ..risk.basic_risk_manager import BasicRiskManager
from ..strategy.ma_strategy import MAStrategy
from ..strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from ..strategy.regime_detector import RegimeDetector

# Import optimization components
from ..strategy.optimization.basic_optimizer import BasicOptimizer
from ..execution.optimization_entrypoint import OptimizationEntrypoint
from ..strategy.optimization.genetic_optimizer import GeneticOptimizer

# Import dependency graph for validation
from .dependency_graph import DependencyGraph


class RunMode(Enum):
    """System run modes per BOOTSTRAP_SYSTEM.md"""
    PRODUCTION = "production"
    OPTIMIZATION = "optimization"
    BACKTEST = "backtest"
    TEST = "test"


class ComponentState(Enum):
    """Component lifecycle states per COMPONENT_LIFECYCLE.md"""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    DISPOSED = "disposed"


@dataclass
class SystemContext:
    """
    System-wide context available to all components.
    Based on INTERFACE_DESIGN.md context pattern.
    """
    config: SimpleConfigLoader
    container: Container
    event_bus: EventBus
    logger: logging.Logger
    run_mode: RunMode
    metadata: Dict[str, Any]  # For extensions


class ComponentLifecycleTracker:
    """
    Track component lifecycle states and transitions.
    Supports debugging and ensures proper state management.
    """
    
    def __init__(self):
        self.states: Dict[str, ComponentState] = {}
        self.history: List[Tuple[str, ComponentState, ComponentState, float]] = []
        
    def set_state(self, component_name: str, state: ComponentState) -> None:
        """Update component state and record transition."""
        old_state = self.states.get(component_name, ComponentState.CREATED)
        self.states[component_name] = state
        self.history.append((component_name, old_state, state, datetime.now().timestamp()))
        
    def get_state(self, component_name: str) -> ComponentState:
        """Get current component state."""
        return self.states.get(component_name, ComponentState.CREATED)
        
    def get_history(self) -> List[Tuple[str, ComponentState, ComponentState, float]]:
        """Get state transition history."""
        return self.history.copy()
        
    def reset(self) -> None:
        """Reset all tracking."""
        self.states.clear()
        self.history.clear()


class ComponentDiscovery:
    """
    Dynamic component discovery based on DEPENDENCY_MANAGEMENT.md.
    Searches for component_meta.yaml files and loads component definitions.
    """
    
    @staticmethod
    def discover_components(search_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Discover components from the filesystem.
        
        Searches for component_meta.yaml files that define:
        - Component class and module
        - Dependencies
        - Configuration requirements
        - Lifecycle hooks
        
        Returns:
            Dictionary of component definitions keyed by name
        """
        import yaml
        
        discovered = {}
        
        for base_path in search_paths:
            path = Path(base_path)
            if not path.exists():
                continue
                
            # Search for component_meta.yaml files
            for meta_file in path.rglob("component_meta.yaml"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = yaml.safe_load(f)
                        
                    if not metadata or 'components' not in metadata:
                        continue
                        
                    # Process each component definition
                    for comp_name, comp_def in metadata['components'].items():
                        # Validate required fields
                        if 'class' not in comp_def or 'module' not in comp_def:
                            logging.warning(
                                f"Skipping {comp_name} in {meta_file}: "
                                "missing required fields (class, module)"
                            )
                            continue
                            
                        # Store the definition
                        discovered[comp_name] = {
                            'class': comp_def['class'],
                            'module': comp_def['module'],
                            'dependencies': comp_def.get('dependencies', []),
                            'config_key': comp_def.get('config_key'),
                            'required': comp_def.get('required', True),
                            'lifecycle_hooks': comp_def.get('lifecycle_hooks', {}),
                            'metadata_file': str(meta_file)
                        }
                        
                except Exception as e:
                    logging.error(f"Error processing {meta_file}: {e}")
                    
        return discovered


class Bootstrap:
    """
    Main bootstrap class that manages system initialization and lifecycle.
    
    Based on architectural principles from:
    - BOOTSTRAP_SYSTEM.md: Single initialization path
    - COMPONENT_LIFECYCLE.md: Proper lifecycle management  
    - DEPENDENCY_MANAGEMENT.md: Dependency injection
    - STRATEGY_LIFECYCLE_MANAGEMENT.md: Clean state isolation
    """
    
    # Standard component definitions (can be overridden by discovery)
    STANDARD_COMPONENTS = {
        'data_handler': {
            'class': 'CSVDataHandler',
            'module': 'src.data.csv_data_handler',
            'dependencies': ['event_bus'],
            'config_key': 'components.data_handler_csv'
        },
        'execution_handler': {
            'class': 'SimulatedExecutionHandler',
            'module': 'src.execution.simulated_execution_handler',
            'dependencies': ['event_bus', 'portfolio_manager'],
            'config_key': 'components.simulated_execution_handler'
        },
        'portfolio_manager': {  # Note: using portfolio_manager to match existing code
            'class': 'BasicPortfolio',
            'module': 'src.risk.basic_portfolio',
            'dependencies': ['event_bus', 'container'],
            'config_key': 'components.basic_portfolio'
        },
        'risk_manager': {
            'class': 'BasicRiskManager',
            'module': 'src.risk.basic_risk_manager',
            'dependencies': ['event_bus', 'portfolio_manager', 'container'],
            'config_key': 'components.basic_risk_manager'
        },
        'strategy': {
            'class': 'RegimeAdaptiveStrategy',  # Default for production
            'module': 'src.strategy.regime_adaptive_strategy',
            'dependencies': ['event_bus', 'data_handler', 'container'],
            'config_key': 'components.regime_adaptive_strategy'
        },
        'MyPrimaryRegimeDetector': {  # Using exact name from current code
            'class': 'RegimeDetector',
            'module': 'src.strategy.regime_detector',
            'dependencies': ['event_bus'],
            'config_key': 'components.MyPrimaryRegimeDetector',
            'required': False
        },
        'optimizer': {
            'class': 'OptimizationRunner',
            'module': 'src.strategy.optimization.optimization_runner',
            'dependencies': ['event_bus', 'container'],
            'config_key': 'components.optimizer',
            'required': False
        },
        'genetic_optimizer': {
            'class': 'GeneticOptimizer',
            'module': 'src.strategy.optimization.genetic_optimizer',
            'dependencies': ['event_bus', 'container'],
            'config_key': 'components.genetic_optimizer',
            'required': False
        },
        'signal_consumer': {
            'class': 'DummyComponent',
            'module': 'src.core.dummy_component',
            'dependencies': ['event_bus'],
            'config_key': 'components.dummy_service',
            'required': False
        },
        'order_consumer': {
            'class': 'DummyComponent', 
            'module': 'src.core.dummy_component',
            'dependencies': ['event_bus'],
            'config_key': 'components.dummy_service',
            'required': False
        },
        'backtest_runner': {
            'class': 'BacktestRunner',
            'module': 'src.execution.backtest_runner',
            'dependencies': ['event_bus', 'container'],  # Remove data_handler dependency so it starts first
            'config_key': 'components.backtest_runner',
            'required': False
        },
        'optimization_entrypoint': {
            'class': 'OptimizationEntrypoint',
            'module': 'src.execution.optimization_entrypoint',
            'dependencies': ['event_bus', 'container'],
            'config_key': 'components.optimization_entrypoint',
            'required': False
        }
    }
    
    def __init__(self):
        """Initialize bootstrap system."""
        self.context: Optional[SystemContext] = None
        self.components: Dict[str, Any] = {}
        self.initialization_order: List[str] = []
        self.lifecycle_tracker = ComponentLifecycleTracker()
        self.hooks: Dict[str, List[callable]] = {}
        self.dependency_graph = DependencyGraph()
        self.component_definitions = self.STANDARD_COMPONENTS.copy()
        self.search_paths = ["src/strategy", "src/risk", "src/data", "src/execution"]
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.teardown()
        return False
        
    def register_hook(self, hook_name: str, callback: callable) -> None:
        """
        Register a lifecycle hook.
        
        Available hooks:
        - pre_initialize: Before system initialization
        - post_initialize: After system initialization
        - pre_component_create: Before creating each component
        - post_component_create: After creating each component
        - pre_start: Before starting components
        - post_start: After starting components
        - pre_teardown: Before system teardown
        - post_teardown: After system teardown
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
        
    def _run_hooks(self, hook_name: str, *args, **kwargs) -> None:
        """Execute registered hooks."""
        for callback in self.hooks.get(hook_name, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                if self.context:
                    self.context.logger.error(f"Hook {hook_name} failed: {e}")
                    
    def initialize(self, 
                  config: SimpleConfigLoader,
                  run_mode: RunMode = RunMode.PRODUCTION,
                  container: Optional[Container] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> SystemContext:
        """
        Initialize the system context.
        
        This creates the core infrastructure but does not create components.
        Based on BOOTSTRAP_SYSTEM.md initialization sequence.
        
        Args:
            config: System configuration
            run_mode: System run mode (production, optimization, etc.)
            container: Optional existing container (creates new if not provided)
            metadata: Optional metadata for extensions
            
        Returns:
            Initialized system context
        """
        # Run pre-initialization hooks
        self._run_hooks('pre_initialize', config=config, run_mode=run_mode)
        
        # Setup logging first (LOGGING_IMPLEMENTATION.md)
        # For now, set up basic logging until we have proper config loader
        import logging
        log_level = config.get("logging", {}).get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("bootstrap")
        
        # Create container if not provided
        if container is None:
            container = Container()
            
        # Create event bus (EVENT_ARCHITECTURE.md)
        event_bus = EventBus()
        
        # Create system context
        self.context = SystemContext(
            config=config,
            container=container,
            event_bus=event_bus,
            logger=logger,
            run_mode=run_mode,
            metadata=metadata or {}
        )
        
        # Register core services in container
        container.register('config', config)
        container.register('event_bus', event_bus)
        container.register('container', container)  # Register container itself
        container.register('logger', logger)
        container.register('context', self.context)
        
        logger.info(f"System initialized in {run_mode.value} mode")
        
        # Add core services to dependency graph so components can depend on them
        self.dependency_graph.add_component('event_bus', dependencies=[], metadata={'required': True})
        self.dependency_graph.add_component('container', dependencies=[], metadata={'required': True})
        
        # Run post-initialization hooks
        self._run_hooks('post_initialize', context=self.context)
        
        return self.context
        
    def create_component(self, 
                        name: str, 
                        component_def: Dict[str, Any],
                        override_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a single component instance.
        
        Uses minimal constructor following the new lifecycle pattern:
        - Constructor takes only essential args (no external dependencies)
        - Dependencies injected later via initialize()
        
        Args:
            name: Component instance name
            component_def: Component definition with class, module, etc.
            override_config: Optional config overrides
            
        Returns:
            Component instance (not yet initialized)
        """
        if not self.context:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        # Run pre-create hook
        self._run_hooks('pre_component_create', name=name, definition=component_def)
        
        try:
            # Import the module and get the class
            module = importlib.import_module(component_def['module'])
            component_class = getattr(module, component_def['class'])
            
            # Get component configuration
            config_key = component_def.get('config_key')
            component_config = {}
            
            if config_key:
                component_config = self.context.config.get(config_key, {})
                
            # Apply overrides
            if override_config:
                component_config = self._merge_configs(component_config, override_config)
                
            # Create component with minimal constructor
            # Check if it's a new-style component (inherits from ComponentBase)
            from src.core.component_base import ComponentBase
            
            if issubclass(component_class, ComponentBase):
                # New style: minimal constructor
                component = component_class(
                    instance_name=name,
                    config_key=config_key
                )
            else:
                # Legacy style: try minimal args first
                try:
                    component = component_class()
                except TypeError:
                    # Fall back to legacy constructor with event_bus
                    component = component_class(self.context.event_bus)
                    
            # Store component and track state
            self.components[name] = component
            self.lifecycle_tracker.set_state(name, ComponentState.CREATED)
            
            # Store the override config for use during initialization
            if override_config and hasattr(component, '_override_config'):
                component._override_config = override_config
            elif override_config:
                # Store it as an attribute even if the component doesn't expect it
                component._bootstrap_override_config = override_config
            
            # Register in container
            self.context.container.register(name, component)
            
            # Add to dependency graph
            self.dependency_graph.add_component(
                name,
                dependencies=component_def.get('dependencies', []),
                metadata={
                    'required': component_def.get('required', True),
                    'class': component_def['class'],
                    'module': component_def['module']
                }
            )
            
            self.context.logger.debug(f"Created component: {name} ({component_def['class']})")
            
            # Run post-create hook
            self._run_hooks('post_component_create', name=name, component=component)
            
            return component
            
        except Exception as e:
            self.context.logger.error(f"Failed to create component {name}: {e}")
            raise
            
    def initialize_components(self) -> None:
        """
        Initialize all components with their dependencies.
        
        This is separate from creation to support the new lifecycle where:
        1. All components are created first (minimal constructors)
        2. Then all are initialized with dependencies injected
        
        Based on COMPONENT_LIFECYCLE.md initialization phase.
        """
        if not self.context:
            raise RuntimeError("System not initialized")
            
        # Validate dependencies first
        errors = self.validate_dependencies()
        if errors:
            raise RuntimeError(f"Dependency validation failed: {errors}")
            
        # Get initialization order from dependency graph
        self.context.logger.debug(f"Components in graph: {list(self.dependency_graph._nodes.keys())}")
        self.context.logger.debug(f"Components created: {list(self.components.keys())}")
        
        # Add core services as pseudo-components for dependency resolution
        if 'event_bus' not in self.components:
            self.components['event_bus'] = self.context.event_bus
        if 'container' not in self.components:
            self.components['container'] = self.context.container
            
        self.initialization_order = self.dependency_graph.get_initialization_order()
        
        self.context.logger.info(f"Initializing {len(self.initialization_order)} components")
        
        # Initialize each component
        for name in self.initialization_order:
            if name not in self.components:
                # Check if component is required
                metadata = self.dependency_graph.get_component_metadata(name)
                if metadata.get('required', True):
                    raise RuntimeError(f"Required component not created: {name}")
                else:
                    self.context.logger.warning(f"Skipping optional component: {name}")
                    continue
                    
            component = self.components[name]
            
            # Check if component has initialize method
            if hasattr(component, 'initialize'):
                try:
                    # New style: pass context dictionary to initialize
                    init_context = {
                        'config': self.context.config,
                        'container': self.context.container,
                        'event_bus': self.context.event_bus,
                        'logger': self.context.logger,
                        'metadata': self.context.metadata,  # Include metadata with CLI args
                        'bootstrap': self,  # Add reference to Bootstrap for optimization
                        # Add any component-specific dependencies
                        'portfolio_manager': self.components.get('portfolio_manager'),
                        'data_handler': self.components.get('data_handler'),
                        'regime_detector': self.components.get('MyPrimaryRegimeDetector')
                    }
                    component.initialize(init_context)
                    self.context.logger.debug(f"Initialized component: {name}")
                except TypeError:
                    # Legacy style: might need different args
                    self.context.logger.warning(
                        f"Component {name} has non-standard initialize method"
                    )
                    
            # Update lifecycle state
            self.lifecycle_tracker.set_state(name, ComponentState.INITIALIZED)
            
        self.context.logger.info("All components initialized")
        
    def create_standard_components(self, 
                                 component_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create all standard components based on configuration.
        
        Args:
            component_overrides: Override component classes (e.g., {'strategy': 'RegimeAdaptiveStrategy'})
            
        Returns:
            Dictionary of created components
        """
        if not self.context:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        # Apply any overrides to component definitions
        if component_overrides:
            for name, class_name in component_overrides.items():
                if name in self.component_definitions:
                    # Update the class while preserving other settings
                    self.component_definitions[name]['class'] = class_name
                    # Update module path if needed
                    if class_name == 'RegimeAdaptiveStrategy':
                        self.component_definitions[name]['module'] = 'strategy.regime_adaptive_strategy'
                        self.component_definitions[name]['dependencies'] = ['event_bus', 'data_handler', 'regime_detector']
                    elif class_name == 'GeneticOptimizer':
                        self.component_definitions[name]['module'] = 'strategy.optimization.genetic_optimizer'
                        
        # First, try to load components from config file
        config_components = self.context.config.get("components", {})
        if config_components:
            self._load_components_from_config(config_components)
        
        # Then create standard components based on run mode (if not already created)
        components_to_create = self._get_components_for_mode(self.context.run_mode)
        
        for name in components_to_create:
            if name not in self.components and name in self.component_definitions:
                self.create_component(name, self.component_definitions[name])
                
        return self.components
        
    def _load_components_from_config(self, config_components: Dict[str, Any]) -> None:
        """
        Load components directly from the config file.
        
        This allows components to be defined in the config file with their
        configuration inline, rather than requiring separate component definitions.
        """
        # Check if we need to handle strategy switching for verification
        cli_args = self.context.metadata.get('cli_args', {}) if self.context.metadata else {}
        dataset = cli_args.get('dataset')
        
        # If we're in backtest mode with test dataset, check for regime_adaptive_strategy
        if (self.context.run_mode == RunMode.BACKTEST and 
            dataset == 'test' and 
            'regime_adaptive_strategy' in config_components and
            'strategy' in config_components):
            # Use regime_adaptive_strategy instead of regular strategy for verification
            self.context.logger.info("Using RegimeAdaptiveStrategy for test dataset verification")
            # Replace strategy config with regime_adaptive_strategy config
            config_components = config_components.copy()
            config_components['strategy'] = config_components['regime_adaptive_strategy']
            # Remove the regime_adaptive_strategy entry to avoid duplicate creation
            del config_components['regime_adaptive_strategy']
        
        for name, component_config in config_components.items():
            if not isinstance(component_config, dict):
                continue
                
            # Skip if component already created
            if name in self.components:
                continue
                
            # Get class path from config
            class_path = component_config.get('class_path')
            if not class_path:
                self.context.logger.warning(f"Component {name} missing class_path, skipping")
                continue
                
            # Parse module and class from class_path
            parts = class_path.rsplit('.', 1)
            if len(parts) != 2:
                self.context.logger.warning(f"Invalid class_path for {name}: {class_path}")
                continue
                
            module_path, class_name = parts
            
            # Create component definition
            component_def = {
                'class': class_name,
                'module': module_path,
                'dependencies': component_config.get('dependencies', []),
                'config_key': f'components.{name}',
                'required': component_config.get('required', True)
            }
            
            # Store in definitions for future reference
            self.component_definitions[name] = component_def
            
            # Create the component
            try:
                # The actual configuration values are under the 'config' key
                actual_config = component_config.get('config', {})
                self.create_component(name, component_def, override_config=actual_config)
                self.context.logger.debug(f"Created component {name} from config")
            except Exception as e:
                self.context.logger.error(f"Failed to create component {name} from config: {e}")
                if component_def.get('required', True):
                    raise
    
    def _get_components_for_mode(self, run_mode: RunMode) -> List[str]:
        """
        Determine which components to create based on run mode.
        
        Based on BOOTSTRAP_SYSTEM.md run mode configurations.
        """
        # Base components needed for all modes
        base_components = ['data_handler', 'portfolio_manager']  # event_bus is not a component, it's a service
        
        if run_mode == RunMode.PRODUCTION:
            # Production needs everything except optimizer
            return base_components + ['execution_handler', 'risk_manager', 'strategy', 'MyPrimaryRegimeDetector']
            
        elif run_mode == RunMode.OPTIMIZATION:
            # Optimization needs strategy and optimizer, but not execution
            return base_components + ['strategy', 'MyPrimaryRegimeDetector', 'optimizer']
            
        elif run_mode == RunMode.BACKTEST:
            # Backtest needs everything including the backtest runner
            return base_components + ['execution_handler', 'risk_manager', 'strategy', 'MyPrimaryRegimeDetector', 'backtest_runner']
            
        elif run_mode == RunMode.TEST:
            # Test mode - minimal components
            return base_components + ['strategy']
            
        else:
            return base_components
            
    def start_components(self) -> None:
        """
        Start all initialized components.
        
        Components are started in dependency order to ensure
        dependencies are available when needed.
        """
        if not self.context:
            raise RuntimeError("System not initialized")
            
        # Run pre-start hooks
        self._run_hooks('pre_start')
        
        self.context.logger.info("Starting components")
        
        for name in self.initialization_order:
            if name not in self.components:
                continue
                
            component = self.components[name]
            
            # Check component state
            current_state = self.lifecycle_tracker.get_state(name)
            if current_state != ComponentState.INITIALIZED:
                self.context.logger.warning(
                    f"Cannot start {name}: not in INITIALIZED state (current: {current_state})"
                )
                continue
                
            # Start component if it has start method
            if hasattr(component, 'start'):
                try:
                    component.start()
                    self.lifecycle_tracker.set_state(name, ComponentState.RUNNING)
                    self.context.logger.debug(f"Started component: {name}")
                except Exception as e:
                    self.context.logger.error(f"Failed to start {name}: {e}")
                    raise
            else:
                # Component doesn't need explicit start
                self.lifecycle_tracker.set_state(name, ComponentState.RUNNING)
                
        # Run post-start hooks
        self._run_hooks('post_start')
        
        self.context.logger.info("All components started")
        
    def stop_components(self) -> None:
        """
        Stop all running components in reverse dependency order.
        
        This ensures dependent components are stopped before their dependencies.
        """
        if not self.context:
            return
            
        self.context.logger.info("Stopping components")
        
        # Stop in reverse order
        for name in reversed(self.initialization_order):
            if name not in self.components:
                continue
                
            component = self.components[name]
            current_state = self.lifecycle_tracker.get_state(name)
            
            if current_state != ComponentState.RUNNING:
                continue
                
            # Stop component if it has stop method
            if hasattr(component, 'stop'):
                try:
                    component.stop()
                    self.context.logger.debug(f"Stopped component: {name}")
                except Exception as e:
                    self.context.logger.error(f"Error stopping {name}: {e}")
                    
            self.lifecycle_tracker.set_state(name, ComponentState.STOPPED)
            
    def reset_components(self) -> None:
        """
        Reset all components to clean state.
        
        Used between optimization runs to ensure clean state isolation
        per STRATEGY_LIFECYCLE_MANAGEMENT.md.
        """
        if not self.context:
            return
            
        self.context.logger.debug("Resetting components")
        
        for name in self.components:
            component = self.components[name]
            
            # Reset component if it has reset method
            if hasattr(component, 'reset'):
                try:
                    component.reset()
                    self.context.logger.debug(f"Reset component: {name}")
                except Exception as e:
                    self.context.logger.error(f"Error resetting {name}: {e}")
                    
        # Clear event bus subscriptions
        if self.context.event_bus:
            # Clear all subscriptions but keep the bus instance
            for event_type in list(self.context.event_bus._subscribers.keys()):
                self.context.event_bus._subscribers[event_type].clear()
                
    def teardown(self) -> None:
        """
        Perform complete system teardown.
        
        This method ensures proper cleanup in reverse dependency order.
        """
        if not self.context:
            return
            
        self.context.logger.info("Tearing down ADMF system")
        
        # Run pre-teardown hooks
        self._run_hooks('pre_teardown')
        
        # Stop all running components first
        self.stop_components()
        
        # Teardown components in reverse order
        teardown_list = list(reversed(self.initialization_order))
        
        for name in teardown_list:
            if name in self.components:
                component = self.components[name]
                
                # Call component teardown if available
                if hasattr(component, 'teardown'):
                    try:
                        component.teardown()
                        self.context.logger.debug(f"Teardown completed: {name}")
                    except Exception as e:
                        self.context.logger.error(f"Error in teardown of {name}: {e}")
                
                # Update state
                self.lifecycle_tracker.set_state(name, ComponentState.DISPOSED)
        
        # Clear event bus subscriptions (EVENT_ARCHITECTURE.md)
        if self.context.event_bus:
            # Unsubscribe all handlers
            for event_type in list(self.context.event_bus._subscribers.keys()):
                self.context.event_bus._subscribers[event_type].clear()
            
        # Reset container (DEPENDENCY_MANAGEMENT.md)
        if self.context.container:
            self.context.container.reset()
        
        # Clear component tracking
        self.components.clear()
        self.initialization_order.clear()
        
        # Run post-teardown hooks
        self._run_hooks('post_teardown')
        
        # Log lifecycle history if in debug mode
        if self.context.config.get("system", {}).get("debug", False):
            self._log_lifecycle_history()
        
        # Clear context
        self.context = None
        
    def get_component(self, name: str) -> Any:
        """Get a component by name."""
        return self.components.get(name)
        
    def get_context(self) -> Optional[SystemContext]:
        """Get current system context."""
        return self.context
        
    def create_scoped_context(self, 
                            scope_name: str,
                            shared_services: Optional[List[str]] = None) -> 'SystemContext':
        """
        Create a scoped context for isolated execution (e.g., backtest trials).
        
        This method creates a child context that inherits certain services from
        the parent while providing isolation for stateful components.
        
        Args:
            scope_name: Name for this scope (e.g., "trial_1", "backtest_2025-01-15")
            shared_services: List of service names to inherit from parent
                           Defaults to ['config', 'logger'] if not specified
            
        Returns:
            A new SystemContext with a scoped container
            
        Raises:
            RuntimeError: If system not initialized
        """
        if not self.context:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        # Default shared services - typically stateless/system-wide
        if shared_services is None:
            shared_services = ['config', 'logger', 'context']
            
        # Create child container
        scoped_container = Container(parent=self.context.container)
        
        # Create new event bus for complete event isolation
        scoped_event_bus = EventBus()
        
        # Create scoped logger as child of main logger
        scoped_logger = self.context.logger.getChild(scope_name)
        
        # Create scoped context
        scoped_context = SystemContext(
            config=self.context.config,  # Share config
            container=scoped_container,
            event_bus=scoped_event_bus,
            logger=scoped_logger,
            run_mode=self.context.run_mode,
            metadata={
                **self.context.metadata,
                'scope': scope_name,
                'parent_context': self.context,
                'shared_services': shared_services
            }
        )
        
        # Register shared services in scoped container
        # These will override parent lookups for explicit control
        for service_name in shared_services:
            try:
                service = self.context.container.resolve(service_name)
                scoped_container.register_instance(service_name, service)
            except DependencyNotFoundError:
                scoped_logger.warning(f"Shared service '{service_name}' not found in parent")
                
        # Always register the scoped context and event bus
        scoped_container.register_instance('context', scoped_context)
        scoped_container.register_instance('event_bus', scoped_event_bus)
        scoped_container.register_instance('logger', scoped_logger)
        
        scoped_logger.info(f"Created scoped context '{scope_name}' with isolated container and event bus")
        
        return scoped_context
        
    def get_lifecycle_state(self, component_name: str) -> ComponentState:
        """Get current lifecycle state of a component."""
        return self.lifecycle_tracker.get_state(component_name)
        
    def get_entrypoint_component(self) -> Any:
        """
        Get the configured entrypoint component for the current run mode.
        
        Based on the roadmap, this method looks up the entrypoint component
        from the configuration based on the current run mode. The config
        structure expected is:
        
        system:
          run_modes:
            production:
              entrypoint_component: "ProductionCoordinator"
            backtest:
              entrypoint_component: "BacktestRunner"
            optimization:
              entrypoint_component: "GeneticOptimizer"
              
        Returns:
            The entrypoint component instance
            
        Raises:
            RuntimeError: If system not initialized or entrypoint not found
            ValueError: If entrypoint not configured for run mode
        """
        if not self.context:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        # Get run mode configuration
        run_mode_config = self.context.config.get("system", {}).get("run_modes", {})
        current_mode = self.context.run_mode.value
        
        # If no explicit config, use defaults
        if not run_mode_config:
            run_mode_config = {
                'production': {'entrypoint_component': 'data_handler'},
                'backtest': {'entrypoint_component': 'backtest_runner'},
                'optimization': {'entrypoint_component': 'optimizer'},
                'test': {'entrypoint_component': 'strategy'}
            }
        
        if current_mode not in run_mode_config:
            raise ValueError(
                f"No configuration found for run mode '{current_mode}'. "
                f"Available modes: {list(run_mode_config.keys())}"
            )
            
        mode_config = run_mode_config[current_mode]
        
        # Get entrypoint component name
        entrypoint_name = mode_config.get("entrypoint_component")
        
        # If not explicitly configured, use sensible defaults
        if not entrypoint_name:
            default_entrypoints = {
                'production': 'strategy',
                'backtest': 'strategy',
                'optimization': 'optimizer',
                'test': 'strategy'
            }
            entrypoint_name = default_entrypoints.get(current_mode)
            
            if not entrypoint_name:
                raise ValueError(
                    f"No entrypoint_component configured for run mode '{current_mode}' "
                    f"and no default available. Add 'entrypoint_component' to "
                    f"config['system']['run_modes']['{current_mode}']"
                )
            
            self.context.logger.info(
                f"No entrypoint configured for {current_mode}, using default: {entrypoint_name}"
            )
            
        # Get component from container
        try:
            component = self.components.get(entrypoint_name) or self.context.container.resolve(entrypoint_name)
        except KeyError:
            available = list(self.components.keys())
            raise RuntimeError(
                f"Entrypoint component '{entrypoint_name}' not found in container. "
                f"Available components: {available}"
            )
            
        # Verify component is in proper state
        state = self.lifecycle_tracker.get_state(entrypoint_name)
        if state not in [ComponentState.INITIALIZED, ComponentState.RUNNING]:
            raise RuntimeError(
                f"Entrypoint component '{entrypoint_name}' is in {state.value} state. "
                f"Expected INITIALIZED or RUNNING."
            )
            
        self.context.logger.info(
            f"Retrieved entrypoint component '{entrypoint_name}' for {current_mode} mode"
        )
        
        return component
        
    def execute_entrypoint(self, method_name: str = "execute") -> Any:
        """
        Execute the configured entrypoint component.
        
        This is a convenience method that:
        1. Gets the entrypoint component
        2. Verifies it has the specified method
        3. Executes the method and returns the result
        
        Args:
            method_name: Name of the method to call on the entrypoint component
            
        Returns:
            Result from the entrypoint component's method
            
        Raises:
            AttributeError: If component doesn't have the specified method
        """
        component = self.get_entrypoint_component()
        
        # Verify the component has the expected method
        if not hasattr(component, method_name):
            raise AttributeError(
                f"Entrypoint component {component.__class__.__name__} "
                f"does not have method '{method_name}'"
            )
            
        method = getattr(component, method_name)
        if not callable(method):
            raise AttributeError(
                f"'{method_name}' on {component.__class__.__name__} is not callable"
            )
            
        self.context.logger.info(
            f"Executing {method_name}() on entrypoint component {component.__class__.__name__}"
        )
        
        # Execute the method
        try:
            result = method()
            self.context.logger.info(
                f"Entrypoint execution completed successfully"
            )
            return result
        except Exception as e:
            self.context.logger.error(
                f"Error executing entrypoint: {e}", exc_info=True
            )
            raise
        
    def validate_dependencies(self) -> List[str]:
        """
        Validate all component dependencies using the dependency graph.
        
        Based on DEPENDENCY_MANAGEMENT.md validation system with:
        - Circular dependency detection
        - Missing dependency checks
        - Component requirement validation
        
        Returns:
            List of validation errors
        """
        errors = []
        
        if not self.context:
            errors.append("System not initialized")
            return errors
            
        # Check for cycles
        cycles = self.dependency_graph.detect_cycles()
        for cycle in cycles:
            errors.append(f"Circular dependency: {' -> '.join(cycle)}")
            
        # Check for missing dependencies
        for component_name in self.components:
            deps = self.dependency_graph.get_dependencies(component_name)
            for dep in deps:
                # Check if dependency exists in components or container
                # Check if it's registered in container by trying to see if it's in instances or providers
                in_container = (dep in self.context.container._instances or 
                               dep in self.context.container._providers)
                if dep not in self.components and not in_container:
                    # Check if dependency is required
                    dep_metadata = self.dependency_graph.get_component_metadata(dep)
                    if dep_metadata.get('required', True):
                        errors.append(f"Missing required dependency: {component_name} -> {dep}")
                    else:
                        self.context.logger.warning(
                            f"Missing optional dependency: {component_name} -> {dep}"
                        )
                        
        # Validate graph integrity
        graph_errors = self.dependency_graph.validate()
        errors.extend(graph_errors)
                    
        return errors
        
    def setup_managed_components(self, 
                                search_paths: Optional[List[str]] = None,
                                component_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Comprehensive component setup method.
        
        This method handles the complete component setup process:
        1. Discovers component definitions from filesystem
        2. Creates components based on run mode
        3. Initializes all components with dependencies
        4. Returns the created components
        
        Args:
            search_paths: Optional paths to search for component_meta.yaml files
            component_overrides: Optional overrides for component classes
            
        Returns:
            Dictionary of created components
        """
        if not self.context:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        self.context.logger.info("Setting up managed components")
        
        # Step 1: Discover components from filesystem
        if search_paths:
            self.discover_components(search_paths)
        else:
            # Use default search paths
            self.discover_components()
            
        # Step 2: Create components
        components = self.create_standard_components(component_overrides)
        
        # Step 3: Initialize components (dependency injection)
        self.initialize_components()
        
        self.context.logger.info(
            f"Setup complete: {len(components)} components created and initialized"
        )
        
        return components
        
    def discover_components(self, additional_paths: Optional[List[str]] = None) -> None:
        """
        Discover components from the filesystem.
        
        Args:
            additional_paths: Extra paths to search beyond default search_paths
        """
        search_paths = self.search_paths.copy()
        if additional_paths:
            search_paths.extend(additional_paths)
            
        discovered = ComponentDiscovery.discover_components(search_paths)
        
        # Merge with standard components
        self.component_definitions = {**self.STANDARD_COMPONENTS, **discovered}
        
        self.context.logger.info(
            f"Discovered {len(discovered)} components from filesystem, "
            f"total {len(self.component_definitions)} component definitions available"
        )
        
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configurations."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def _log_lifecycle_history(self) -> None:
        """Log component lifecycle history for debugging."""
        history = self.lifecycle_tracker.get_history()
        if not history:
            return
            
        self.context.logger.debug("Component Lifecycle History:")
        self.context.logger.debug("-" * 80)
        
        for component, old_state, new_state, timestamp in history:
            self.context.logger.debug(
                f"{timestamp:.3f}: {component} - {old_state.value} -> {new_state.value}"
            )
            
        self.context.logger.debug("-" * 80)


# Example usage of the new bootstrap system:
"""
Example usage of the Bootstrap system with proper lifecycle management:

# 1. Basic usage with manual steps:
bootstrap = Bootstrap()
context = bootstrap.initialize(
    config=config,
    run_mode=RunMode.PRODUCTION,
    container=container  # Optional: provide existing container
)
bootstrap.discover_components(["custom/components"])
components = bootstrap.create_standard_components()
bootstrap.initialize_components()
bootstrap.start_components()
# Run your application logic
bootstrap.teardown()

# 2. Simplified usage with setup_managed_components:
bootstrap = Bootstrap()
context = bootstrap.initialize(config=config, run_mode=RunMode.OPTIMIZATION)
components = bootstrap.setup_managed_components(search_paths=["./src"])
bootstrap.start_components()
result = bootstrap.execute_entrypoint()  # Runs the configured entrypoint
bootstrap.teardown()

# 3. Using context manager with entrypoint:
with Bootstrap() as bootstrap:
    context = bootstrap.initialize(config=config, run_mode=RunMode.BACKTEST)
    bootstrap.setup_managed_components()
    bootstrap.start_components()
    result = bootstrap.execute_entrypoint()
# Automatic cleanup on exit

# 4. Example main.py pattern:
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--mode', choices=['production', 'backtest', 'optimization', 'test'])
    args = parser.parse_args()
    
    config = Config(args.config)
    run_mode = RunMode(args.mode)
    
    with Bootstrap() as bootstrap:
        bootstrap.initialize(config=config, run_mode=run_mode)
        bootstrap.setup_managed_components()
        bootstrap.start_components()
        
        try:
            result = bootstrap.execute_entrypoint()
            print(f"Execution completed: {result}")
        except Exception as e:
            print(f"Execution failed: {e}")
            raise
"""