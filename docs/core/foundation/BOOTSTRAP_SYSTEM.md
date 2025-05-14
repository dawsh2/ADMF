# Bootstrap System

## Overview

The Bootstrap System is responsible for initializing and orchestrating the entire ADMF-Trader application. It manages the loading of configurations, creates the dependency injection container, registers components, and controls the startup sequence. This system ensures that all components are properly configured, initialized, and connected before the application begins operation.

## Key Responsibilities

1. **Configuration Loading**: Load and validate configuration files
2. **Component Registration**: Discover and register components
3. **Dependency Injection**: Set up the DI container and resolve dependencies
4. **Startup Sequence**: Control the initialization and startup sequence
5. **Error Handling**: Handle errors during startup
6. **Extension Points**: Provide hooks for customization

## Bootstrap Process

The bootstrap process follows these steps:

1. **Load Configuration**: Load configuration from files and environment variables
2. **Create Container**: Create the dependency injection container
3. **Register Core Components**: Register essential system components
4. **Discover and Register Additional Components**: Find and register optional components
5. **Initialize Components**: Initialize components in dependency order
6. **Start Components**: Start components in the correct sequence
7. **Run Application**: Hand control to the application

## Implementation

```python
class Bootstrap:
    """Bootstraps the ADMF-Trader system."""
    
    def __init__(self, config_files=None, debug=False, log_level=None, log_file=None):
        """
        Initialize bootstrap system.
        
        Args:
            config_files: List of configuration files to load
            debug: Whether to enable debug mode
            log_level: Log level to use
            log_file: Log file path
        """
        self.config_files = config_files or ["config.yaml"]
        self.debug = debug
        self.log_level = log_level
        self.log_file = log_file
        self.hooks = {}
        self.container = None
        self.config = None
        
    def setup(self):
        """
        Set up the system.
        
        Returns:
            tuple: (container, config)
        """
        # Configure logging first
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Create container
        self.container = self._create_container()
        
        # Register core components
        self._register_core_components()
        
        # Discover and register additional components
        self._discover_components()
        
        # Run pre-initialization hooks
        self._run_hooks("pre_initialize")
        
        # Initialize components
        self._initialize_components()
        
        # Run post-initialization hooks
        self._run_hooks("post_initialize")
        
        return self.container, self.config
        
    def register_hook(self, hook_point, callback):
        """
        Register a hook callback.
        
        Args:
            hook_point: Hook point name
            callback: Callback function
        """
        if hook_point not in self.hooks:
            self.hooks[hook_point] = []
        self.hooks[hook_point].append(callback)
        
    def _run_hooks(self, hook_point):
        """
        Run hooks for a hook point.
        
        Args:
            hook_point: Hook point name
        """
        for callback in self.hooks.get(hook_point, []):
            callback(self.container, self.config)
            
    def _setup_logging(self):
        """Set up logging configuration."""
        import logging
        
        # Configure root logger
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_level = self.log_level or (logging.DEBUG if self.debug else logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=self.log_file
        )
        
        # Create console handler if no file specified
        if not self.log_file:
            console = logging.StreamHandler()
            console.setLevel(log_level)
            formatter = logging.Formatter(log_format)
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)
            
        # Create logger for bootstrap
        self.logger = logging.getLogger('bootstrap')
        self.logger.info("Logging initialized")
        
    def _load_configuration(self):
        """
        Load configuration from files.
        
        Returns:
            Config: Configuration object
        """
        from core.config import Config
        
        config = Config()
        
        # Load each config file
        for config_file in self.config_files:
            try:
                config.load(config_file)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_file}: {e}")
                if self.debug:
                    raise
                    
        # Load environment variables
        config.load_env(prefix="ADMF_")
        
        # Apply debug mode if enabled
        if self.debug:
            config.set('debug', True)
            
        return config
        
    def _create_container(self):
        """
        Create dependency injection container.
        
        Returns:
            Container: DI container
        """
        from core.container import Container
        
        container = Container()
        
        # Register config instance
        container.register_instance('config', self.config)
        
        # Register bootstrap instance
        container.register_instance('bootstrap', self)
        
        return container
        
    def _register_core_components(self):
        """Register core system components."""
        # Register event bus
        from core.events import EventBus
        self.container.register('event_bus', EventBus)
        
        # Register logger factory
        def logger_factory(name):
            import logging
            return logging.getLogger(name)
        self.container.register_factory('logger', logger_factory)
        
        # Register state verifier if in debug mode
        if self.debug:
            from core.state import StateVerifier
            self.container.register('state_verifier', StateVerifier)
            
    def _discover_components(self):
        """Discover and register components."""
        # Get component discovery mode
        discovery_mode = self.config.get('component_discovery.mode', 'auto')
        
        if discovery_mode == 'auto':
            # Auto-discover components
            self._auto_discover_components()
        elif discovery_mode == 'config':
            # Register components from configuration
            self._register_configured_components()
        else:
            self.logger.warning(f"Unknown component discovery mode: {discovery_mode}")
            
    def _auto_discover_components(self):
        """Auto-discover components using introspection."""
        import importlib
        import pkgutil
        import inspect
        
        # Get component base class
        from core.component import Component
        
        # Get modules to scan
        modules_to_scan = ['data', 'strategy', 'risk', 'execution']
        
        # Scan modules for components
        for module_name in modules_to_scan:
            try:
                # Import module
                module = importlib.import_module(f'src.{module_name}')
                
                # Scan module for component classes
                for _, name, _ in pkgutil.iter_modules(module.__path__):
                    try:
                        # Import submodule
                        submodule = importlib.import_module(f'src.{module_name}.{name}')
                        
                        # Find component classes
                        for attr_name in dir(submodule):
                            attr = getattr(submodule, attr_name)
                            
                            # Check if it's a component class
                            if (
                                inspect.isclass(attr) and 
                                issubclass(attr, Component) and 
                                attr != Component
                            ):
                                # Register component class
                                component_name = f"{module_name}.{name}.{attr_name}"
                                self.container.register(component_name, attr)
                                self.logger.debug(f"Discovered component: {component_name}")
                    except Exception as e:
                        self.logger.warning(f"Error scanning {module_name}.{name}: {e}")
            except Exception as e:
                self.logger.warning(f"Error scanning module {module_name}: {e}")
                
    def _register_configured_components(self):
        """Register components from configuration."""
        import importlib
        
        # Get component registrations from config
        component_config = self.config.get('components', {})
        
        # Register each component
        for name, config in component_config.items():
            try:
                # Get component class
                module_path = config.get('module')
                class_name = config.get('class')
                
                if not module_path or not class_name:
                    self.logger.warning(f"Invalid component config for {name}: missing module or class")
                    continue
                    
                # Import module
                module = importlib.import_module(module_path)
                
                # Get class
                component_class = getattr(module, class_name)
                
                # Register component
                singleton = config.get('singleton', True)
                self.container.register(name, component_class, singleton=singleton)
                self.logger.debug(f"Registered configured component: {name}")
            except Exception as e:
                self.logger.warning(f"Error registering component {name}: {e}")
                
    def _initialize_components(self):
        """Initialize components in dependency order."""
        # Get component initialization order
        init_order = self._get_initialization_order()
        
        # Create initialization context
        context = {
            'event_bus': self.container.get('event_bus'),
            'config': self.config,
            'logger': self.container.get_factory('logger')('system')
        }
        
        if self.debug:
            context['state_verifier'] = self.container.get('state_verifier')
            
        # Initialize components
        for component_name in init_order:
            try:
                component = self.container.get(component_name)
                if hasattr(component, 'initialize') and not getattr(component, 'initialized', False):
                    component.initialize(context)
                    self.logger.debug(f"Initialized component: {component_name}")
            except Exception as e:
                self.logger.error(f"Error initializing component {component_name}: {e}")
                if self.debug:
                    raise
                    
    def _get_initialization_order(self):
        """
        Get component initialization order.
        
        Returns:
            list: Component names in initialization order
        """
        # Get configured initialization order
        configured_order = self.config.get('component_initialization.order', [])
        
        if configured_order:
            return configured_order
            
        # Default order (core components first, then by module)
        default_order = [
            'event_bus',
            'data_handler',
            'strategy',
            'risk_manager',
            'broker',
            'portfolio',
            'analytics'
        ]
        
        # Filter to only include registered components
        return [name for name in default_order if self.container.has(name)]
        
    def teardown(self):
        """Clean up resources."""
        if not self.container:
            return
            
        # Get components in reverse initialization order
        component_names = self._get_initialization_order()
        component_names.reverse()
        
        # Teardown components
        for component_name in component_names:
            try:
                component = self.container.get(component_name)
                if hasattr(component, 'teardown'):
                    component.teardown()
                    self.logger.debug(f"Teardown component: {component_name}")
            except Exception as e:
                self.logger.warning(f"Error in teardown of component {component_name}: {e}")
                
        # Reset container
        self.container.reset()
        self.logger.info("System teardown complete")
```

## Main Entry Point

The bootstrap system is used from the main entry point:

```python
def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ADMF-Trader')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--mode', choices=['backtest', 'optimize', 'live'], default='backtest', help='Operation mode')
    args = parser.parse_args()
    
    try:
        # Create bootstrap instance
        bootstrap = Bootstrap(
            config_files=[args.config],
            debug=args.debug,
            log_level=logging.DEBUG if args.verbose else None,
            log_file=args.log_file
        )
        
        # Set up system
        container, config = bootstrap.setup()
        
        # Run in specified mode
        if args.mode == 'backtest':
            run_backtest(container, config)
        elif args.mode == 'optimize':
            run_optimization(container, config)
        elif args.mode == 'live':
            run_live_trading(container, config)
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
            
        # Clean up
        bootstrap.teardown()
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
```

## Extension through Hooks

The bootstrap system provides hooks for extending or customizing the startup process:

```python
# Register a pre-initialization hook
bootstrap.register_hook("pre_initialize", lambda container, config: print("Before initialization"))

# Register a post-initialization hook
bootstrap.register_hook("post_initialize", lambda container, config: print("After initialization"))
```

## Customizing Component Discovery

The component discovery process can be customized through configuration:

```yaml
# Auto-discover components
component_discovery:
  mode: auto

# Or use configuration-based discovery
component_discovery:
  mode: config

# Component configurations
components:
  data_handler:
    module: src.data.historical
    class: HistoricalDataHandler
    singleton: true
    
  strategy:
    module: src.strategy.ma_crossover
    class: MACrossoverStrategy
    singleton: true
```

## Customizing Initialization Order

The initialization order can be customized through configuration:

```yaml
component_initialization:
  order:
    - event_bus
    - logger
    - data_handler
    - strategy
    - risk_manager
    - broker
    - portfolio
```

## Error Handling

The bootstrap system includes comprehensive error handling:

1. **Configuration Errors**: Problems with configuration files are reported
2. **Dependency Errors**: Missing or circular dependencies are detected
3. **Initialization Errors**: Problems during component initialization are handled
4. **Graceful Degradation**: The system can continue operation with missing non-essential components
5. **Debug Mode**: In debug mode, errors cause immediate termination with full stack traces

## Best Practices

1. **Configuration-Driven**: Use configuration to control component behavior
2. **Minimal Main**: Keep the main entry point simple and delegate to the bootstrap system
3. **Clear Dependency Order**: Ensure component dependencies are well-defined
4. **Graceful Startup and Shutdown**: Handle errors during startup and ensure clean shutdown
5. **Testability**: Design for easy testing by allowing component mocking

By following these patterns, the bootstrap system provides a robust, extensible foundation for starting up and shutting down the ADMF-Trader application.