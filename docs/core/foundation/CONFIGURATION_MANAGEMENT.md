# Configuration Management

## Overview

This document defines the configuration management system for the ADMF-Trader platform. It outlines the architecture for managing environment-specific configurations, schema validation, dynamic updates, and secure credential handling.

## Motivation

Effective configuration management is critical for:

1. **Environment Flexibility**: Seamless transition between development, testing, and production
2. **Security**: Safe handling of credentials and sensitive configuration values
3. **Validation**: Ensuring configuration values are valid before system startup
4. **Runtime Adaptability**: Supporting configuration changes without system restarts
5. **Auditability**: Tracking configuration changes for debugging and compliance

## Architecture

### 1. Configuration Structure

The configuration system uses a hierarchical structure with inheritance:

```
Configuration
├── Defaults
│   └── Base configuration values
├── Environment-specific
│   ├── Development
│   ├── Testing
│   └── Production
└── Instance-specific
    └── Override values for specific deployment
```

### 2. Configuration Provider

The `ConfigurationProvider` manages configuration loading and access:

```python
from typing import Dict, Any, Optional, List
import os
import yaml
import json
from pathlib import Path

class ConfigurationProvider:
    """
    Configuration provider for ADMF-Trader.
    
    Manages loading, validating, and accessing configuration.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize configuration provider.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or os.environ.get("ADMF_CONFIG_DIR", "./config")
        self.env = os.environ.get("ADMF_ENV", "development")
        self.config = {}
        self.validators = {}
        self._loaded = False
        self._schema = None
        
    def load(self) -> bool:
        """
        Load configuration.
        
        Returns:
            bool: Success status
        """
        if self._loaded:
            return True
            
        # Load in sequence:
        # 1. Defaults
        # 2. Environment-specific
        # 3. Instance-specific
        # 4. Environment variables
        
        # Load defaults
        default_path = Path(self.config_dir) / "defaults.yaml"
        if default_path.exists():
            with open(default_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
                
        # Load environment-specific
        env_path = Path(self.config_dir) / f"{self.env}.yaml"
        if env_path.exists():
            with open(env_path, "r") as f:
                env_config = yaml.safe_load(f) or {}
                self._deep_update(self.config, env_config)
                
        # Load instance-specific if exists
        instance_path = Path(self.config_dir) / "instance.yaml"
        if instance_path.exists():
            with open(instance_path, "r") as f:
                instance_config = yaml.safe_load(f) or {}
                self._deep_update(self.config, instance_config)
                
        # Load from environment variables
        self._load_from_env()
        
        # Load schema for validation
        schema_path = Path(self.config_dir) / "schema.yaml"
        if schema_path.exists():
            with open(schema_path, "r") as f:
                self._schema = yaml.safe_load(f)
                
        # Validate configuration
        if self._schema:
            validation_result = self._validate_schema()
            if not validation_result[0]:
                raise ValueError(f"Configuration validation failed: {validation_result[1]}")
                
        self._loaded = True
        return True
        
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            path: Configuration path (dot notation)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if not self._loaded:
            self.load()
            
        # Split path into parts
        parts = path.split(".")
        
        # Traverse configuration
        value = self.config
        for part in parts:
            if not isinstance(value, dict):
                return default
                
            if part not in value:
                return default
                
            value = value[part]
            
        return value
        
    def get_section(self, path: str) -> Dict[str, Any]:
        """
        Get configuration section.
        
        Args:
            path: Section path (dot notation)
            
        Returns:
            Dict containing section
        """
        value = self.get(path, {})
        if not isinstance(value, dict):
            return {}
            
        return value
        
    def set(self, path: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            path: Configuration path (dot notation)
            value: Value to set
            
        Returns:
            bool: Success status
        """
        if not self._loaded:
            self.load()
            
        # Split path into parts
        parts = path.split(".")
        
        # Traverse configuration
        config = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
                
            if not isinstance(config[part], dict):
                # Can't set nested path in non-dict
                return False
                
            config = config[part]
            
        # Set value
        config[parts[-1]] = value
        
        # Validate after change
        if self._schema:
            validation_result = self._validate_schema()
            if not validation_result[0]:
                # Revert change
                self.load()
                return False
                
        return True
        
    def register_validator(self, path: str, validator: callable) -> None:
        """
        Register custom validator for configuration path.
        
        Args:
            path: Configuration path (dot notation)
            validator: Validation function
        """
        self.validators[path] = validator
        
    def reload(self) -> bool:
        """
        Reload configuration.
        
        Returns:
            bool: Success status
        """
        self._loaded = False
        return self.load()
        
    def save(self) -> bool:
        """
        Save current configuration to instance file.
        
        Returns:
            bool: Success status
        """
        instance_path = Path(self.config_dir) / "instance.yaml"
        
        try:
            # Create directory if not exists
            instance_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration
            with open(instance_path, "w") as f:
                yaml.dump(self.config, f)
                
            return True
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
            return False
            
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep update dictionary.
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
                
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith("ADMF_CONFIG_"):
                # Convert to path
                path = key[12:].lower().replace("__", ".").replace("_", ".")
                
                # Try to parse as JSON
                try:
                    value = json.loads(value)
                except:
                    # Use as string
                    pass
                    
                # Set value
                self.set(path, value)
                
    def _validate_schema(self) -> tuple:
        """
        Validate configuration against schema.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            from jsonschema import validate, ValidationError
            
            # Validate against schema
            validate(self.config, self._schema)
            
            # Run custom validators
            for path, validator in self.validators.items():
                value = self.get(path)
                if not validator(value):
                    return False, f"Custom validation failed for {path}"
                    
            return True, None
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Schema validation error: {str(e)}"
```

### 3. Secure Credential Management

The `CredentialManager` handles sensitive configuration values:

```python
import os
import base64
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

class CredentialManager:
    """
    Secure credential management.
    
    Manages encryption and access to sensitive configuration values.
    """
    
    def __init__(self, config_provider):
        """
        Initialize credential manager.
        
        Args:
            config_provider: Configuration provider
        """
        self.config = config_provider
        self._key = None
        self._init_key()
        
    def _init_key(self):
        """Initialize encryption key."""
        # Get master key from environment or configuration
        master_key = os.environ.get("ADMF_MASTER_KEY")
        if not master_key:
            # Use configuration value
            master_key = self.config.get("security.master_key")
            
        if not master_key:
            # Generate new key
            master_key = Fernet.generate_key().decode()
            
            # Save to configuration
            self.config.set("security.master_key", master_key)
            
        # Derive encryption key
        salt = self.config.get("security.salt", "ADMF-Trader").encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self._key = Fernet(key)
        
    def encrypt(self, value):
        """
        Encrypt value.
        
        Args:
            value: Value to encrypt
            
        Returns:
            str: Encrypted value
        """
        if not value:
            return None
            
        # Convert to JSON string
        value_str = json.dumps(value)
        
        # Encrypt
        encrypted = self._key.encrypt(value_str.encode())
        
        # Encode as base64
        return base64.urlsafe_b64encode(encrypted).decode()
        
    def decrypt(self, encrypted_value):
        """
        Decrypt value.
        
        Args:
            encrypted_value: Encrypted value
            
        Returns:
            Decrypted value
        """
        if not encrypted_value:
            return None
            
        try:
            # Decode from base64
            encrypted = base64.urlsafe_b64decode(encrypted_value)
            
            # Decrypt
            decrypted = self._key.decrypt(encrypted).decode()
            
            # Parse JSON
            return json.loads(decrypted)
        except Exception as e:
            print(f"Error decrypting value: {str(e)}")
            return None
            
    def get_credential(self, path, default=None):
        """
        Get credential value.
        
        Args:
            path: Configuration path
            default: Default value if not found
            
        Returns:
            Credential value
        """
        # Get encrypted value
        encrypted = self.config.get(path)
        if not encrypted:
            return default
            
        # Decrypt
        return self.decrypt(encrypted) or default
        
    def set_credential(self, path, value):
        """
        Set credential value.
        
        Args:
            path: Configuration path
            value: Value to set
            
        Returns:
            bool: Success status
        """
        # Encrypt value
        encrypted = self.encrypt(value)
        
        # Set in configuration
        return self.config.set(path, encrypted)
```

### 4. Dynamic Configuration Updates

The `ConfigurationWatcher` enables runtime configuration updates:

```python
import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigurationWatcher(FileSystemEventHandler):
    """
    Watch for configuration file changes.
    
    Enables dynamic configuration updates.
    """
    
    def __init__(self, config_provider, callback=None):
        """
        Initialize configuration watcher.
        
        Args:
            config_provider: Configuration provider
            callback: Function to call on configuration change
        """
        self.config = config_provider
        self.callback = callback
        self.observer = Observer()
        self._running = False
        self._last_update = 0
        
    def start(self):
        """Start watching for changes."""
        if self._running:
            return
            
        # Start observer
        self.observer.schedule(self, self.config.config_dir, recursive=False)
        self.observer.start()
        self._running = True
        
    def stop(self):
        """Stop watching for changes."""
        if not self._running:
            return
            
        # Stop observer
        self.observer.stop()
        self.observer.join()
        self._running = False
        
    def on_modified(self, event):
        """
        Handle file modification.
        
        Args:
            event: File system event
        """
        # Debounce updates
        current_time = time.time()
        if current_time - self._last_update < 1.0:
            return
            
        self._last_update = current_time
        
        # Check if configuration file
        file_path = event.src_path
        filename = os.path.basename(file_path)
        
        if filename in ["defaults.yaml", f"{self.config.env}.yaml", "instance.yaml"]:
            # Reload configuration
            self.config.reload()
            
            # Call callback
            if self.callback:
                self.callback()
```

### 5. Configuration Schema

The schema definition for configuration validation:

```yaml
# schema.yaml
type: object
required: [system, execution, components]
properties:
  system:
    type: object
    required: [name]
    properties:
      name:
        type: string
      version:
        type: string
        
  execution:
    type: object
    required: [mode]
    properties:
      mode:
        type: string
        enum: [BACKTEST_SINGLE, BACKTEST_PARALLEL, OPTIMIZATION, LIVE_TRADING, PAPER_TRADING, REPLAY]
      thread_model:
        type: string
        enum: [SINGLE_THREADED, MULTI_THREADED, PROCESS_PARALLEL, ASYNC_SINGLE, ASYNC_MULTI, MIXED]
      thread_pools:
        type: object
        
  components:
    type: object
    properties:
      data_handler:
        type: object
        required: [class]
      strategy:
        type: object
        required: [class]
      risk_manager:
        type: object
      portfolio:
        type: object
      broker:
        type: object
        
  security:
    type: object
    properties:
      master_key:
        type: string
      salt:
        type: string
```

## Environment-Specific Configuration

### 1. Development Configuration

```yaml
# development.yaml
system:
  name: ADMF-Trader-Dev
  version: 0.1.0
  
execution:
  mode: BACKTEST_SINGLE
  thread_model: SINGLE_THREADED
  
logging:
  level: DEBUG
  console_output: true
  file_output: false
  
components:
  data_handler:
    class: HistoricalDataHandler
    thread_safe: false
  strategy:
    class: MovingAverageCrossover
    thread_safe: false
  broker:
    type: simulated
    simulation:
      enabled: true
      slippage: 0.001
```

### 2. Testing Configuration

```yaml
# testing.yaml
system:
  name: ADMF-Trader-Test
  version: 0.1.0
  
execution:
  mode: BACKTEST_PARALLEL
  thread_model: MULTI_THREADED
  thread_pools:
    data:
      max_workers: 2
    compute:
      max_workers: 2
  
logging:
  level: INFO
  console_output: true
  file_output: true
  file_path: ./logs/test.log
  
components:
  data_handler:
    class: ParallelDataHandler
    thread_safe: true
  strategy:
    class: MovingAverageCrossover
    thread_safe: true
  broker:
    type: simulated
    simulation:
      enabled: true
      slippage: 0.001
```

### 3. Production Configuration

```yaml
# production.yaml
system:
  name: ADMF-Trader-Prod
  version: 0.1.0
  
execution:
  mode: LIVE_TRADING
  thread_model: ASYNC_MULTI
  thread_pools:
    market_data:
      max_workers: 2
    order_processing:
      max_workers: 2
  
logging:
  level: WARNING
  console_output: false
  file_output: true
  file_path: /var/log/admf-trader/trading.log
  
components:
  data_handler:
    class: LiveMarketDataHandler
    thread_safe: true
  strategy:
    class: MovingAverageCrossover
    thread_safe: true
  broker:
    type: live
    api:
      endpoint: ${BROKER_API_ENDPOINT}
```

## Configuration Injection

### 1. Component Configuration

Components receive their configuration at initialization:

```python
class Component:
    """Base component with configuration support."""
    
    def __init__(self, name, config=None):
        """
        Initialize component.
        
        Args:
            name: Component name
            config: Component configuration
        """
        self.name = name
        self.config = config or {}
        
    def get_config(self, key, default=None):
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
        
    def set_config(self, key, value):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
```

### 2. System Bootstrap

The system uses configuration during bootstrap:

```python
def bootstrap_system(config_path=None, env=None):
    """
    Bootstrap system with configuration.
    
    Args:
        config_path: Path to configuration directory
        env: Environment name
        
    Returns:
        Initialized system
    """
    # Override environment if specified
    if env:
        os.environ["ADMF_ENV"] = env
        
    # Override config directory if specified
    if config_path:
        os.environ["ADMF_CONFIG_DIR"] = config_path
        
    # Create configuration provider
    config_provider = ConfigurationProvider()
    
    # Load configuration
    config_provider.load()
    
    # Create credential manager
    credential_manager = CredentialManager(config_provider)
    
    # Create context
    context = ExecutionContext(
        name=config_provider.get("system.name"),
        execution_mode=ExecutionMode[config_provider.get("execution.mode", "BACKTEST_SINGLE")],
        thread_model=ThreadModel[config_provider.get("execution.thread_model")] if config_provider.get("execution.thread_model") else None,
        config=config_provider
    )
    
    # Create container
    container = Container()
    
    # Register components
    container.register_instance("config", config_provider)
    container.register_instance("credentials", credential_manager)
    container.register_instance("context", context)
    
    # Register component factories
    component_configs = config_provider.get_section("components")
    for name, component_config in component_configs.items():
        register_component(container, name, component_config)
        
    # Start configuration watcher if in live mode
    if context.is_live:
        def on_config_change():
            # Handle configuration change
            pass
            
        watcher = ConfigurationWatcher(config_provider, on_config_change)
        watcher.start()
        container.register_instance("config_watcher", watcher)
        
    return container, context
```

## Dynamic Updates

### 1. Component Reconfiguration

Components can handle runtime configuration changes:

```python
class ReconfigurableComponent(Component):
    """Component that supports runtime reconfiguration."""
    
    def __init__(self, name, config=None):
        """Initialize reconfigurable component."""
        super().__init__(name, config)
        self.dynamic_config_keys = []
        
    def register_dynamic_config(self, key, handler=None):
        """
        Register dynamic configuration key.
        
        Args:
            key: Configuration key
            handler: Function to call when key changes
        """
        self.dynamic_config_keys.append((key, handler))
        
    def update_config(self, new_config):
        """
        Update configuration.
        
        Args:
            new_config: New configuration
            
        Returns:
            bool: Whether restart is required
        """
        restart_required = False
        
        for key, value in new_config.items():
            # Check if dynamic key
            dynamic_handlers = [h for k, h in self.dynamic_config_keys if k == key]
            
            if dynamic_handlers:
                # Dynamic key, can update at runtime
                old_value = self.config.get(key)
                
                # Update config
                self.config[key] = value
                
                # Call handlers
                for handler in dynamic_handlers:
                    if handler:
                        handler(key, old_value, value)
            else:
                # Non-dynamic key, requires restart
                self.config[key] = value
                restart_required = True
                
        return restart_required
```

### 2. System Reconfiguration

The system can handle configuration changes:

```python
def handle_config_change(container, changes):
    """
    Handle configuration changes.
    
    Args:
        container: Component container
        changes: Configuration changes
    """
    restart_required = False
    
    # Group changes by component
    component_changes = {}
    for path, old_value, new_value in changes:
        if path.startswith("components."):
            # Extract component name
            parts = path.split(".")
            if len(parts) >= 2:
                component_name = parts[1]
                config_key = ".".join(parts[2:])
                
                if component_name not in component_changes:
                    component_changes[component_name] = {}
                    
                component_changes[component_name][config_key] = new_value
                
    # Update components
    for component_name, config_changes in component_changes.items():
        if container.has(component_name):
            component = container.get(component_name)
            
            # Check if component supports reconfiguration
            if hasattr(component, "update_config"):
                # Update component configuration
                component_restart = component.update_config(config_changes)
                restart_required = restart_required or component_restart
                
    return restart_required
```

## Security Considerations

### 1. Sensitive Value Handling

Guidelines for handling sensitive configuration values:

1. **Never store credentials in plaintext** in configuration files
2. **Use environment variables** for sensitive values in production
3. **Use credential manager** for encrypting sensitive values
4. **Limit access** to configuration files containing sensitive values
5. **Redact sensitive values** in logs and error messages

### 2. Configuration Validation

Guidelines for secure configuration validation:

1. **Validate all values** before use
2. **Apply strict schema** to configuration
3. **Verify permissions** for security-related configuration
4. **Check for insecure defaults**
5. **Validate credentials** before connecting to external services

## Implementation Strategy

### 1. Core Implementation

1. Implement `ConfigurationProvider` class
2. Create schema validation mechanism
3. Implement environment-specific loading

### 2. Credential Management

1. Implement `CredentialManager` class
2. Create secure value storage
3. Implement encryption/decryption

### 3. Dynamic Updates

1. Implement `ConfigurationWatcher` class
2. Create change detection mechanism
3. Implement component update notification

### 4. Environment Configuration

1. Create default configuration templates
2. Implement environment-specific overrides
3. Create secure production configuration template

## Best Practices

### 1. Configuration Design

- Use hierarchical structure for configuration
- Group related settings by component
- Provide sensible defaults for all settings
- Document all configuration options
- Use consistent naming conventions

### 2. Environment Management

- Create distinct configurations for each environment
- Keep environment-specific details in separate files
- Use environment variables for deployment-specific values
- Test configuration in each environment
- Automate configuration deployment

### 3. Credential Management

- Use credential manager for all sensitive values
- Store credentials separate from regular configuration
- Use different credential sets for each environment
- Rotate credentials regularly
- Implement access control for credential files

## Conclusion

The configuration management system provides a robust foundation for configuring and operating the ADMF-Trader system across different environments. By supporting environment-specific configuration, schema validation, dynamic updates, and secure credential handling, the system ensures consistent operation while maintaining flexibility and security.

Key benefits include:

1. **Environment Consistency**: Reliable operation across development, testing, and production
2. **Security**: Secure handling of sensitive configuration and credentials
3. **Validation**: Early detection of configuration errors
4. **Adaptability**: Dynamic updates without system restarts
5. **Maintainability**: Clear separation of configuration from code

The implementation approach prioritizes security, flexibility, and ease of use, ensuring that the ADMF-Trader system can be effectively configured and operated in various environments.