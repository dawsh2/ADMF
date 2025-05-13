# Strategy Lifecycle Management & Parameter Versioning

## 1. Overview

This document outlines the comprehensive approach to managing the full lifecycle of trading strategies in the ADMF-Trader system, with a focus on parameter optimization, versioning, monitoring, and deployment. The framework ensures long-term viability, reproducibility, and performance sustainability beyond initial strategy development.

## 2. Motivation

To ensure the long-term viability, profitability, and operational integrity of the ADMF-Trader system and its deployed strategies, it's crucial to move beyond initial development and backtesting. Strategies degrade, market conditions shift, and operational complexities grow. A proactive and systematic approach to managing the entire lifecycle of trading strategies is therefore essential to prevent performance drift, maintain reproducibility, enable data-driven decisions, and ensure the system remains robust and auditable.

## 3. Core Components

### 3.1 Parameter Versioning System

```
ParameterVersioningSystem
  ├── ParameterStore
  │   ├── VersionedParameterSet
  │   ├── ParameterMetadata
  │   └── VersionHistory
  ├── ParameterRepository
  │   ├── StorageAdapter
  │   ├── QueryService
  │   └── VersionManager
  └── ParameterResolver
      ├── ConfigurationIntegration
      └── ReferenceResolver
```

#### 3.1.1 VersionedParameterSet

The core data structure for storing strategy parameters:

```python
class VersionedParameterSet:
    """Immutable, versioned set of strategy parameters."""
    
    def __init__(self, strategy_id, parameters, version=None, metadata=None):
        """
        Initialize a versioned parameter set.
        
        Args:
            strategy_id (str): Identifier of the strategy
            parameters (dict): Strategy parameters
            version (str, optional): Version identifier (auto-generated if None)
            metadata (dict, optional): Parameter metadata
        """
        self.strategy_id = strategy_id
        self.parameters = dict(parameters)  # Make a copy to ensure immutability
        self.version = version or self._generate_version()
        self.metadata = metadata or {}
        self.creation_timestamp = datetime.datetime.now()
        
    def _generate_version(self):
        """Generate a unique version identifier."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{self.strategy_id}_{timestamp}"
        
    def get_parameter(self, name, default=None):
        """Get a parameter value."""
        return self.parameters.get(name, default)
        
    def get_all_parameters(self):
        """Get a copy of all parameters."""
        return dict(self.parameters)
        
    def get_metadata(self):
        """Get parameter metadata."""
        return dict(self.metadata)
        
    def with_updated_metadata(self, metadata_updates):
        """Create a new instance with updated metadata."""
        new_metadata = {**self.metadata, **metadata_updates}
        return VersionedParameterSet(
            self.strategy_id,
            self.parameters,
            self.version,
            new_metadata
        )
```

#### 3.1.2 ParameterMetadata

Standard metadata fields for parameter sets:

```python
class ParameterMetadata:
    """Standard metadata for parameter sets."""
    
    # Strategy information
    STRATEGY_VERSION = "strategy_version"
    STRATEGY_CODE_HASH = "strategy_code_hash"
    
    # Optimization information
    OPTIMIZATION_DATE = "optimization_date"
    OPTIMIZATION_OBJECTIVE = "optimization_objective"
    OPTIMIZATION_METHOD = "optimization_method"
    
    # Data information
    TRAIN_DATA_START = "train_data_start"
    TRAIN_DATA_END = "train_data_end"
    TEST_DATA_START = "test_data_start"
    TEST_DATA_END = "test_data_end"
    DATA_RESOLUTION = "data_resolution"
    SYMBOLS = "symbols"
    
    # Performance metrics
    TRAIN_SHARPE_RATIO = "train_sharpe_ratio"
    TRAIN_RETURNS = "train_returns"
    TRAIN_DRAWDOWN = "train_drawdown"
    TEST_SHARPE_RATIO = "test_sharpe_ratio"
    TEST_RETURNS = "test_returns"
    TEST_DRAWDOWN = "test_drawdown"
    
    # Deployment information
    DEPLOYMENT_STATUS = "deployment_status"
    DEPLOYMENT_DATE = "deployment_date"
    APPROVED_BY = "approved_by"
    
    # Lifecycle information
    ACTIVE = "active"
    RETIRED_DATE = "retired_date"
    RETIREMENT_REASON = "retirement_reason"
```

#### 3.1.3 ParameterRepository

Service for storing and retrieving parameter sets:

```python
class ParameterRepository:
    """Repository for versioned parameter sets."""
    
    def __init__(self, storage_adapter):
        """
        Initialize repository.
        
        Args:
            storage_adapter: Storage backend adapter
        """
        self.storage = storage_adapter
        
    def save(self, parameter_set):
        """
        Save a parameter set.
        
        Args:
            parameter_set: VersionedParameterSet to save
            
        Returns:
            str: Version identifier
        """
        return self.storage.save(parameter_set)
        
    def get_by_version(self, strategy_id, version):
        """
        Get parameter set by version.
        
        Args:
            strategy_id: Strategy identifier
            version: Version identifier
            
        Returns:
            VersionedParameterSet: The parameter set
        """
        return self.storage.get_by_version(strategy_id, version)
        
    def get_latest(self, strategy_id, filter_metadata=None):
        """
        Get latest parameter set, optionally filtered by metadata.
        
        Args:
            strategy_id: Strategy identifier
            filter_metadata: Optional metadata filter
            
        Returns:
            VersionedParameterSet: The latest parameter set
        """
        return self.storage.get_latest(strategy_id, filter_metadata)
        
    def get_version_history(self, strategy_id):
        """
        Get version history for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            list: Version history information
        """
        return self.storage.get_version_history(strategy_id)
        
    def update_metadata(self, strategy_id, version, metadata_updates):
        """
        Update metadata for a parameter set.
        
        Args:
            strategy_id: Strategy identifier
            version: Version identifier
            metadata_updates: Metadata updates to apply
            
        Returns:
            bool: Success status
        """
        return self.storage.update_metadata(strategy_id, version, metadata_updates)
```

### 3.2 Configuration Management

Integration with the system configuration to resolve parameter references:

```python
class ParameterResolver:
    """Resolves parameter references in configurations."""
    
    def __init__(self, parameter_repository):
        """
        Initialize resolver.
        
        Args:
            parameter_repository: ParameterRepository instance
        """
        self.repository = parameter_repository
        
    def resolve_references(self, config):
        """
        Resolve parameter references in a configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            dict: Configuration with resolved parameters
            
        Raises:
            ValueError: If a reference cannot be resolved
        """
        if 'strategy' not in config:
            return config
            
        strategy_config = config['strategy']
        
        if 'parameters_ref' in strategy_config:
            param_ref = strategy_config['parameters_ref']
            strategy_id = strategy_config['id']
            
            # Parse reference format (strategy_id@version or strategy_id@latest)
            if '@' in param_ref:
                ref_parts = param_ref.split('@')
                if len(ref_parts) != 2:
                    raise ValueError(f"Invalid parameter reference format: {param_ref}")
                    
                ref_strategy_id, version = ref_parts
                
                if ref_strategy_id != strategy_id:
                    raise ValueError(
                        f"Parameter reference strategy ID ({ref_strategy_id}) "
                        f"does not match strategy ID ({strategy_id})"
                    )
                    
                if version.lower() == 'latest':
                    param_set = self.repository.get_latest(strategy_id)
                else:
                    param_set = self.repository.get_by_version(strategy_id, version)
            else:
                # Simple reference format (version only)
                param_set = self.repository.get_by_version(strategy_id, param_ref)
                
            if param_set is None:
                raise ValueError(f"Could not resolve parameter reference: {param_ref}")
                
            # Replace parameters_ref with actual parameters
            strategy_config.pop('parameters_ref')
            strategy_config['parameters'] = param_set.get_all_parameters()
            
            # Add metadata about the parameters used
            if 'metadata' not in strategy_config:
                strategy_config['metadata'] = {}
                
            strategy_config['metadata']['parameters_version'] = param_set.version
            
        return config
```

### 3.3 Optimization Workflow

The systematic process for optimizing, validating, and deploying strategies:

```python
class OptimizationWorkflow:
    """Workflow for strategy optimization and validation."""
    
    def __init__(self, parameter_repository, optimizer, validator):
        """
        Initialize workflow.
        
        Args:
            parameter_repository: ParameterRepository instance
            optimizer: Strategy optimizer
            validator: Strategy validator
        """
        self.repository = parameter_repository
        self.optimizer = optimizer
        self.validator = validator
        
    def run_optimization(self, strategy_id, config, train_data, test_data):
        """
        Run optimization workflow.
        
        Args:
            strategy_id: Strategy identifier
            config: Optimization configuration
            train_data: Training data
            test_data: Test data
            
        Returns:
            dict: Optimization results
        """
        # Step 1: Run optimization on training data
        optimization_results = self.optimizer.optimize(
            strategy_id=strategy_id,
            config=config,
            data=train_data
        )
        
        # Step 2: Extract best parameters
        best_parameters = optimization_results['best_parameters']
        train_metrics = optimization_results['metrics']
        
        # Step 3: Validate on test data
        validation_results = self.validator.validate(
            strategy_id=strategy_id,
            parameters=best_parameters,
            data=test_data
        )
        
        test_metrics = validation_results['metrics']
        
        # Step 4: Create and store parameter set with metadata
        metadata = {
            ParameterMetadata.STRATEGY_VERSION: config.get('strategy_version', '1.0.0'),
            ParameterMetadata.OPTIMIZATION_DATE: datetime.datetime.now().isoformat(),
            ParameterMetadata.OPTIMIZATION_OBJECTIVE: config.get('objective', 'sharpe_ratio'),
            ParameterMetadata.OPTIMIZATION_METHOD: config.get('method', 'grid_search'),
            ParameterMetadata.TRAIN_DATA_START: train_data.get_start_date().isoformat(),
            ParameterMetadata.TRAIN_DATA_END: train_data.get_end_date().isoformat(),
            ParameterMetadata.TEST_DATA_START: test_data.get_start_date().isoformat(),
            ParameterMetadata.TEST_DATA_END: test_data.get_end_date().isoformat(),
            ParameterMetadata.SYMBOLS: ','.join(train_data.get_symbols()),
            ParameterMetadata.TRAIN_SHARPE_RATIO: train_metrics.get('sharpe_ratio', 0),
            ParameterMetadata.TRAIN_RETURNS: train_metrics.get('returns', 0),
            ParameterMetadata.TRAIN_DRAWDOWN: train_metrics.get('max_drawdown', 0),
            ParameterMetadata.TEST_SHARPE_RATIO: test_metrics.get('sharpe_ratio', 0),
            ParameterMetadata.TEST_RETURNS: test_metrics.get('returns', 0),
            ParameterMetadata.TEST_DRAWDOWN: test_metrics.get('max_drawdown', 0),
            ParameterMetadata.DEPLOYMENT_STATUS: 'pending_approval'
        }
        
        parameter_set = VersionedParameterSet(
            strategy_id=strategy_id,
            parameters=best_parameters,
            metadata=metadata
        )
        
        # Step 5: Save parameter set
        version = self.repository.save(parameter_set)
        
        # Step 6: Return results
        return {
            'version': version,
            'parameters': best_parameters,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'validation_passed': validation_results['passed']
        }
```

### 3.4 Performance Monitoring

Continuous monitoring of strategy performance against benchmarks:

```python
class PerformanceMonitor:
    """Monitors strategy performance against benchmarks."""
    
    def __init__(self, parameter_repository, metrics_service, alert_service):
        """
        Initialize monitor.
        
        Args:
            parameter_repository: ParameterRepository instance
            metrics_service: Service for calculating performance metrics
            alert_service: Service for sending alerts
        """
        self.repository = parameter_repository
        self.metrics_service = metrics_service
        self.alert_service = alert_service
        
    def monitor_strategy(self, strategy_id, version, window_days=30):
        """
        Monitor strategy performance.
        
        Args:
            strategy_id: Strategy identifier
            version: Parameter version in use
            window_days: Monitoring window in days
            
        Returns:
            dict: Monitoring results
        """
        # Get parameter set
        param_set = self.repository.get_by_version(strategy_id, version)
        if param_set is None:
            raise ValueError(f"Invalid parameter version: {version}")
            
        # Get benchmark metrics from metadata
        benchmark_metrics = {
            'sharpe_ratio': param_set.metadata.get(ParameterMetadata.TEST_SHARPE_RATIO, 0),
            'returns': param_set.metadata.get(ParameterMetadata.TEST_RETURNS, 0),
            'drawdown': param_set.metadata.get(ParameterMetadata.TEST_DRAWDOWN, 0)
        }
        
        # Get live metrics for monitoring window
        live_metrics = self.metrics_service.calculate_metrics(
            strategy_id=strategy_id,
            window_days=window_days
        )
        
        # Calculate deviation from benchmark
        deviations = {}
        for metric, benchmark_value in benchmark_metrics.items():
            if metric in live_metrics and benchmark_value != 0:
                live_value = live_metrics[metric]
                deviation = (live_value - benchmark_value) / abs(benchmark_value)
                deviations[metric] = deviation
        
        # Check for significant deviations
        alerts = []
        for metric, deviation in deviations.items():
            threshold = self._get_threshold(metric)
            if abs(deviation) > threshold:
                alert = {
                    'strategy_id': strategy_id,
                    'version': version,
                    'metric': metric,
                    'benchmark_value': benchmark_metrics[metric],
                    'live_value': live_metrics[metric],
                    'deviation': deviation,
                    'threshold': threshold,
                    'severity': 'high' if abs(deviation) > threshold * 2 else 'medium'
                }
                alerts.append(alert)
                
                # Send alert
                self.alert_service.send_alert(alert)
        
        # Return monitoring results
        return {
            'benchmark_metrics': benchmark_metrics,
            'live_metrics': live_metrics,
            'deviations': deviations,
            'alerts': alerts,
            'status': 'alert' if alerts else 'normal'
        }
        
    def _get_threshold(self, metric):
        """Get deviation threshold for a metric."""
        thresholds = {
            'sharpe_ratio': 0.3,
            'returns': 0.3,
            'drawdown': 0.5
        }
        return thresholds.get(metric, 0.3)
```

### 3.5 Deployment and Rollback System

Controlled process for deploying parameter updates and rolling back if needed:

```python
class DeploymentManager:
    """Manages strategy deployment and rollback."""
    
    def __init__(self, parameter_repository, system_manager, change_logger):
        """
        Initialize manager.
        
        Args:
            parameter_repository: ParameterRepository instance
            system_manager: Trading system manager
            change_logger: Change log service
        """
        self.repository = parameter_repository
        self.system_manager = system_manager
        self.change_logger = change_logger
        
    def deploy_parameters(self, strategy_id, version, approver=None):
        """
        Deploy a parameter version.
        
        Args:
            strategy_id: Strategy identifier
            version: Parameter version to deploy
            approver: Person approving the deployment
            
        Returns:
            bool: Success status
        """
        # Get parameter set
        param_set = self.repository.get_by_version(strategy_id, version)
        if param_set is None:
            raise ValueError(f"Invalid parameter version: {version}")
            
        # Record current version for potential rollback
        current_version = self._get_current_version(strategy_id)
        
        # Update deployment metadata
        metadata_updates = {
            ParameterMetadata.DEPLOYMENT_STATUS: 'deployed',
            ParameterMetadata.DEPLOYMENT_DATE: datetime.datetime.now().isoformat()
        }
        if approver:
            metadata_updates[ParameterMetadata.APPROVED_BY] = approver
            
        self.repository.update_metadata(strategy_id, version, metadata_updates)
        
        # Update system configuration
        success = self.system_manager.update_strategy_parameters(
            strategy_id=strategy_id,
            parameters=param_set.get_all_parameters()
        )
        
        # Log the change
        self.change_logger.log_change(
            component="strategy",
            action="parameter_deployment",
            details={
                "strategy_id": strategy_id,
                "version": version,
                "previous_version": current_version,
                "approved_by": approver,
                "success": success
            }
        )
        
        return success
        
    def rollback(self, strategy_id, target_version=None):
        """
        Rollback to a previous version.
        
        Args:
            strategy_id: Strategy identifier
            target_version: Target version (or previous deployed if None)
            
        Returns:
            bool: Success status
        """
        if target_version is None:
            # Find previous deployed version
            history = self.repository.get_version_history(strategy_id)
            deployed_versions = [
                v for v in history 
                if v.get('metadata', {}).get(ParameterMetadata.DEPLOYMENT_STATUS) == 'deployed'
            ]
            
            # Sort by deployment date (descending)
            deployed_versions.sort(
                key=lambda v: v.get('metadata', {}).get(ParameterMetadata.DEPLOYMENT_DATE, ''),
                reverse=True
            )
            
            # Get previous version (second most recent)
            if len(deployed_versions) < 2:
                raise ValueError("No previous version available for rollback")
                
            target_version = deployed_versions[1]['version']
        
        # Deploy the target version
        return self.deploy_parameters(
            strategy_id=strategy_id,
            version=target_version,
            approver="SYSTEM_ROLLBACK"
        )
        
    def _get_current_version(self, strategy_id):
        """Get current deployed version for a strategy."""
        # Get latest with deployed status
        param_set = self.repository.get_latest(
            strategy_id=strategy_id,
            filter_metadata={ParameterMetadata.DEPLOYMENT_STATUS: 'deployed'}
        )
        
        return param_set.version if param_set else None
```

### 3.6 Change Management Logging

Comprehensive logging of all strategy-related changes:

```python
class ChangeLogger:
    """Logs strategy and parameter changes."""
    
    def __init__(self, storage_adapter):
        """
        Initialize logger.
        
        Args:
            storage_adapter: Storage backend adapter
        """
        self.storage = storage_adapter
        
    def log_change(self, component, action, details):
        """
        Log a change.
        
        Args:
            component: System component (e.g., "strategy", "parameters")
            action: Change action (e.g., "deployment", "optimization")
            details: Change details
            
        Returns:
            str: Change log ID
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "component": component,
            "action": action,
            "user": self._get_current_user(),
            "details": details
        }
        
        return self.storage.save_log(log_entry)
        
    def get_changes(self, filters=None, limit=100):
        """
        Get change logs.
        
        Args:
            filters: Optional filters
            limit: Maximum number of logs to return
            
        Returns:
            list: Change logs
        """
        return self.storage.get_logs(filters, limit)
        
    def get_strategy_changes(self, strategy_id, limit=100):
        """
        Get changes for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            limit: Maximum number of logs to return
            
        Returns:
            list: Change logs
        """
        filters = {
            "details.strategy_id": strategy_id
        }
        return self.get_changes(filters, limit)
        
    def _get_current_user(self):
        """Get current user information."""
        # Implementation depends on authentication system
        return "system"  # Default
```

## 4. Implementation Guidelines

### 4.1 Parameter Versioning

- Use immutable parameter sets to prevent unintended changes
- Include comprehensive metadata for tracking and context
- Implement a robust storage mechanism (database preferred)
- Use unique, meaningful version identifiers

### 4.2 Configuration Integration

- Configuration should reference parameter versions, not contain parameters directly
- Strict validation of parameter references during configuration loading
- Clear error messages for invalid or missing references
- Support for both specific versions and "latest" references

### 4.3 Optimization Workflow

- Clearly separate training and testing datasets
- Always validate on out-of-sample data
- Consider walk-forward optimization for time series data
- Document optimization objectives and constraints

### 4.4 Performance Monitoring

- Establish clear thresholds for performance deviations
- Implement multiple notification channels for alerts
- Consider contextual monitoring that accounts for market regimes
- Track correlation between strategy performance and market conditions

### 4.5 Deployment Process

- Implement pre-deployment checklist validation
- Require explicit approval for production deployments
- Support gradual deployment with percentage-based allocation
- Maintain history of all deployments and rollbacks

## 5. Integration with System Architecture

The Strategy Lifecycle Management framework integrates with other architectural components:

### 5.1 Core Component Integration

- **Interface-Based Design**: Parameter versioning leverages the interface-based design defined in [INTERFACE_DESIGN.md](/Users/daws/ADMF/docs/core/INTERFACE_DESIGN.md)
- **State Reset**: Deployment and rollback utilize the state reset verification in the system
- **Thread Safety**: Performance monitoring considers the thread safety model of the system

### 5.2 Risk Management Integration

- **Risk Limits**: Parameter updates maintain awareness of risk limit configurations
- **Position Tracking**: Performance monitoring integrates with position tracking for accurate metrics
- **Risk Validation**: Parameter changes undergo risk validation before deployment

### 5.3 Validation Framework Integration

- **Configuration Validation**: Parameter references are validated during configuration loading
- **Consistency Checks**: System validates consistency between parameters and strategy code version
- **Data Validation**: Training and testing datasets undergo validation checks

## 6. Implementation Plan

### 6.1 Phase 1: Core Parameter Versioning

1. Implement parameter data structures and repository
2. Create storage backend (database or file-based)
3. Implement basic metadata tracking
4. Integrate with configuration system

### 6.2 Phase 2: Optimization and Validation Workflow

1. Develop optimization workflow framework
2. Implement walk-forward testing capabilities
3. Create parameter validation system
4. Build optimization report generation

### 6.3 Phase 3: Deployment and Monitoring

1. Implement deployment management system
2. Create performance monitoring framework
3. Develop alerting and notification system
4. Build rollback capabilities

### 6.4 Phase 4: Change Management and Reporting

1. Implement change logging system
2. Create historical performance visualization
3. Develop strategy lifecycle dashboards
4. Build audit and compliance reporting

## 7. Benefits

1. **Reproducibility**: Every strategy version can be reproduced exactly
2. **Traceability**: Full history of parameter changes and performance
3. **Reliability**: Controlled deployment and rollback capabilities
4. **Adaptability**: Systematic re-optimization and validation process
5. **Early Warning**: Performance monitoring identifies strategy degradation
6. **Auditability**: Comprehensive change logs for all strategy modifications

## 8. Conclusion

The Strategy Lifecycle Management framework provides a comprehensive approach to managing trading strategies through their entire lifecycle. By implementing rigorous parameter versioning, validation, deployment, and monitoring practices, the ADMF-Trader system can maintain long-term performance sustainability and operational integrity.

This framework addresses the critical challenge of strategy decay over time by establishing a systematic process for continuous improvement, validation, and monitoring. It bridges the gap between initial strategy development and long-term production operation, providing the foundation for a robust, adaptable trading system.