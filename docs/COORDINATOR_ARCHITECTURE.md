# ADMF-Trader: Coordinator Architecture for Protocol + Composition System

## Table of Contents

1. [Overview](#1-overview)
2. [Core Coordinator Pattern](#2-core-coordinator-pattern)
3. [Optimization Workflow Coordinator](#3-optimization-workflow-coordinator)
4. [Phased Execution Framework](#4-phased-execution-framework)
5. [Container-Based Isolation](#5-container-based-isolation)
6. [Feature Mapping from Current System](#6-feature-mapping-from-current-system)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Overview

The Coordinator pattern serves as the orchestration layer in the new Protocol + Composition architecture, managing complex workflows while keeping individual components simple and focused. It leverages the Universal Scoped Container architecture for isolation and the Protocol-based design for maximum flexibility.

### Key Principles

1. **Workflow Orchestration**: Coordinators manage the flow of execution through multiple components
2. **Container Isolation**: Each execution context runs in an isolated container
3. **Protocol-Based Communication**: Components interact through well-defined protocols
4. **Configuration-Driven**: All behavior is specified through configuration
5. **Phased Execution**: Support for stopping at any phase for manual intervention

### Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│                  User Interface                      │
│        (CLI, API, or Future GUI)                    │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│              System Coordinator                      │
│   (Routes requests to appropriate coordinators)     │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│          Specialized Coordinators                    │
│  ┌───────────────┐ ┌────────────┐ ┌──────────────┐ │
│  │  Optimization │ │  Backtest  │ │ Live Trading │ │
│  │  Coordinator  │ │Coordinator │ │ Coordinator  │ │
│  └───────────────┘ └────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│         Container Orchestration Layer                │
│  (Creates and manages isolated containers)           │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│              Component Layer                         │
│  (Strategies, Indicators, Risk Managers, etc.)      │
└─────────────────────────────────────────────────────┘
```

---

## 2. Core Coordinator Pattern

### 2.1 Base Coordinator Protocol

```python
from typing import Protocol, Dict, Any, List, Optional
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

class ExecutionPhase(Enum):
    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    PROCESSING = "processing"
    ANALYSIS = "analysis"
    FINALIZATION = "finalization"

@dataclass
class PhaseResult:
    phase: ExecutionPhase
    success: bool
    data: Dict[str, Any]
    message: str
    next_phase: Optional[ExecutionPhase] = None

@runtime_checkable
class Coordinator(Protocol):
    """Base protocol for all coordinators"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize coordinator with configuration"""
        ...
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the coordinated workflow"""
        ...
    
    @abstractmethod
    def get_current_phase(self) -> ExecutionPhase:
        """Get current execution phase"""
        ...
    
    @abstractmethod
    def can_resume(self) -> bool:
        """Check if execution can be resumed"""
        ...

@runtime_checkable
class PhasedCoordinator(Protocol):
    """Coordinator that supports phased execution"""
    
    @abstractmethod
    def execute_phase(self, phase: ExecutionPhase) -> PhaseResult:
        """Execute a specific phase"""
        ...
    
    @abstractmethod
    def get_phase_dependencies(self, phase: ExecutionPhase) -> List[ExecutionPhase]:
        """Get phases that must complete before this phase"""
        ...
    
    @abstractmethod
    def save_phase_state(self, phase: ExecutionPhase, state: Dict[str, Any]) -> None:
        """Save state after phase completion"""
        ...
    
    @abstractmethod
    def load_phase_state(self, phase: ExecutionPhase) -> Optional[Dict[str, Any]]:
        """Load saved state for a phase"""
        ...
```

### 2.2 System Coordinator

```python
class SystemCoordinator:
    """Main system coordinator that routes to specialized coordinators"""
    
    def __init__(self, config_path: str):
        self.config = Config.load(config_path)
        self.container_factory = ContainerFactory(SharedServicesProvider())
        self.coordinators: Dict[str, Coordinator] = {}
        self._initialize_coordinators()
    
    def _initialize_coordinators(self) -> None:
        """Initialize all available coordinators"""
        # Create coordinators based on configuration
        if 'optimization' in self.config:
            self.coordinators['optimization'] = OptimizationCoordinator(
                self.config['optimization'],
                self.container_factory
            )
        
        if 'backtest' in self.config:
            self.coordinators['backtest'] = BacktestCoordinator(
                self.config['backtest'],
                self.container_factory
            )
        
        if 'live_trading' in self.config:
            self.coordinators['live_trading'] = LiveTradingCoordinator(
                self.config['live_trading'],
                self.container_factory
            )
    
    def execute(self, mode: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific mode"""
        if mode not in self.coordinators:
            raise ValueError(f"Unknown execution mode: {mode}")
        
        coordinator = self.coordinators[mode]
        
        # Handle phased execution if requested
        if kwargs.get('phased', False) and isinstance(coordinator, PhasedCoordinator):
            return self._execute_phased(coordinator, **kwargs)
        else:
            return coordinator.execute(kwargs)
    
    def _execute_phased(self, coordinator: PhasedCoordinator, **kwargs) -> Dict[str, Any]:
        """Execute coordinator in phased mode"""
        stop_after = kwargs.get('stop_after_phase', None)
        resume_from = kwargs.get('resume_from_phase', None)
        
        phases = self._get_execution_order(coordinator)
        results = {}
        
        # Find starting phase
        start_idx = 0
        if resume_from:
            start_idx = phases.index(resume_from)
        
        # Execute phases
        for phase in phases[start_idx:]:
            print(f"Executing phase: {phase.value}")
            result = coordinator.execute_phase(phase)
            results[phase.value] = result
            
            if not result.success:
                print(f"Phase {phase.value} failed: {result.message}")
                break
            
            if stop_after == phase.value:
                print(f"Stopping after phase {phase.value} as requested")
                coordinator.save_phase_state(phase, result.data)
                break
        
        return results
```

---

## 3. Optimization Workflow Coordinator

### 3.1 Optimization Coordinator Implementation

```python
class OptimizationCoordinator:
    """Coordinator for complex optimization workflows"""
    
    def __init__(self, config: Dict[str, Any], container_factory: ContainerFactory):
        self.config = config
        self.container_factory = container_factory
        self.lifecycle_manager = ContainerLifecycleManager()
        self.phase_states: Dict[str, Dict[str, Any]] = {}
        self.current_phase = ExecutionPhase.INITIALIZATION
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete optimization workflow"""
        workflow_type = self.config['workflow']['type']
        
        if workflow_type == 'sequential':
            return self._execute_sequential_workflow(context)
        elif workflow_type == 'parallel':
            return self._execute_parallel_workflow(context)
        elif workflow_type == 'adaptive':
            return self._execute_adaptive_workflow(context)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    def execute_phase(self, phase: ExecutionPhase) -> PhaseResult:
        """Execute a specific optimization phase"""
        if phase == ExecutionPhase.DATA_PREPARATION:
            return self._prepare_data()
        elif phase == ExecutionPhase.PROCESSING:
            return self._run_optimization()
        elif phase == ExecutionPhase.ANALYSIS:
            return self._analyze_results()
        elif phase == ExecutionPhase.FINALIZATION:
            return self._finalize_results()
        else:
            return PhaseResult(phase, False, {}, f"Unknown phase: {phase}")
    
    def _execute_sequential_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sequential optimization workflow"""
        results = {
            'workflow_type': 'sequential',
            'steps': [],
            'final_parameters': {}
        }
        
        steps = self.config['workflow']['steps']
        
        for step in steps:
            step_result = self._execute_optimization_step(step, context)
            results['steps'].append(step_result)
            
            # Update context with results for next steps
            if 'parameters' in step_result:
                context['previous_parameters'] = step_result['parameters']
        
        # Aggregate final results
        results['final_parameters'] = self._aggregate_optimization_results(results['steps'])
        
        return results
    
    def _execute_optimization_step(self, step_config: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single optimization step in isolated container"""
        step_name = step_config['name']
        component_name = step_config['component']
        method = step_config['method']
        
        # Create optimization container
        container_spec = {
            'strategy_class': component_name,
            'strategy_params': step_config.get('params', {}),
            'optimization_method': method,
            'evaluator': step_config.get('evaluator', 'standard')
        }
        
        container_id = self.lifecycle_manager.create_and_start_container(
            "optimization", 
            container_spec
        )
        
        try:
            container = self.lifecycle_manager.active_containers[container_id]
            
            # Run optimization
            if method == 'grid_search':
                result = self._run_grid_search(container, step_config)
            elif method == 'ensemble_weights':
                result = self._run_weight_optimization(container, step_config, context)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            result['step_name'] = step_name
            result['container_id'] = container_id
            
            return result
            
        finally:
            # Always cleanup container
            self.lifecycle_manager.stop_and_destroy_container(container_id)
    
    def _run_grid_search(self, container: UniversalScopedContainer, 
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Run grid search optimization in container"""
        # Get components from container
        strategy = container.resolve("strategy")
        evaluator = container.resolve("evaluator")
        optimizer = container.resolve("optimizer")
        
        # Get parameter space
        parameter_space = strategy.get_parameter_space()
        
        # Run optimization
        results = optimizer.optimize(
            objective_func=lambda params: evaluator.evaluate(strategy, params),
            parameter_space=parameter_space,
            n_trials=config.get('max_trials', 100)
        )
        
        # Analyze by regime if applicable
        if hasattr(evaluator, 'get_regime_analysis'):
            results['regime_analysis'] = evaluator.get_regime_analysis()
        
        return results
    
    def _prepare_data(self) -> PhaseResult:
        """Phase 1: Data Mining/Preparation"""
        print("Phase 1: Preparing data for optimization...")
        
        # Create data preparation container
        container_id = self.lifecycle_manager.create_and_start_container(
            "optimization",
            {"phase": "data_preparation"}
        )
        
        try:
            container = self.lifecycle_manager.active_containers[container_id]
            data_handler = container.resolve("data_handler")
            
            # Load and prepare data
            symbols = self.config.get('symbols', ['EURUSD'])
            data_handler.load_data(symbols)
            
            # Set up train/test split
            split_config = self.config.get('data_split', {'method': 'ratio', 'train_ratio': 0.7})
            data_handler.setup_split(**split_config)
            
            # Prepare regime detection if configured
            regime_data = {}
            if 'regime_detection' in self.config:
                regime_detector = container.resolve("regime_detector")
                regime_data = regime_detector.prepare_regime_data()
            
            result_data = {
                'symbols': symbols,
                'data_points': data_handler.get_data_statistics(),
                'split_info': split_config,
                'regime_data': regime_data
            }
            
            return PhaseResult(
                phase=ExecutionPhase.DATA_PREPARATION,
                success=True,
                data=result_data,
                message="Data preparation complete",
                next_phase=ExecutionPhase.PROCESSING
            )
            
        finally:
            self.lifecycle_manager.stop_and_destroy_container(container_id)
    
    def _analyze_results(self) -> PhaseResult:
        """Phase 3: Analysis (can be manual or automated)"""
        print("Phase 3: Analyzing optimization results...")
        
        # Load results from previous phase
        processing_results = self.load_phase_state(ExecutionPhase.PROCESSING)
        if not processing_results:
            return PhaseResult(
                phase=ExecutionPhase.ANALYSIS,
                success=False,
                data={},
                message="No processing results found"
            )
        
        # Perform automated analysis
        analysis = {
            'summary_statistics': self._calculate_summary_stats(processing_results),
            'regime_performance': self._analyze_regime_performance(processing_results),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(processing_results),
            'recommendations': self._generate_recommendations(processing_results)
        }
        
        # Save analysis results
        self._save_analysis_report(analysis)
        
        return PhaseResult(
            phase=ExecutionPhase.ANALYSIS,
            success=True,
            data=analysis,
            message="Analysis complete. Review results in analysis_report.json",
            next_phase=ExecutionPhase.FINALIZATION
        )
```

### 3.2 Three-Phase Optimization Workflow

```python
class ThreePhaseOptimizationCoordinator(OptimizationCoordinator):
    """Specialized coordinator for three-phase optimization workflow"""
    
    def __init__(self, config: Dict[str, Any], container_factory: ContainerFactory):
        super().__init__(config, container_factory)
        self.phases = [
            ExecutionPhase.DATA_PREPARATION,  # Phase 1: Data Mining
            ExecutionPhase.ANALYSIS,          # Phase 2: Manual Analysis
            ExecutionPhase.FINALIZATION       # Phase 3: OOS Testing
        ]
    
    def execute_data_mining_phase(self) -> PhaseResult:
        """Phase 1: Data Mining - Generate optimization data"""
        print("=" * 60)
        print("PHASE 1: DATA MINING")
        print("=" * 60)
        
        results = {
            'isolated_optimizations': {},
            'regime_analysis': {},
            'ensemble_candidates': []
        }
        
        # Step 1: Isolated rule optimization
        print("\nStep 1: Optimizing individual rules...")
        for rule_config in self.config['rules']:
            rule_name = rule_config['name']
            print(f"\nOptimizing {rule_name}...")
            
            rule_results = self._optimize_isolated_rule(rule_config)
            results['isolated_optimizations'][rule_name] = rule_results
            
            # Extract regime-specific best parameters
            regime_params = self._extract_regime_parameters(rule_results)
            results['regime_analysis'][rule_name] = regime_params
        
        # Step 2: Generate ensemble weight candidates
        print("\nStep 2: Testing ensemble weight combinations...")
        weight_results = self._test_weight_combinations()
        results['ensemble_candidates'] = weight_results
        
        # Save comprehensive results
        self._save_data_mining_results(results)
        
        return PhaseResult(
            phase=ExecutionPhase.DATA_PREPARATION,
            success=True,
            data=results,
            message="Data mining complete. Results saved to optimization_results/",
            next_phase=None  # Stops here for manual analysis
        )
    
    def execute_manual_analysis_phase(self) -> PhaseResult:
        """Phase 2: Manual Analysis - User reviews and formulates hypotheses"""
        print("=" * 60)
        print("PHASE 2: MANUAL ANALYSIS")
        print("=" * 60)
        
        # Load Phase 1 results
        phase1_results = self.load_phase_state(ExecutionPhase.DATA_PREPARATION)
        
        # Generate analysis templates
        self._generate_analysis_templates(phase1_results)
        
        print("\nManual analysis phase activated.")
        print("Please review the following files:")
        print("1. optimization_results/phase1_summary.json")
        print("2. optimization_results/regime_performance_matrix.csv")
        print("3. optimization_results/parameter_sensitivity_analysis.json")
        print("\nCreate your hypothesis in: optimization_results/oos_test_configurations.yaml")
        print("\nWhen ready, run with --resume-from=finalization")
        
        return PhaseResult(
            phase=ExecutionPhase.ANALYSIS,
            success=True,
            data={'status': 'awaiting_user_input'},
            message="Awaiting manual analysis completion",
            next_phase=ExecutionPhase.FINALIZATION
        )
    
    def execute_oos_testing_phase(self) -> PhaseResult:
        """Phase 3: Out-of-Sample Testing - Test hypotheses on unseen data"""
        print("=" * 60)
        print("PHASE 3: OUT-OF-SAMPLE TESTING")
        print("=" * 60)
        
        # Load test configurations created during manual analysis
        test_configs = self._load_oos_test_configurations()
        
        if not test_configs:
            return PhaseResult(
                phase=ExecutionPhase.FINALIZATION,
                success=False,
                data={},
                message="No OOS test configurations found"
            )
        
        results = {}
        
        # Run each test configuration
        for config_name, config in test_configs.items():
            print(f"\nTesting configuration: {config_name}")
            test_result = self._run_oos_test(config)
            results[config_name] = test_result
            
            # Print summary
            print(f"  Sharpe Ratio: {test_result['sharpe_ratio']:.3f}")
            print(f"  Total Return: {test_result['total_return']:.2%}")
            print(f"  Max Drawdown: {test_result['max_drawdown']:.2%}")
        
        # Generate final report
        self._generate_final_report(results)
        
        return PhaseResult(
            phase=ExecutionPhase.FINALIZATION,
            success=True,
            data=results,
            message="OOS testing complete. See final_report.html",
            next_phase=None
        )
```

---

## 4. Phased Execution Framework

### 4.1 Phase Management

```python
class PhaseManager:
    """Manages phased execution state and transitions"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.current_phase: Optional[ExecutionPhase] = None
        self.completed_phases: List[ExecutionPhase] = []
        self.phase_results: Dict[ExecutionPhase, PhaseResult] = {}
        
    def start_phase(self, phase: ExecutionPhase) -> None:
        """Mark phase as started"""
        self.current_phase = phase
        print(f"Starting phase: {phase.value}")
        
    def complete_phase(self, phase: ExecutionPhase, result: PhaseResult) -> None:
        """Mark phase as completed and save state"""
        self.completed_phases.append(phase)
        self.phase_results[phase] = result
        
        # Save checkpoint
        checkpoint_path = f"{self.checkpoint_dir}/{phase.value}_checkpoint.json"
        self._save_checkpoint(checkpoint_path, result)
        
        print(f"Phase {phase.value} completed successfully")
        
    def can_execute_phase(self, phase: ExecutionPhase, 
                         dependencies: List[ExecutionPhase]) -> bool:
        """Check if phase can be executed based on dependencies"""
        return all(dep in self.completed_phases for dep in dependencies)
    
    def load_checkpoint(self, phase: ExecutionPhase) -> Optional[PhaseResult]:
        """Load saved checkpoint for a phase"""
        checkpoint_path = f"{self.checkpoint_dir}/{phase.value}_checkpoint.json"
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                return PhaseResult(**data)
        return None
    
    def _save_checkpoint(self, path: str, result: PhaseResult) -> None:
        """Save phase checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'phase': result.phase.value,
                'success': result.success,
                'data': result.data,
                'message': result.message,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
```

### 4.2 CLI Integration

```python
class PhasedCLI:
    """Command-line interface with phased execution support"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='ADMF-Trader')
        self._setup_arguments()
        
    def _setup_arguments(self):
        """Set up command-line arguments"""
        self.parser.add_argument('--config', required=True, help='Configuration file')
        self.parser.add_argument('--mode', choices=['optimize', 'backtest', 'live'], 
                               default='backtest')
        
        # Phased execution arguments
        self.parser.add_argument('--phased', action='store_true',
                               help='Enable phased execution')
        self.parser.add_argument('--stop-after', 
                               choices=['data_preparation', 'processing', 'analysis'],
                               help='Stop after specified phase')
        self.parser.add_argument('--resume-from',
                               choices=['processing', 'analysis', 'finalization'],
                               help='Resume from specified phase')
        self.parser.add_argument('--list-phases', action='store_true',
                               help='List available phases')
        
    def main(self):
        """Main entry point"""
        args = self.parser.parse_args()
        
        # Create system coordinator
        coordinator = SystemCoordinator(args.config)
        
        # Handle phase listing
        if args.list_phases:
            self._list_phases(coordinator, args.mode)
            return
        
        # Execute with phase options
        results = coordinator.execute(
            mode=args.mode,
            phased=args.phased,
            stop_after_phase=args.stop_after,
            resume_from_phase=args.resume_from
        )
        
        # Display results
        self._display_results(results, args.phased)
    
    def _list_phases(self, coordinator: SystemCoordinator, mode: str):
        """List available phases for a mode"""
        if mode not in coordinator.coordinators:
            print(f"Mode '{mode}' not available")
            return
            
        coord = coordinator.coordinators[mode]
        if isinstance(coord, PhasedCoordinator):
            print(f"\nAvailable phases for {mode}:")
            # Would need to implement phase listing in coordinator
            phases = coord.get_available_phases()
            for phase in phases:
                deps = coord.get_phase_dependencies(phase)
                dep_str = f" (requires: {', '.join(d.value for d in deps)})" if deps else ""
                print(f"  - {phase.value}{dep_str}")
        else:
            print(f"Mode '{mode}' does not support phased execution")
```

---

## 5. Container-Based Isolation

### 5.1 Container Integration

```python
class ContainerizedExecutor:
    """Executes components in isolated containers"""
    
    def __init__(self, container_factory: ContainerFactory):
        self.container_factory = container_factory
        self.resource_manager = ContainerResourceManager()
        
    def execute_isolated(self, component_spec: Dict[str, Any], 
                        task: Callable) -> Any:
        """Execute task in isolated container"""
        # Check resources
        if not self.resource_manager.can_create_container():
            raise RuntimeError("Insufficient resources for new container")
        
        # Create container
        container = self.container_factory.create_optimization_container(component_spec)
        container_id = container.container_id
        
        # Register for monitoring
        self.resource_manager.register_container(container_id)
        
        try:
            # Initialize container
            container.initialize_scope()
            
            # Execute task
            result = task(container)
            
            # Update metrics
            self.resource_manager.update_container_metrics(
                container_id,
                {'execution_time': time.time() - start_time}
            )
            
            return result
            
        finally:
            # Cleanup
            container.teardown_scope()
            self.resource_manager.unregister_container(container_id)
    
    def execute_parallel_batch(self, specs: List[Dict[str, Any]], 
                             task: Callable) -> List[Any]:
        """Execute multiple tasks in parallel containers"""
        # Get optimal batch size
        batch_size = self.resource_manager.get_optimization_recommendations()['optimal_batch_size']
        
        results = []
        
        # Process in batches
        for i in range(0, len(specs), batch_size):
            batch = specs[i:i + batch_size]
            batch_results = self._execute_batch(batch, task)
            results.extend(batch_results)
            
            # Cleanup between batches
            gc.collect()
            
        return results
```

---

## 6. Feature Mapping from Current System

### 6.1 Current to New Architecture Mapping

| Current Component | New Implementation | Container Scope | Protocols Used |
|------------------|-------------------|-----------------|----------------|
| **Core Module** |
| Bootstrap | SystemCoordinator + ComponentFactory | System | Coordinator |
| Container | UniversalScopedContainer | Per-execution | ScopedContainer |
| EventBus | EventBus (shared per container) | Container | - |
| Config | Config (shared read-only) | System | Configurable |
| **Data Module** |
| CSVDataHandler | HistoricalDataHandler | Container | DataProvider, DataSplitter |
| DataSplitter | Part of DataHandler | Container | DataSplitter |
| **Strategy Module** |
| BaseStrategy | No base class - use protocols | Container | SignalGenerator, Optimizable |
| Indicator classes | Composed within strategies | Container | Indicator |
| Rule classes | Standalone components | Container | TradingRule |
| RegimeDetector | RegimeDetector component | Container | Classifier, EventSubscriber |
| **Risk Module** |
| RiskManager | RiskManager component | Container | EventSubscriber, Configurable |
| Portfolio | Portfolio component | Container | Portfolio, Resettable |
| PositionSizer | Composed within RiskManager | Container | PositionSizer |
| **Execution Module** |
| SimulatedBroker | SimulatedBroker component | Container | Broker, EventSubscriber |
| OrderManager | Part of Broker | Container | OrderProcessor |
| **Optimization Module** |
| GridSearchOptimizer | GridOptimizer component | Container | Optimizer |
| WorkflowOrchestrator | OptimizationCoordinator | System | PhasedCoordinator |
| ComponentOptimizer | Part of OptimizationCoordinator | System | - |

### 6.2 Migration Benefits

1. **Simpler Components**: No base class overhead, components only include needed functionality
2. **Better Isolation**: Each execution runs in its own container with isolated state
3. **Flexible Composition**: Any component can be combined with any other through protocols
4. **Resource Efficiency**: Shared read-only data, isolated mutable state
5. **Easier Testing**: Components can be tested in complete isolation
6. **Configuration-Driven**: Entire system behavior controlled through configuration

---

## 7. Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Implement base Coordinator protocols
- [ ] Create SystemCoordinator
- [ ] Integrate with UniversalScopedContainer
- [ ] Set up PhaseManager for state management

### Phase 2: Optimization Coordinator (Weeks 3-4)
- [ ] Implement OptimizationCoordinator
- [ ] Create ThreePhaseOptimizationCoordinator
- [ ] Integrate isolated rule optimization
- [ ] Add ensemble weight optimization
- [ ] Implement regime-aware parameter selection

### Phase 3: Component Migration (Weeks 5-6)
- [ ] Migrate strategies to Protocol + Composition
- [ ] Update indicators and rules
- [ ] Convert risk management components
- [ ] Adapt execution components

### Phase 4: Integration & Testing (Weeks 7-8)
- [ ] Create comprehensive test suite
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Example configurations

### Phase 5: Advanced Features (Weeks 9-10)
- [ ] Multi-strategy coordination
- [ ] Live trading coordinator
- [ ] A/B testing framework
- [ ] Advanced monitoring and analytics

## Example Configuration

```yaml
# config/phased_optimization.yaml
system:
  mode: "optimization"
  phased_execution: true
  
optimization:
  workflow:
    type: "three_phase"
    phases:
      data_mining:
        isolated_optimization:
          rules:
            - name: "ma_crossover"
              parameter_space:
                fast_period: [5, 10, 15]
                slow_period: [20, 30, 40]
            - name: "rsi"
              parameter_space:
                period: [14, 21, 30]
                oversold: [20, 30]
                overbought: [70, 80]
        
        ensemble_weights:
          combinations:
            - [0.3, 0.7]
            - [0.5, 0.5]
            - [0.7, 0.3]
      
      analysis:
        auto_generate_reports: true
        templates:
          - "regime_performance_matrix"
          - "parameter_sensitivity"
          - "ensemble_comparison"
      
      oos_testing:
        test_configurations_file: "oos_test_configs.yaml"
        metrics:
          - "sharpe_ratio"
          - "total_return"
          - "max_drawdown"
          - "win_rate"

container_resources:
  max_memory_gb: 16
  max_containers: 50
  optimization_batch_size: 10

components:
  # Component definitions using Protocol + Composition
  regime_detector:
    class: "RegimeDetector"
    capabilities: ["lifecycle", "events", "logging"]
    params:
      method: "trend_volatility"
      
  ma_crossover:
    class: "MACrossoverRule"
    capabilities: ["optimization", "reset"]
    
  rsi:
    class: "RSIRule" 
    capabilities: ["optimization", "reset"]
```

## Summary

The Coordinator architecture provides:

1. **Clear Separation of Concerns**: Coordinators handle workflow, containers provide isolation, components focus on their specific tasks
2. **Phased Execution Support**: Natural stopping points for manual intervention
3. **Maximum Flexibility**: Any component can participate through protocols
4. **Resource Efficiency**: Optimal use of system resources through batching and container management
5. **Configuration-Driven Behavior**: All aspects controlled through configuration files

This architecture enables the complex optimization workflows of the current system while providing a cleaner, more maintainable foundation for future enhancements.