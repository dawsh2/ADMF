# VectorBT-Powered Hybrid Optimization Strategy

## Core Principle

Leverage VectorBT as a high-speed vectorized component within the ADMF container architecture for rapid parameter space exploration, then validate the best candidates through the event-driven system. This provides massive speed benefits while maintaining the integrity and production-readiness of the event-driven architecture.

```python
# Key: VectorBT for exploration speed + ADMF containers for validation integrity
```

## VectorBT Integration Architecture

### VectorBT as a Component
VectorBT becomes a specialized optimization component within the ADMF ecosystem:

```python
class VectorBTOptimizer:
    """VectorBT-powered fast parameter exploration component"""
    
    def __init__(self, strategy_func):
        self.vbt_strategy = strategy_func
        self.vbt_portfolio = None
        
    def get_parameter_space(self):
        # Define parameter ranges for VectorBT optimization
        return {
            'ma_short': range(5, 50, 5),
            'ma_long': range(20, 200, 10),
            'rsi_period': range(10, 30, 2)
        }
    
    def optimize_vectorized(self, param_combinations):
        # Use VectorBT's built-in optimization capabilities
        self.vbt_portfolio = vbt.Portfolio.from_signals(
            data=self.data,
            entries=self.generate_entries,
            exits=self.generate_exits
        )
        
        # VectorBT can test 1000s of combinations in minutes
        results = self.vbt_portfolio.optimize(
            param_combinations,
            metric='sharpe_ratio',
            max_workers=None  # Use all cores
        )
        
        return results.stats()
```

### Container-Based Isolation
Even with VectorBT's speed, maintain container isolation:

```python
class VectorBTContainer(UniversalScopedContainer):
    """Specialized container for VectorBT operations"""
    
    def __init__(self, data_slice, container_id):
        super().__init__(container_id)
        self.data_slice = data_slice  # Isolated data
        
    def create_vbt_component(self, strategy_spec):
        # VectorBT component with isolated data
        vbt_optimizer = VectorBTOptimizer(strategy_spec)
        vbt_optimizer.set_data(self.data_slice)
        
        self.local_components['vbt_optimizer'] = vbt_optimizer
        return vbt_optimizer
        
    def run_parameter_batch(self, param_batch):
        # VectorBT processes entire batch internally
        # Container provides the isolation boundary
        vbt_optimizer = self.resolve('vbt_optimizer')
        return vbt_optimizer.optimize_vectorized(param_batch)
```

## Three-Stage Hybrid Framework

### Stage 1: VectorBT Exploration (Fast & Broad)
- Use VectorBT's vectorized engine to test 10,000+ parameter combinations
- Leverage VectorBT's built-in performance metrics and analytics
- Complete parameter space exploration in minutes
- Identify top 1-5% of candidates

### Stage 2: Analysis Phase Integration
- Export VectorBT results to Jupyter notebooks
- Leverage VectorBT's rich visualization capabilities
- Use VectorBT's built-in statistical analysis tools
- Generate hypotheses for Stage 3 testing

### Stage 3: ADMF Event-Driven Validation (Precise & Realistic)
- Validate top candidates through full ADMF event-driven backtesting
- Realistic execution simulation with slippage, commissions
- Production-ready risk management validation
- Final parameter selection with confidence

## Implementation

### VectorBT-ADMF Hybrid Optimizer
```python
class VectorBTADMFHybridOptimizer:
    """Combines VectorBT's speed with ADMF's precision and container isolation"""
    
    def __init__(self, vbt_container_factory, admf_container_factory):
        self.vbt_factory = vbt_container_factory
        self.admf_factory = admf_container_factory
        self.validation_threshold = 0.95
        self.lifecycle_manager = ContainerLifecycleManager()
    
    def optimize(self, strategy_config: Dict, param_space: Dict, top_n: int = 20):
        """
        Three-stage optimization leveraging VectorBT's strengths
        """
        
        # Stage 1: VectorBT bulk exploration
        self.logger.info(f"Stage 1: VectorBT exploration of {len(param_space)} combinations")
        vbt_results = self._run_vectorbt_exploration(strategy_config, param_space)
        
        # Stage 2: Analysis phase with VectorBT's rich analytics
        analysis_results = self._generate_vbt_analysis(vbt_results, strategy_config)
        
        # Stage 3: ADMF validation of top candidates
        self.logger.info(f"Stage 3: ADMF validation of top {top_n} candidates")
        validated_results = self._run_admf_validation(vbt_results, strategy_config, top_n)
        
        return {
            'vbt_exploration': vbt_results,
            'analysis': analysis_results,
            'validated_results': validated_results
        }
    
    def _run_vectorbt_exploration(self, strategy_config, param_space):
        """Use VectorBT for rapid parameter space exploration"""
        
        # Create VectorBT container with isolated data
        vbt_container_id = self.lifecycle_manager.create_and_start_container(
            "vectorbt_optimization",
            {**strategy_config, 'param_space': param_space}
        )
        
        try:
            vbt_container = self.lifecycle_manager.active_containers[vbt_container_id]
            vbt_optimizer = vbt_container.resolve('vbt_optimizer')
            
            # VectorBT's vectorized optimization - test thousands of combinations
            optimization_results = vbt_optimizer.optimize(
                param_space,
                metric='sharpe_ratio',
                direction='max',
                max_workers=None,  # Use all available cores
                show_progress=True
            )
            
            # Extract results with VectorBT's rich metrics
            return {
                'best_params': optimization_results.params,
                'best_metrics': optimization_results.stats(),
                'full_results': optimization_results,
                'performance_surface': optimization_results.heatmap_data,
                'container_metrics': vbt_container.get_performance_metrics()
            }
            
        finally:
            self.lifecycle_manager.stop_and_destroy_container(vbt_container_id)
    
    def _generate_vbt_analysis(self, vbt_results, strategy_config):
        """Leverage VectorBT's built-in analysis capabilities"""
        
        full_results = vbt_results['full_results']
        
        # Use VectorBT's visualization and analysis tools
        analysis = {
            'parameter_heatmaps': full_results.heatmap(),
            'performance_surface': full_results.plot(),
            'correlation_matrix': full_results.correlation_matrix(),
            'drawdown_analysis': full_results.drawdown_stats(),
            'trade_analysis': full_results.trade_stats(),
            
            # Export for Jupyter analysis
            'jupyter_data': {
                'results_df': full_results.to_dataframe(),
                'trades_df': full_results.trades.to_dataframe(),
                'returns_df': full_results.returns.to_dataframe()
            }
        }
        
        return analysis
    
    def _run_admf_validation(self, vbt_results, strategy_config, top_n):
        """Validate top VectorBT candidates through ADMF event-driven system"""
        
        # Get top N parameter sets from VectorBT results
        full_results = vbt_results['full_results']
        top_params = full_results.get_top_n_params(top_n, metric='sharpe_ratio')
        
        validated_results = {}
        discrepancies = []
        
        for i, params in enumerate(top_params):
            self.logger.info(f"Validating candidate {i+1}/{top_n}: {params}")
            
            # Create isolated ADMF container for validation
            admf_container_id = self.lifecycle_manager.create_and_start_container(
                "admf_validation",
                {**strategy_config, 'parameters': params}
            )
            
            try:
                admf_container = self.lifecycle_manager.active_containers[admf_container_id]
                
                # Run full event-driven backtest
                admf_results = self._execute_admf_backtest(admf_container)
                
                # Compare with VectorBT results
                vbt_metrics = full_results.get_metrics_for_params(params)
                discrepancy = self._compare_results(vbt_metrics, admf_results)
                
                if discrepancy['significant']:
                    discrepancies.append({
                        'params': params,
                        'vbt_metrics': vbt_metrics,
                        'admf_metrics': admf_results,
                        'discrepancy': discrepancy
                    })
                
                validated_results[str(params)] = {
                    'vbt_metrics': vbt_metrics,
                    'admf_metrics': admf_results,
                    'discrepancy': discrepancy,
                    'validation_passed': not discrepancy['significant']
                }
                
            finally:
                self.lifecycle_manager.stop_and_destroy_container(admf_container_id)
        
        # Report validation results
        if discrepancies:
            self.logger.warning(f"Found {len(discrepancies)} significant discrepancies!")
            self._analyze_discrepancies(discrepancies)
        
        return validated_results
```

### Lookahead Detection System
```python
class LookaheadDetector:
    """Detects potential lookahead bias by comparing vectorized vs event-driven results"""
    
    def validate_vectorized_implementation(self, param_set: Dict):
        """Run same parameters through both engines"""
        
        # Run through both systems
        vectorized_result = self.vectorized_engine.run(param_set)
        event_driven_result = self.event_driven_engine.run(param_set)
        
        # Check key metrics
        checks = {
            'trade_count_match': abs(vectorized_result['trades'] - event_driven_result['trades']) < 2,
            'entry_price_match': self._compare_entry_prices(vectorized_result, event_driven_result),
            'sharpe_match': abs(vectorized_result['sharpe'] - event_driven_result['sharpe']) < 0.1,
            'return_match': abs(vectorized_result['return'] - event_driven_result['return']) < 0.01
        }
        
        if not all(checks.values()):
            self.logger.error(f"Lookahead bias detected! Failed checks: {checks}")
            self._diagnose_discrepancy(vectorized_result, event_driven_result)
            
        return all(checks.values())
```

### Vectorized Pre-Screener
```python
class VectorizedPreScreener:
    """Use vectorized ops to quickly filter parameter space"""
    
    def prescreen_parameters(self, param_combinations: List[Dict]) -> List[Dict]:
        """
        Quickly eliminate obviously bad parameters
        Returns top 5% for full event-driven testing
        """
        scores = []
        
        for params in param_combinations:
            # Vectorized scoring - just rough estimates
            score = self._quick_score(params)
            scores.append((score, params))
            
        # Return top 5% for proper backtesting
        scores.sort(reverse=True)
        n_top = max(1, len(scores) // 20)
        
        return [params for _, params in scores[:n_top]]
    
    def _quick_score(self, params):
        # Simple vectorized calculations
        # This is just for filtering, not final decisions
        ma_short = self.price_df.rolling(params['ma_short']).mean()
        ma_long = self.price_df.rolling(params['ma_long']).mean()
        signals = (ma_short > ma_long).astype(int).diff()
        
        # Quick performance estimate
        returns = self.price_df.pct_change() * signals.shift(1)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        
        return sharpe
```

## VectorBT Integration Benefits

### 1. **Massive Speed Advantage**
- VectorBT can test 10,000+ parameter combinations in minutes vs hours
- Built-in multiprocessing and optimized NumPy operations
- Mature performance optimization

### 2. **Rich Analytics Out-of-the-Box**
- Comprehensive performance metrics
- Built-in visualization capabilities  
- Statistical analysis tools
- No need to re-implement common calculations

### 3. **Perfect Analysis Phase Integration**
- VectorBT results export naturally to DataFrames
- Built-in plotting for parameter surfaces and heatmaps
- Rich trade-level analysis
- Seamless Jupyter notebook integration

### 4. **Container Architecture Synergy**
```python
# VectorBT provides the engine, ADMF provides the architecture
class VectorBTADMFWorkflow:
    def run_three_phase_optimization(self):
        
        # Phase 1: VectorBT Data Mining
        vbt_results = self.run_vbt_exploration()
        
        # Phase 2: Analysis (VectorBT + Custom Tools)
        analysis = self.generate_analysis_notebooks(vbt_results)
        
        # Phase 3: ADMF OOS Validation
        final_results = self.validate_with_admf(vbt_results.top_candidates)
        
        return final_results
```

## Optimization Workflow Example

```yaml
# VectorBT-ADMF Configuration
optimization:
  workflow_type: "vectorbt_admf_hybrid"
  
  stage1_vectorbt:
    engine: "vectorbt"
    parameter_combinations: 50000  # VectorBT can handle massive spaces
    multiprocessing: true
    chunk_size: 1000
    metrics: ['sharpe_ratio', 'total_return', 'max_drawdown']
    
  stage2_analysis:
    auto_export: true
    formats: ['notebook', 'html', 'parquet']
    vbt_visualizations: true
    custom_analysis: ['regime_breakdown', 'correlation_matrix']
    
  stage3_validation:
    engine: "admf_event_driven"
    top_n_candidates: 25
    validation_metrics: ['sharpe', 'return', 'trade_count', 'max_drawdown']
    container_isolation: true

# Container specifications
containers:
  vectorbt_container:
    type: "vectorbt_optimization"
    resources:
      memory: "8GB"
      cpu_cores: 8
    components:
      - name: "vbt_optimizer"
        class: "VectorBTOptimizer"
        
  admf_container:
    type: "event_driven_validation"  
    resources:
      memory: "2GB"
      cpu_cores: 1
    components:
      - name: "strategy"
        class: "RegimeAdaptiveStrategy"
      - name: "risk_manager"  
        class: "RiskManager"
      - name: "portfolio"
        class: "Portfolio"
```

```python
# Usage Example
from admf.optimization import VectorBTADMFHybridOptimizer

# Initialize hybrid optimizer
optimizer = VectorBTADMFHybridOptimizer(
    vbt_container_factory=VectorBTContainerFactory(),
    admf_container_factory=ADMFContainerFactory()
)

# Define massive parameter space (VectorBT can handle it)
param_space = {
    'ma_short': range(3, 50, 1),      # 47 values
    'ma_long': range(20, 200, 2),     # 90 values  
    'rsi_period': range(5, 50, 1),    # 45 values
    'rsi_oversold': range(15, 35, 1), # 20 values
    'rsi_overbought': range(65, 85, 1) # 20 values
}
# Total: 47 * 90 * 45 * 20 * 20 = 76,140,000 combinations!

# VectorBT can explore this massive space
results = optimizer.optimize(
    strategy_config={
        'class': 'TrendMeanReversionStrategy',
        'data_path': 'data/EURUSD_1min.csv'
    },
    param_space=param_space,
    top_n=50  # Validate top 50 with ADMF
)

# Results include:
# - VectorBT exploration of 76M combinations  
# - Rich analysis with built-in VectorBT tools
# - ADMF validation of top 50 candidates
# - Discrepancy analysis between engines
```

## Safety Guidelines

### ✅ SAFE Vectorized Uses:
- Technical indicator calculations
- Signal frequency analysis  
- Correlation studies
- Initial parameter filtering
- Performance rough estimates
- Parameter sensitivity analysis

### ❌ UNSAFE Vectorized Uses:
- Final performance metrics
- Position sizing calculations
- Risk management decisions
- Order execution logic
- Anything with path dependency
- Production trading decisions

## Practical Implementation Tips

### 1. Progressive Validation
```python
# Start with 1 parameter set to ensure systems match
# Then scale up to full optimization
validation_stages = [1, 10, 100, 1000, 10000]
for n in validation_stages:
    if not validate_sample(n):
        break  # Fix issues before scaling up
```

### 2. Discrepancy Analysis
```python
def analyze_discrepancy(self, vectorized, event_driven):
    """Discrepancies often reveal subtle bugs"""
    
    # Common causes:
    # 1. Different fill assumptions
    # 2. Slippage/commission handling  
    # 3. Signal timing (close vs next open)
    # 4. Data alignment issues
    
    # Use this as a debugging tool!
    discrepancy_report = {
        'fill_price_diff': self._compare_fills(vectorized, event_driven),
        'signal_timing_diff': self._compare_signal_timing(vectorized, event_driven),
        'position_size_diff': self._compare_positions(vectorized, event_driven)
    }
    
    return discrepancy_report
```

### 3. Caching Layer
```python
from functools import lru_cache

# Cache event-driven results to avoid re-running
@lru_cache(maxsize=1000)
def cached_event_driven_backtest(param_hash):
    return event_driven_engine.run(params)
```

### 4. Parallel Processing
```python
from multiprocessing import Pool

def parallel_vectorized_scan(param_grid):
    """Run vectorized scans in parallel for even more speed"""
    with Pool() as pool:
        results = pool.map(vectorized_engine.evaluate, param_grid)
    return results
```

## Benefits of This Approach

1. **Speed**: Test 10,000+ combinations quickly
2. **Integrity**: Final results are always from event-driven system
3. **Debugging**: Discrepancies between systems reveal implementation bugs
4. **Confidence**: Production system matches research results
5. **Scalability**: Can explore much larger parameter spaces

## Example: Complete Workflow

```python
class OptimizationPipeline:
    def __init__(self):
        self.vectorized = VectorizedBacktester()
        self.event_driven = EventDrivenBacktester()
        self.validator = LookaheadDetector()
        
    def run_complete_optimization(self, strategy_config):
        # 1. Generate parameter grid
        param_grid = self.generate_parameter_grid(strategy_config)
        print(f"Generated {len(param_grid)} parameter combinations")
        
        # 2. Vectorized pre-screening
        print("Stage 1: Vectorized pre-screening...")
        top_candidates = self.vectorized.find_top_performers(
            param_grid, 
            top_percent=5  # Keep top 5%
        )
        
        # 3. Event-driven validation
        print(f"Stage 2: Validating {len(top_candidates)} candidates...")
        validated_results = {}
        for params in top_candidates:
            result = self.event_driven.run(params)
            validated_results[params] = result
            
        # 4. Final selection
        best_params = max(validated_results.items(), 
                         key=lambda x: x[1]['sharpe'])
        
        # 5. Sanity check
        self.validator.validate_vectorized_implementation(best_params[0])
        
        return best_params
```

## Why VectorBT Integration is Game-Changing

### 1. **Best of Both Worlds**
- **VectorBT's Speed**: 100x faster parameter exploration
- **VectorBT's Maturity**: Battle-tested analytics and visualizations
- **ADMF's Architecture**: Container isolation, regime awareness, production path
- **ADMF's Precision**: Event-driven realism for final validation

### 2. **Exponential Scale Improvement**
```
Without VectorBT: Test 1,000 combinations in 2 hours
With VectorBT: Test 100,000 combinations in 30 minutes + validate top 50 in 1 hour
Net Result: 100x more parameter space exploration in less time
```

### 3. **Natural Three-Phase Fit**
- **Phase 1**: VectorBT's strength (bulk exploration)
- **Phase 2**: VectorBT's analytics + ADMF's analysis framework
- **Phase 3**: ADMF's strength (realistic validation)

### 4. **Ecosystem Leverage**
- Don't reinvent VectorBT's optimized calculations
- Leverage their community and documentation
- Use their proven visualization tools
- Build on their testing and optimization

### 5. **Strategic Positioning**
Your system becomes "VectorBT-powered" rather than "competing with VectorBT":
- VectorBT provides the high-performance engine
- ADMF provides the sophisticated architecture
- Perfect division of responsibilities

## Implementation Priority

This VectorBT integration should be **Phase 1** of your architecture implementation:

1. **Start Here**: Build VectorBT containers and hybrid optimizer
2. **Prove Concept**: Demonstrate 100x speed improvement  
3. **Add Sophistication**: Layer on regime awareness, three-phase workflow
4. **Scale Up**: Add distributed optimization, production features

The VectorBT foundation gives you immediate, massive performance gains while you build out the more sophisticated features of your architecture.

## Avoiding the Two-System Trap

### The Integration Challenge

VectorBT operates on entire arrays at once (vectorized), while your event-driven components (regime classifiers, risk managers) operate event-by-event. Here's how to bridge this elegantly without maintaining two separate systems:

### Architectural Solutions

#### 1. **Shared Computation Layer**
Create components that can operate in both modes:

```python
class RegimeDetector:
    """Works in both vectorized and event-driven modes"""
    
    def detect_vectorized(self, price_data: pd.DataFrame) -> pd.Series:
        """Vectorized regime detection for VectorBT"""
        # Process entire series at once
        returns = price_data.pct_change()
        volatility = returns.rolling(20).std()
        
        regimes = pd.Series(index=price_data.index)
        regimes[volatility > 0.02] = 'high_vol'
        regimes[volatility <= 0.02] = 'low_vol'
        
        return regimes
    
    def detect_event(self, event: Event) -> RegimeType:
        """Event-driven detection for ADMF"""
        # Process single event using same logic
        self.price_buffer.append(event.price)
        if len(self.price_buffer) >= 20:
            recent_vol = self._calculate_volatility(self.price_buffer[-20:])
            return 'high_vol' if recent_vol > 0.02 else 'low_vol'
        
    def get_vectorbt_indicator(self):
        """Return VectorBT-compatible indicator"""
        return vbt.IndicatorFactory(
            class_name="RegimeIndicator",
            module_name="admf.regime",
            short_name="regime",
            input_names=["close"],
            param_names=["window", "threshold"]
        ).from_apply_func(self.detect_vectorized)
```

#### 2. **Protocol-Based Dual Mode Components**
```python
@runtime_checkable
class DualModeComponent(Protocol):
    """Components that work in both systems"""
    
    def compute_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Batch computation for VectorBT"""
        ...
    
    def compute_event(self, event: Event) -> Any:
        """Single event computation for ADMF"""
        ...
    
    def to_vbt_indicator(self) -> vbt.indicators.factory:
        """Convert to VectorBT indicator"""
        ...
```

#### 3. **Computation Core + Adapters Pattern** (Recommended)
This is the cleanest approach - separate business logic from framework-specific code:

```python
# Core computation logic (shared)
class RegimeDetectionCore:
    """Pure computation logic, no framework dependencies"""
    
    @staticmethod
    def calculate_volatility_regime(prices, window=20, threshold=0.02):
        # Pure numpy/pandas computation
        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(window).std()
        
        regime = np.where(volatility > threshold, 'high_vol', 'low_vol')
        return regime

# VectorBT Adapter
class VectorBTRegimeAdapter:
    def __init__(self, core: RegimeDetectionCore):
        self.core = core
        
    def create_indicator(self):
        return vbt.IndicatorFactory.from_custom_func(
            self.core.calculate_volatility_regime,
            param_names=['window', 'threshold']
        )

# ADMF Event Adapter  
class ADMFRegimeAdapter:
    def __init__(self, core: RegimeDetectionCore):
        self.core = core
        self.price_buffer = []
        
    def on_event(self, event: Event):
        self.price_buffer.append(event.price)
        if len(self.price_buffer) >= self.window:
            regime = self.core.calculate_volatility_regime(
                self.price_buffer[-self.window:]
            )
            return regime[-1]
```

### Recommended Architecture: Thin Adapter Layer

```
┌─────────────────────────────────────────┐
│          Core Business Logic            │
│  (RegimeDetection, RiskCalcs, etc.)     │
│      Pure Python/NumPy/Pandas           │
└────────────┬───────────────┬────────────┘
             │               │
    ┌────────▼─────┐   ┌────▼──────┐
    │ VectorBT     │   │   ADMF    │
    │  Adapter     │   │  Adapter  │
    └──────────────┘   └───────────┘
```

**Advantages**:
- Single source of truth for logic
- Easy to test core logic independently
- Minimal code duplication
- Clear separation of concerns

### Pre-computation Pipeline for Complex Features

For features that are truly event-driven and can't be easily vectorized:

```python
class PreComputePipeline:
    def prepare_for_vectorbt(self, raw_data):
        # Step 1: Run through ADMF to compute complex features
        features = self.run_admf_feature_extraction(raw_data)
        
        # Step 2: Merge with original data
        enhanced_data = pd.concat([raw_data, features], axis=1)
        
        # Step 3: Now VectorBT can use all features
        return enhanced_data
        
class HybridOptimizationWorkflow:
    """Uses pre-computation for complex features"""
    
    def optimize(self, data):
        # Pre-compute regime classifications using ADMF
        self.logger.info("Pre-computing regimes using ADMF event system...")
        regime_data = self.compute_regimes_event_driven(data)
        
        # Pass to VectorBT with pre-computed regimes
        enhanced_data = data.copy()
        enhanced_data['regime'] = regime_data
        enhanced_data['regime_confidence'] = regime_data['confidence']
        
        # VectorBT optimization using pre-computed features
        results = vbt.Portfolio.from_signals(
            data=enhanced_data,
            entries=self.generate_entries_with_regime,
            exits=self.generate_exits_with_regime
        ).optimize(
            window=range(10, 50),
            threshold=np.arange(0.01, 0.05, 0.01)
        )
        
        return results
```

### Complete ML Feature Engineering Example

```python
class UnifiedMLFeatureEngineering:
    """Shared ML feature engineering for both systems"""
    
    def __init__(self):
        self.technical_features = TechnicalFeatureCore()
        self.regime_detector = RegimeDetectionCore()
        self.ml_models = {}
        
    def compute_features_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all features in batch mode for VectorBT"""
        features = pd.DataFrame(index=data.index)
        
        # Technical features (easily vectorized)
        features['rsi'] = self.technical_features.rsi(data.close)
        features['macd_signal'] = self.technical_features.macd_signal(data.close)
        features['bb_position'] = self.technical_features.bollinger_position(data.close)
        
        # Regime features
        features['regime'] = self.regime_detector.calculate_regimes(data)
        features['regime_duration'] = self._calculate_regime_duration(features['regime'])
        
        # ML predictions (batch inference)
        feature_matrix = features[['rsi', 'macd_signal', 'bb_position']].values
        features['ml_signal'] = self.ml_models['main'].predict_proba(feature_matrix)[:, 1]
        
        return features
    
    def compute_features_event(self, event_buffer: EventBuffer) -> Dict[str, float]:
        """Compute features incrementally for ADMF"""
        # Use the same core logic but in streaming mode
        latest_prices = event_buffer.get_recent_prices(100)
        
        features = {
            'rsi': self.technical_features.rsi(latest_prices)[-1],
            'macd_signal': self.technical_features.macd_signal(latest_prices)[-1],
            'bb_position': self.technical_features.bollinger_position(latest_prices)[-1]
        }
        
        # ML inference on single sample
        feature_vector = np.array([[features['rsi'], features['macd_signal'], features['bb_position']]])
        features['ml_signal'] = self.ml_models['main'].predict_proba(feature_vector)[0, 1]
        
        return features
    
    def create_vbt_strategy(self, data: pd.DataFrame):
        """Create VectorBT strategy using unified features"""
        # Compute features once
        features = self.compute_features_vectorized(data)
        
        # Define entry/exit rules using features
        entries = (
            (features['ml_signal'] > 0.7) & 
            (features['regime'] != 'high_vol') &
            (features['rsi'] < 70)
        )
        
        exits = (
            (features['ml_signal'] < 0.3) | 
            (features['rsi'] > 80) |
            (features['regime_duration'] > 50)
        )
        
        return vbt.Portfolio.from_signals(
            close=data.close,
            entries=entries,
            exits=exits,
            fees=0.001
        )
```

### Best Practices for Unified System

#### DO:
1. **Extract core logic** into framework-agnostic modules
2. **Use adapters** to integrate with each framework
3. **Pre-compute** truly event-driven features when needed
4. **Share data structures** (DataFrames work in both systems)
5. **Design for both paradigms** from the start

#### DON'T:
1. **Duplicate logic** in both systems
2. **Force vectorization** of inherently sequential operations
3. **Create deep dependencies** on either framework
4. **Mix paradigms** within the same component

### Integration with Advanced Features

All the advanced features you plan to implement can follow this pattern:

```python
# Optuna Integration
class OptunaOptimizer:
    def __init__(self, core_strategy: StrategyCore):
        self.core = core_strategy
        self.vbt_adapter = VectorBTAdapter(core_strategy)
        self.admf_adapter = ADMFAdapter(core_strategy)
    
    def optimize(self, data, mode='hybrid'):
        if mode == 'vectorized':
            return self._optimize_with_vectorbt(data)
        elif mode == 'event_driven':
            return self._optimize_with_admf(data)
        else:  # hybrid
            # Use VectorBT for exploration
            candidates = self._explore_with_vectorbt(data)
            # Validate with ADMF
            return self._validate_with_admf(candidates, data)

# Walk-Forward Analysis
class WalkForwardAnalyzer:
    def __init__(self, feature_engine: UnifiedMLFeatureEngineering):
        self.features = feature_engine
        
    def analyze(self, data, window_size, step_size):
        results = []
        
        for train_start, train_end, test_start, test_end in self.get_windows(data, window_size, step_size):
            # Pre-compute features for this window
            train_features = self.features.compute_features_vectorized(data[train_start:train_end])
            
            # Optimize on training window using VectorBT
            vbt_results = self.optimize_window_vectorbt(train_features)
            
            # Validate on test window using ADMF
            admf_results = self.validate_window_admf(data[test_start:test_end], vbt_results.best_params)
            
            results.append({
                'window': (train_start, train_end, test_start, test_end),
                'vbt_insample': vbt_results,
                'admf_outofsample': admf_results
            })
            
        return results
```

## Summary

This unified approach ensures you're not maintaining two systems. Instead, you have:

1. **One set of business logic** (regime detection, ML models, risk calculations)
2. **Two execution adapters** (VectorBT for speed, ADMF for precision)
3. **Shared data pipeline** (features computed once, used by both)
4. **Consistent results** (same logic, different execution modes)

The VectorBT component accesses all your advanced features through adapters or pre-computation, ensuring consistency while leveraging each framework's strengths. This architecture scales beautifully as you add more sophisticated features - each new feature is implemented once in the core layer and automatically becomes available to both execution engines.