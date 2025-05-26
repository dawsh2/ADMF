#!/usr/bin/env python3
"""
Example demonstrating component-based optimization in ADMF.

This example shows how to:
1. Create optimizable components
2. Define parameter spaces
3. Run optimization on individual components
4. Use optimized parameters in a strategy
"""

import logging
from typing import Dict, Any, Optional
from src.core.component_base import ComponentBase
from src.strategy.base.parameter import Parameter, ParameterSpace
from src.strategy.optimization.component_optimizer import ComponentOptimizer
from src.strategy.components.indicators.oscillators import RSIIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class OptimizableMAComponent(ComponentBase):
    """
    Example of a moving average component with optimization support.
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        
        # Default parameters
        self._default_period = 20
        self._default_type = 'SMA'
        
        # Current parameters
        self.period = self._default_period
        self.ma_type = self._default_type
        
        # State
        self._values = []
        self._current_ma = None
        
    def _initialize(self) -> None:
        """Initialize component."""
        # Load from config
        self.period = self.get_specific_config('period', self._default_period)
        self.ma_type = self.get_specific_config('type', self._default_type)
        
        self.logger.info(f"MA Component initialized: period={self.period}, type={self.ma_type}")
        
    def get_parameter_space(self) -> ParameterSpace:
        """Define the parameter space for optimization."""
        space = ParameterSpace(f"{self.instance_name}_space")
        
        # MA period
        space.add_parameter(Parameter(
            name="period",
            param_type="discrete",
            values=[10, 20, 50, 100, 200],
            default=self._default_period,
            description="Moving average period"
        ))
        
        # MA type
        space.add_parameter(Parameter(
            name="ma_type",
            param_type="discrete",
            values=['SMA', 'EMA'],
            default=self._default_type,
            description="Moving average type"
        ))
        
        return space
        
    def get_optimizable_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return {
            "period": self.period,
            "ma_type": self.ma_type
        }
        
    def validate_parameters(self, parameters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters."""
        if "period" in parameters:
            if not isinstance(parameters["period"], int) or parameters["period"] <= 0:
                return False, "Period must be a positive integer"
                
        if "ma_type" in parameters:
            if parameters["ma_type"] not in ['SMA', 'EMA']:
                return False, "MA type must be 'SMA' or 'EMA'"
                
        return True, None
        
    def apply_parameters(self, parameters: Dict[str, Any]) -> None:
        """Apply new parameters."""
        valid, error = self.validate_parameters(parameters)
        if not valid:
            raise ValueError(f"Invalid parameters: {error}")
            
        # Apply parameters
        if "period" in parameters:
            self.period = parameters["period"]
        if "ma_type" in parameters:
            self.ma_type = parameters["ma_type"]
            
        # Reset state
        self._values = []
        self._current_ma = None
        
        self.logger.info(f"Applied parameters: {parameters}")
        
    def update(self, price: float) -> Optional[float]:
        """Update MA with new price."""
        self._values.append(price)
        
        # Keep only needed values
        if len(self._values) > self.period:
            self._values = self._values[-self.period:]
            
        # Calculate MA
        if len(self._values) >= self.period:
            if self.ma_type == 'SMA':
                self._current_ma = sum(self._values) / len(self._values)
            elif self.ma_type == 'EMA':
                # Simple EMA calculation
                if self._current_ma is None:
                    self._current_ma = sum(self._values) / len(self._values)
                else:
                    alpha = 2 / (self.period + 1)
                    self._current_ma = price * alpha + self._current_ma * (1 - alpha)
                    
        return self._current_ma
        
    @property
    def value(self) -> Optional[float]:
        """Get current MA value."""
        return self._current_ma


def demonstrate_component_optimization():
    """Demonstrate component-based optimization workflow."""
    
    print("=== ADMF Component-Based Optimization Demo ===\n")
    
    # 1. Create components
    print("1. Creating optimizable components...")
    
    # Create RSI component (already optimizable)
    rsi = RSIIndicator(instance_name="rsi_indicator")
    
    # Create MA component
    ma_fast = OptimizableMAComponent(instance_name="ma_fast")
    ma_slow = OptimizableMAComponent(instance_name="ma_slow")
    
    # Initialize components (normally done by Bootstrap)
    context = {
        'logger': logging.getLogger('demo'),
        'config': {},
        'event_bus': None,
        'container': None
    }
    
    rsi.initialize(context)
    ma_fast.initialize(context)
    ma_slow.initialize(context)
    
    print("   ✓ Created RSI and MA components\n")
    
    # 2. Show parameter spaces
    print("2. Component parameter spaces:")
    
    for component in [rsi, ma_fast, ma_slow]:
        space = component.get_parameter_space()
        if space:
            print(f"\n   {component.instance_name}:")
            space_dict = space.to_dict()
            for param_name, param_info in space_dict['parameters'].items():
                print(f"     - {param_name}: {param_info}")
                
    # 3. Create optimizer
    print("\n3. Creating ComponentOptimizer...")
    optimizer = ComponentOptimizer(
        backtest_engine=None,  # Would use real backtest engine
        results_dir="demo_optimization_results"
    )
    print("   ✓ Optimizer created\n")
    
    # 4. Optimize RSI component
    print("4. Optimizing RSI component...")
    print("   Running grid search optimization...")
    
    rsi_results = optimizer.optimize_component(
        component=rsi,
        method="grid_search",
        objective_metric="sharpe_ratio",
        minimize=False
    )
    
    print(f"   ✓ Optimization complete!")
    print(f"   - Total iterations: {rsi_results['total_iterations']}")
    print(f"   - Best parameters: {rsi_results['best_parameters']}")
    print(f"   - Best performance: {rsi_results['best_performance']}\n")
    
    # 5. Optimize MA components with constraints
    print("5. Optimizing MA components with constraints...")
    
    # Ensure fast MA has shorter period than slow MA
    constraints = {
        'bounds': {
            'period': {'min': 10, 'max': 50}  # Fast MA constraint
        }
    }
    
    ma_fast_results = optimizer.optimize_component(
        component=ma_fast,
        method="grid_search",
        objective_metric="sharpe_ratio",
        minimize=False,
        constraints=constraints
    )
    
    print(f"   ✓ Fast MA optimization complete!")
    print(f"   - Best parameters: {ma_fast_results['best_parameters']}")
    
    # Optimize slow MA with constraint based on fast MA result
    slow_constraints = {
        'bounds': {
            'period': {
                'min': ma_fast_results['best_parameters']['period'] + 10,
                'max': 200
            }
        }
    }
    
    ma_slow_results = optimizer.optimize_component(
        component=ma_slow,
        method="grid_search",
        objective_metric="sharpe_ratio",
        minimize=False,
        constraints=slow_constraints
    )
    
    print(f"   ✓ Slow MA optimization complete!")
    print(f"   - Best parameters: {ma_slow_results['best_parameters']}\n")
    
    # 6. Show how to use optimized components in a strategy
    print("6. Using optimized components in a strategy:")
    print(f"   - RSI period: {rsi.period}")
    print(f"   - Fast MA: {ma_fast.ma_type} with period {ma_fast.period}")
    print(f"   - Slow MA: {ma_slow.ma_type} with period {ma_slow.period}")
    
    # Simulate some updates
    print("\n   Simulating price updates...")
    prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
    
    for i, price in enumerate(prices):
        rsi_val = rsi.update({'close': price})
        fast_ma = ma_fast.update(price)
        slow_ma = ma_slow.update(price)
        
        if i == len(prices) - 1:  # Last update
            print(f"\n   Final values:")
            print(f"   - Price: {price}")
            print(f"   - RSI: {rsi_val:.2f}" if rsi_val else "   - RSI: Not ready")
            print(f"   - Fast MA: {fast_ma:.2f}" if fast_ma else "   - Fast MA: Not ready")
            print(f"   - Slow MA: {slow_ma:.2f}" if slow_ma else "   - Slow MA: Not ready")
            
    # 7. Show optimization history
    print("\n7. Optimization History:")
    history = optimizer.get_optimization_history()
    for opt in history:
        print(f"   - {opt['component']}: {opt['method']} optimization")
        print(f"     Start: {opt['start_time']}")
        print(f"     Iterations: {len(opt['iterations'])}")
        
    print("\n=== Demo Complete ===")
    print("\nKey takeaways:")
    print("- Components can be optimized independently")
    print("- Each component defines its own parameter space")
    print("- Optimization results are tracked and versioned")
    print("- Optimized parameters can be applied dynamically")
    print("- Components maintain their state through parameter changes")


if __name__ == "__main__":
    demonstrate_component_optimization()