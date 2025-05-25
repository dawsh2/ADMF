#!/usr/bin/env python3
"""
Example showing how to use scoped contexts for isolated backtests.

This demonstrates how an optimizer can run multiple backtests with
complete isolation between trials using the new scoped container feature.
"""

from typing import Dict, List, Any, Optional
from src.core.bootstrap import Bootstrap, RunMode, SystemContext
from src.core.config import Config


class ScopedBacktestRunner:
    """Example of using scoped contexts for backtest isolation."""
    
    def __init__(self, bootstrap: Bootstrap):
        self.bootstrap = bootstrap
        self.main_context = bootstrap.get_context()
        self.logger = self.main_context.logger.getChild("BacktestRunner")
        
    def run_isolated_backtest(self, 
                            trial_id: str, 
                            parameters: Dict[str, Any]) -> Optional[float]:
        """
        Run a single backtest in complete isolation.
        
        Args:
            trial_id: Unique identifier for this trial
            parameters: Parameters to test
            
        Returns:
            Metric value from the backtest
        """
        # Create scoped context for this trial
        scope_name = f"trial_{trial_id}"
        scoped_context = self.bootstrap.create_scoped_context(
            scope_name=scope_name,
            shared_services=['config', 'logger']  # Only share stateless services
        )
        
        try:
            # Create fresh components in the scoped context
            # These are completely isolated from other trials
            
            # 1. Register data handler in scoped container
            from src.data.csv_data_handler import CSVDataHandler
            data_handler = CSVDataHandler(scoped_context.event_bus)
            scoped_context.container.register_instance('data_handler', data_handler)
            
            # 2. Register portfolio in scoped container  
            from src.risk.basic_portfolio import BasicPortfolio
            portfolio = BasicPortfolio(scoped_context.event_bus)
            scoped_context.container.register_instance('portfolio', portfolio)
            
            # 3. Register strategy with test parameters
            from src.strategy.ma_strategy import MAStrategy
            strategy = MAStrategy(scoped_context.event_bus)
            strategy.set_parameters(parameters)
            scoped_context.container.register_instance('strategy', strategy)
            
            # 4. Register other required components...
            # (execution handler, risk manager, etc.)
            
            # Initialize all components with scoped context
            for component in [data_handler, portfolio, strategy]:
                if hasattr(component, 'initialize'):
                    component.initialize(scoped_context)
                    
            # Start components
            for component in [data_handler, portfolio, strategy]:
                if hasattr(component, 'start'):
                    component.start()
                    
            # Run the backtest
            self.logger.info(f"Running isolated backtest {trial_id} with params: {parameters}")
            
            # Data handler streams data, strategy generates signals, 
            # portfolio tracks performance - all in complete isolation
            
            # Get results
            metric_value = portfolio.get_total_return()
            self.logger.info(f"Trial {trial_id} completed: return = {metric_value:.4f}")
            
            return metric_value
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial_id}: {e}", exc_info=True)
            return None
            
        finally:
            # Cleanup is automatic - scoped context and all its components
            # will be garbage collected when this method returns
            self.logger.debug(f"Trial {trial_id} cleanup complete")


class ImprovedOptimizer:
    """
    Example optimizer using scoped contexts for complete isolation.
    
    This eliminates state pollution between trials.
    """
    
    def __init__(self, bootstrap: Bootstrap):
        self.bootstrap = bootstrap
        self.context = bootstrap.get_context()
        self.logger = self.context.logger.getChild("Optimizer")
        self.runner = ScopedBacktestRunner(bootstrap)
        
    def optimize(self, parameter_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run optimization with complete isolation between trials.
        
        Args:
            parameter_sets: List of parameter configurations to test
            
        Returns:
            Best parameters and results
        """
        results = []
        
        for i, params in enumerate(parameter_sets):
            # Each trial runs in complete isolation
            metric = self.runner.run_isolated_backtest(
                trial_id=f"opt_{i}", 
                parameters=params
            )
            
            if metric is not None:
                results.append({
                    'parameters': params,
                    'metric': metric
                })
                
        # Find best result
        if results:
            best = max(results, key=lambda x: x['metric'])
            self.logger.info(f"Best parameters: {best['parameters']} with metric: {best['metric']}")
            return best
        else:
            self.logger.error("No successful trials")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize system
    config = Config("config/config.yaml")
    
    with Bootstrap() as bootstrap:
        # Initialize main system
        bootstrap.initialize(config=config, run_mode=RunMode.OPTIMIZATION)
        bootstrap.setup_managed_components()
        
        # Create optimizer that uses scoped contexts
        optimizer = ImprovedOptimizer(bootstrap)
        
        # Define parameter sets to test
        parameter_sets = [
            {"ma_rule.weight": 0.7, "rsi_rule.weight": 0.3},
            {"ma_rule.weight": 0.6, "rsi_rule.weight": 0.4},
            {"ma_rule.weight": 0.5, "rsi_rule.weight": 0.5},
        ]
        
        # Run optimization with complete isolation
        best = optimizer.optimize(parameter_sets)
        
        print(f"Optimization complete. Best: {best}")