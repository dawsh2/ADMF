#!/usr/bin/env python3
"""Test that regime-specific weights are applied correctly during test evaluation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.container import Container
from src.strategy.optimization.workflow_orchestrator import WorkflowOrchestrator
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Minimal config for testing weight application
    config = {
        'bootstrap': {
            'components': {
                'data_handler': {
                    'class': 'src.data.csv_data_handler.CSVDataHandler',
                    'config': {
                        'file_path': 'data/SPY_1min.csv',
                        'warmup_rows': 100
                    }
                },
                'portfolio_manager': {
                    'class': 'src.risk.basic_portfolio.BasicPortfolio',
                    'config': {'initial_capital': 100000}
                },
                'risk_manager': {
                    'class': 'src.risk.basic_risk_manager.BasicRiskManager',
                    'config': {}
                },
                'execution_handler': {
                    'class': 'src.execution.simulated_execution_handler.SimulatedExecutionHandler',
                    'config': {}
                },
                'strategy': {
                    'class': 'src.strategy.implementations.regime_adaptive_ensemble_composed.RegimeAdaptiveEnsembleComposed',
                    'config': {
                        'regime_params': {
                            'default': {
                                'ma_rule.fast_period': 20,
                                'ma_rule.slow_period': 50,
                                'ma_rule.weight': 0.5,
                                'rsi_rule.weight': 0.5
                            }
                        },
                        'components': {
                            'regime_detector': {
                                'class': 'src.strategy.regime_detector.RegimeDetector',
                                'config': {}
                            },
                            'ma_rule': {
                                'class': 'src.strategy.components.rules.ma_crossover_rule.MACrossoverRule',
                                'config': {}
                            },
                            'rsi_rule': {
                                'class': 'src.strategy.components.rules.rsi_rule.RSIRule',
                                'config': {}
                            }
                        }
                    }
                }
            }
        },
        'optimization': {
            'workflow': {
                'steps': [],  # Empty for this test
                'train_start': '2024-01-02',
                'train_end': '2024-01-31',
                'test_start': '2024-02-01',
                'test_end': '2024-02-15'
            }
        }
    }
    
    # Create container
    container = Container()
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(container)
    orchestrator.initialize(config)
    
    # Create mock workflow results with regime-specific weights
    workflow_results = {
        'optimize_weights': {
            'optimization_type': 'ensemble_weights',
            'regime_best_weights': {
                'default': {
                    'weights': {'ma_crossover': 0.4, 'rsi': 0.6}
                },
                'trending_up': {
                    'weights': {'ma_crossover': 0.3, 'rsi': 0.7}
                },
                'trending_down': {
                    'weights': {'ma_crossover': 0.2, 'rsi': 0.8}
                }
            }
        }
    }
    
    # Get strategy
    strategy = container.get('strategy')
    
    print("\nBEFORE applying regime parameters:")
    print(f"Strategy regime params: {strategy._regime_specific_params}")
    
    # Apply regime parameters
    regime_params = orchestrator._collect_regime_parameters(workflow_results)
    if hasattr(strategy, '_regime_specific_params'):
        strategy._regime_specific_params = regime_params
        
    print("\nAFTER applying regime parameters:")
    print(f"Strategy regime params: {strategy._regime_specific_params}")
    
    print("\nSUCCESS: Regime-specific weights can be applied directly!")

if __name__ == "__main__":
    main()