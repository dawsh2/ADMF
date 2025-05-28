#!/usr/bin/env python3
"""Test regime-specific weight optimization"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.container import Container
from src.strategy.optimization.workflow_orchestrator import WorkflowOrchestrator
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_regime_weight_optimization():
    """Test that weight optimization runs per regime with different results"""
    
    # Test configuration focusing on weight optimization
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
                    'config': {
                        'initial_capital': 100000,
                        'position_sizer': {'type': 'fixed', 'size': 1000}
                    }
                },
                'risk_manager': {
                    'class': 'src.risk.basic_risk_manager.BasicRiskManager',
                    'config': {
                        'max_position_size': 10000,
                        'max_portfolio_risk': 0.02,
                        'stop_loss_pct': 0.02
                    }
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
                                'rsi_rule.period': 14,
                                'rsi_rule.oversold': 30,
                                'rsi_rule.overbought': 70,
                                'ma_rule.weight': 0.5,
                                'rsi_rule.weight': 0.5
                            }
                        },
                        'components': {
                            'regime_detector': {
                                'class': 'src.strategy.regime_detector.RegimeDetector',
                                'config': {
                                    'window_sizes': {'trend': 20, 'volatility': 20},
                                    'trend_threshold': 0.001,
                                    'volatility_percentile': 75
                                }
                            },
                            'ma_rule': {
                                'class': 'src.strategy.components.rules.ma_crossover_rule.MACrossoverRule',
                                'config': {
                                    'fast_period': 20,
                                    'slow_period': 50
                                }
                            },
                            'rsi_rule': {
                                'class': 'src.strategy.components.rules.rsi_rule.RSIRule', 
                                'config': {
                                    'period': 14,
                                    'oversold': 30,
                                    'overbought': 70
                                }
                            }
                        }
                    }
                }
            }
        },
        'optimization': {
            'workflow': {
                'steps': [
                    {
                        'name': 'ensemble_weight_optimization',
                        'type': 'ensemble_weight_optimization',
                        'config': {
                            'weight_combinations': [
                                {'ma_rule.weight': 0.8, 'rsi_rule.weight': 0.2},
                                {'ma_rule.weight': 0.6, 'rsi_rule.weight': 0.4},
                                {'ma_rule.weight': 0.5, 'rsi_rule.weight': 0.5},
                                {'ma_rule.weight': 0.4, 'rsi_rule.weight': 0.6},
                                {'ma_rule.weight': 0.2, 'rsi_rule.weight': 0.8}
                            ]
                        }
                    }
                ],
                'train_start': '2024-01-02',
                'train_end': '2024-01-31',
                'test_start': '2024-02-01', 
                'test_end': '2024-02-15'
            }
        }
    }
    
    # Create container and orchestrator
    container = Container()
    orchestrator = WorkflowOrchestrator(container)
    
    # Initialize with our test config
    orchestrator.initialize(config)
    
    # Run optimization
    print("\n" + "="*80)
    print("STARTING REGIME-SPECIFIC WEIGHT OPTIMIZATION TEST")
    print("="*80 + "\n")
    
    results = orchestrator.run()
    
    # Verify results
    print("\n" + "="*80)
    print("VERIFICATION OF RESULTS")
    print("="*80 + "\n")
    
    if results and 'ensemble_weight_optimization' in results:
        weight_results = results['ensemble_weight_optimization']
        
        # Check if we have regime-specific results
        if 'regime_weights' in weight_results:
            print("✓ Found regime-specific weight results")
            
            regime_weights = weight_results['regime_weights']
            print(f"\nNumber of regimes optimized: {len(regime_weights)}")
            
            # Display weights for each regime
            for regime, weights in regime_weights.items():
                print(f"\n{regime.upper()} regime optimal weights:")
                for rule, weight in weights.items():
                    print(f"  - {rule}: {weight:.2f}")
            
            # Check if different regimes have different optimal weights
            weight_values = [tuple(w.values()) for w in regime_weights.values()]
            if len(set(weight_values)) > 1:
                print("\n✓ SUCCESS: Different regimes have different optimal weights!")
            else:
                print("\n⚠ WARNING: All regimes have the same optimal weights")
                
        else:
            print("✗ ERROR: No regime-specific weight results found")
            
    else:
        print("✗ ERROR: No optimization results returned")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_regime_weight_optimization()