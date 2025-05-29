"""
Regime performance analyzer for training phase optimization.
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd


class RegimePerformanceAnalyzer:
    """
    Analyzes strategy performance across different market regimes.
    
    This analyzer:
    1. Tracks performance metrics per regime
    2. Identifies best parameters for each regime
    3. Provides regime transition statistics
    """
    
    def __init__(self):
        self.regime_metrics = defaultdict(lambda: {
            'trades': [],
            'returns': [],
            'sharpe_ratios': [],
            'win_rates': [],
            'parameters': [],
            'trade_counts': []  # Track trade counts per parameter set
        })
        self.regime_transitions = []
        
    def analyze_backtest_results(
        self, 
        results: Dict[str, Any],
        parameters: Dict[str, Any],
        regime_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze backtest results by regime.
        
        Args:
            results: Backtest results including trades and performance
            parameters: Parameters used for this backtest
            regime_history: List of regime changes with timestamps
            
        Returns:
            Dict containing regime-specific performance metrics
        """
        # Check if we have regime performance data from portfolio
        regime_performance_data = results.get('regime_performance', {})
        if regime_performance_data:
            # Process portfolio's regime performance data
            regime_performance = {}
            for regime, data in regime_performance_data.items():
                # Skip summary entries
                if regime.startswith('_'):
                    continue
                    
                trade_count = data.get('count', 0)
                if trade_count > 0:
                    metrics = {
                        'total_return': data.get('gross_pnl', 0),
                        'sharpe_ratio': data.get('sharpe_ratio', 0) if data.get('sharpe_ratio') is not None else 0,
                        'win_rate': data.get('win_rate', 0),
                        'trade_count': trade_count,
                        'avg_win': data.get('avg_pnl', 0) if data.get('avg_pnl', 0) > 0 else 0,
                        'avg_loss': abs(data.get('avg_pnl', 0)) if data.get('avg_pnl', 0) < 0 else 0,
                        'parameters': parameters
                    }
                    
                    regime_performance[regime] = metrics
                    
                    # Store for aggregation across multiple backtests
                    self.regime_metrics[regime]['returns'].append(metrics['total_return'])
                    self.regime_metrics[regime]['sharpe_ratios'].append(metrics['sharpe_ratio'])
                    self.regime_metrics[regime]['win_rates'].append(metrics['win_rate'])
                    self.regime_metrics[regime]['parameters'].append(parameters)
                    self.regime_metrics[regime]['trade_counts'].append(metrics['trade_count'])
                    # Note: not storing individual trades since portfolio doesn't expose them
                    
            return regime_performance
        
        # Fallback to old method if no regime performance data
        trades = results.get('trades', [])
        if not trades:
            return {}
            
        # Create regime timeline
        regime_timeline = self._create_regime_timeline(regime_history)
        
        # Assign trades to regimes
        trades_by_regime = self._assign_trades_to_regimes(trades, regime_timeline)
        
        # Calculate metrics per regime
        regime_performance = {}
        for regime, regime_trades in trades_by_regime.items():
            if not regime_trades:
                continue
                
            metrics = self._calculate_regime_metrics(regime_trades)
            metrics['parameters'] = parameters
            
            regime_performance[regime] = metrics
            
            # Store for aggregation
            self.regime_metrics[regime]['trades'].extend(regime_trades)
            self.regime_metrics[regime]['returns'].append(metrics['total_return'])
            self.regime_metrics[regime]['sharpe_ratios'].append(metrics['sharpe_ratio'])
            self.regime_metrics[regime]['win_rates'].append(metrics['win_rate'])
            self.regime_metrics[regime]['parameters'].append(parameters)
            self.regime_metrics[regime]['trade_counts'].append(metrics['trade_count'])
            
        return regime_performance
        
    def _create_regime_timeline(self, regime_history: List[Dict[str, Any]]) -> List[Tuple[Any, str]]:
        """Create timeline of regime changes."""
        timeline = []
        
        for i, regime_info in enumerate(regime_history):
            timestamp = regime_info['timestamp']
            regime = regime_info['regime']
            timeline.append((timestamp, regime))
            
        return sorted(timeline, key=lambda x: x[0])
        
    def _assign_trades_to_regimes(
        self, 
        trades: List[Dict[str, Any]], 
        regime_timeline: List[Tuple[Any, str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Assign each trade to its regime based on entry time."""
        trades_by_regime = defaultdict(list)
        
        for trade in trades:
            entry_time = trade.get('entry_time')
            if not entry_time:
                continue
                
            # Find the regime at entry time
            regime = self._get_regime_at_time(entry_time, regime_timeline)
            trades_by_regime[regime].append(trade)
            
        return dict(trades_by_regime)
        
    def _get_regime_at_time(self, timestamp: Any, regime_timeline: List[Tuple[Any, str]]) -> str:
        """Get the regime at a specific timestamp."""
        current_regime = "default"
        
        for regime_time, regime in regime_timeline:
            if regime_time <= timestamp:
                current_regime = regime
            else:
                break
                
        return current_regime
        
    def _calculate_regime_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics for trades in a specific regime."""
        if not trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'trade_count': 0
            }
            
        # Extract returns
        returns = []
        wins = 0
        losses = 0
        win_amounts = []
        loss_amounts = []
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            returns.append(pnl)
            
            if pnl > 0:
                wins += 1
                win_amounts.append(pnl)
            elif pnl < 0:
                losses += 1
                loss_amounts.append(abs(pnl))
                
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Calculate returns metrics
        total_return = sum(returns)
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio (simplified - no risk-free rate)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Average win/loss
        avg_win = np.mean(win_amounts) if win_amounts else 0
        avg_loss = np.mean(loss_amounts) if loss_amounts else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trade_count': total_trades,
            'avg_return': avg_return,
            'std_return': std_return
        }
        
    def get_best_parameters_per_regime(self) -> Dict[str, Dict[str, Any]]:
        """Get the best parameters for each regime based on Sharpe ratio."""
        best_params = {}
        
        for regime, metrics in self.regime_metrics.items():
            if not metrics['sharpe_ratios']:
                continue
                
            # Find parameters with highest Sharpe ratio
            best_idx = np.argmax(metrics['sharpe_ratios'])
            best_params[regime] = {
                'parameters': metrics['parameters'][best_idx],
                'sharpe_ratio': metrics['sharpe_ratios'][best_idx],
                'win_rate': metrics['win_rates'][best_idx],
                'total_return': metrics['returns'][best_idx],
                'num_trades': metrics['trade_counts'][best_idx]
            }
            
        return best_params
        
    def get_regime_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated statistics for each regime."""
        stats = {}
        
        for regime, metrics in self.regime_metrics.items():
            if not metrics['returns']:
                continue
                
            # Calculate total trades from the trade_counts we stored
            total_trades = 0
            if 'trade_counts' in metrics:
                total_trades = sum(metrics['trade_counts'])
            elif metrics['trades']:
                # Fallback to old method if available
                total_trades = sum(len(trades) for trades in metrics['trades'])
                
            stats[regime] = {
                'avg_return': np.mean(metrics['returns']),
                'std_return': np.std(metrics['returns']),
                'avg_sharpe': np.mean(metrics['sharpe_ratios']),
                'avg_win_rate': np.mean(metrics['win_rates']),
                'total_trades': total_trades,
                'parameter_sets_tested': len(metrics['parameters'])
            }
            
        return stats
        
    def save_analysis(self, filepath: str):
        """Save analysis results to file."""
        import json
        
        analysis = {
            'best_parameters': self.get_best_parameters_per_regime(),
            'regime_statistics': self.get_regime_statistics()
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
            
        analysis = convert_numpy(analysis)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)