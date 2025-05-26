"""
Base class for optimization metrics/objectives.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class OptimizationMetric(ABC):
    """
    Base class for optimization metrics.
    
    Metrics calculate a score from backtest results that
    the optimizer tries to maximize or minimize.
    """
    
    def __init__(self, name: str, direction: str = 'maximize'):
        self.name = name
        self.direction = direction  # 'maximize' or 'minimize'
        
    @abstractmethod
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate metric value from backtest results.
        
        Args:
            results: Dictionary of backtest results
            
        Returns:
            Metric value (higher is better if direction='maximize')
        """
        pass
        
    def normalize(self, value: float) -> float:
        """
        Normalize metric value to standard range.
        
        Default implementation returns value as-is.
        Override for metric-specific normalization.
        """
        return value
        
    def is_better(self, value1: float, value2: float) -> bool:
        """Check if value1 is better than value2."""
        if self.direction == 'maximize':
            return value1 > value2
        else:
            return value1 < value2
            

class SharpeRatioMetric(OptimizationMetric):
    """Sharpe ratio optimization metric."""
    
    def __init__(self, risk_free_rate: float = 0.0):
        super().__init__('sharpe_ratio', 'maximize')
        self.risk_free_rate = risk_free_rate
        
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio from results."""
        # Get returns
        returns = results.get('returns', [])
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        
        # Calculate excess returns
        excess_returns = returns_array - self.risk_free_rate
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(returns_array)
        
        if std_returns == 0:
            return 0.0
            
        # Annualize (assuming daily returns)
        sharpe = mean_excess / std_returns * np.sqrt(252)
        
        return float(sharpe)
        

class TotalReturnMetric(OptimizationMetric):
    """Total return optimization metric."""
    
    def __init__(self):
        super().__init__('total_return', 'maximize')
        
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate total return from results."""
        # Direct total return if available
        if 'total_return' in results:
            return float(results['total_return'])
            
        # Calculate from initial and final values
        initial_value = results.get('initial_value', 100000)
        final_value = results.get('final_value', initial_value)
        
        if initial_value == 0:
            return 0.0
            
        return (final_value - initial_value) / initial_value
        

class MaxDrawdownMetric(OptimizationMetric):
    """Maximum drawdown optimization metric."""
    
    def __init__(self):
        super().__init__('max_drawdown', 'minimize')
        
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate maximum drawdown from results."""
        # Direct max drawdown if available
        if 'max_drawdown' in results:
            return abs(float(results['max_drawdown']))
            
        # Calculate from equity curve
        equity_curve = results.get('equity_curve', [])
        if not equity_curve:
            return 0.0
            
        # Calculate running maximum and drawdowns
        running_max = equity_curve[0]
        max_drawdown = 0.0
        
        for value in equity_curve[1:]:
            running_max = max(running_max, value)
            drawdown = (running_max - value) / running_max if running_max > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
        

class WinRateMetric(OptimizationMetric):
    """Win rate optimization metric."""
    
    def __init__(self):
        super().__init__('win_rate', 'maximize')
        
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate win rate from results."""
        # Direct win rate if available
        if 'win_rate' in results:
            return float(results['win_rate'])
            
        # Calculate from trades
        trades = results.get('trades', [])
        if not trades:
            return 0.0
            
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        total_trades = len(trades)
        
        return winning_trades / total_trades if total_trades > 0 else 0.0
        

class CompositeMetric(OptimizationMetric):
    """Composite metric combining multiple metrics."""
    
    def __init__(self, name: str = 'composite'):
        super().__init__(name, 'maximize')
        self.metrics: Dict[str, OptimizationMetric] = {}
        self.weights: Dict[str, float] = {}
        
    def add_metric(self, metric: OptimizationMetric, weight: float = 1.0) -> None:
        """Add a metric with weight."""
        self.metrics[metric.name] = metric
        self.weights[metric.name] = weight
        
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate weighted composite score."""
        if not self.metrics:
            return 0.0
            
        total_score = 0.0
        total_weight = 0.0
        
        for name, metric in self.metrics.items():
            weight = self.weights.get(name, 1.0)
            score = metric.calculate(results)
            
            # Normalize if metric direction is minimize
            if metric.direction == 'minimize':
                score = -score
                
            total_score += score * weight
            total_weight += weight
            
        return total_score / total_weight if total_weight > 0 else 0.0