"""
Base classes and interfaces for the ADMF Strategy framework.
"""

from .strategy import Strategy, StrategyComponent
from .indicator import (
    IndicatorBase, 
    IndicatorResult,
    MovingAverageIndicator,
    ExponentialMovingAverageIndicator,
    RSIIndicator
)
from .rule import (
    RuleBase, 
    RuleResult,
    CrossoverRule,
    ThresholdRule,
    MomentumRule
)
from .parameter import ParameterSet, ParameterSpace, Parameter

__all__ = [
    'Strategy',
    'StrategyComponent', 
    'IndicatorBase',
    'IndicatorResult',
    'MovingAverageIndicator',
    'ExponentialMovingAverageIndicator',
    'RSIIndicator',
    'RuleBase',
    'RuleResult',
    'CrossoverRule',
    'ThresholdRule',
    'MomentumRule',
    'ParameterSet',
    'ParameterSpace',
    'Parameter'
]