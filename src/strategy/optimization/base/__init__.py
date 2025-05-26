"""
Base classes for the optimization framework.
"""

from .target import OptimizationTarget
from .method import OptimizationMethod, OptimizationResult
from .metric import OptimizationMetric
from .sequence import OptimizationSequence

__all__ = [
    'OptimizationTarget',
    'OptimizationMethod',
    'OptimizationResult',
    'OptimizationMetric',
    'OptimizationSequence'
]