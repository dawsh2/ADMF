"""
Optimization mixins for component-based optimization.

This module provides mixins that add optimization capabilities to components
without modifying their core functionality.
"""

from .base import OptimizationMixin
from .grid_search import GridSearchMixin
from .genetic import GeneticOptimizationMixin

__all__ = [
    'OptimizationMixin',
    'GridSearchMixin', 
    'GeneticOptimizationMixin'
]