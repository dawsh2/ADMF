#!/usr/bin/env python3
"""Test script to verify RegimeAdaptiveStrategy refactoring."""

import unittest
from unittest.mock import Mock, MagicMock, patch
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.core.event import Event, EventType


class TestRegimeAdaptiveStrategyRefactor(unittest.TestCase):
    """Test the refactored RegimeAdaptiveStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = {
            'regime_adaptive_strategy': {
                'symbol': 'TEST',
                'initial_capital': 100000,
                'position_size': 1000,
                'regime_detector_service_name': 'TestRegimeDetector',
                'regime_params_file_path': 'test_params.json',
                'fallback_to_overall_best': True,
                'short_window_default': 10,
                'long_window_default': 20
            }
        }
        
        # Create mock container
        self.mock_container = Mock()
        self.mock_container.get_config.return_value = self.mock_config
        
        # Create mock event bus
        self.mock_event_bus = Mock()
        
        # Create mock subscription manager
        self.mock_subscription_manager = Mock()
        
        # Create mock data handler
        self.mock_data_handler = Mock()
        
        # Create mock portfolio
        self.mock_portfolio = Mock()
        
        # Create mock regime detector
        self.mock_regime_detector = Mock()
        self.mock_regime_detector.get_current_classification.return_value = 'trending_up'
        
        # Set up container resolution
        def resolve_side_effect(key):
            if key == 'TestDataHandler':
                return self.mock_data_handler
            elif key == 'TestPortfolio':
                return self.mock_portfolio
            elif key == 'TestRegimeDetector':
                return self.mock_regime_detector
            else:
                raise KeyError(f"Component not found: {key}")
        
        self.mock_container.resolve.side_effect = resolve_side_effect
    
    def test_constructor(self):
        """Test minimal constructor pattern."""
        strategy = RegimeAdaptiveStrategy('TestRegimeAdaptive', 'regime_adaptive_strategy')
        
        # Verify basic initialization
        self.assertEqual(strategy.instance_name, 'TestRegimeAdaptive')
        self.assertEqual(strategy.config_key, 'regime_adaptive_strategy')
        self.assertEqual(strategy._regime_detector_key, 'MyPrimaryRegimeDetector')
        self.assertIsNone(strategy._regime_detector)
        self.assertEqual(strategy._current_regime, 'default')
    
    @patch('src.strategy.regime_adaptive_strategy.os.path.isfile')
    @patch('builtins.open', create=True)
    def test_initialize(self, mock_open, mock_isfile):
        """Test initialization with context."""
        # Mock file existence
        mock_isfile.return_value = True
        
        # Mock file content
        mock_file_content = '''{
            "regime_best_parameters": {
                "trending_up": {
                    "parameters": {
                        "short_window": 5,
                        "long_window": 15
                    },
                    "weights": {
                        "ma_rule.weight": 0.7,
                        "rsi_rule.weight": 0.3
                    }
                }
            }
        }'''
        mock_open.return_value.__enter__.return_value.read.return_value = mock_file_content
        
        # Create strategy
        strategy = RegimeAdaptiveStrategy('TestRegimeAdaptive', 'regime_adaptive_strategy')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config,
            'data_handler_key': 'TestDataHandler',
            'portfolio_key': 'TestPortfolio'
        }
        
        strategy.initialize(context)
        
        # Verify initialization
        self.assertEqual(strategy.container, self.mock_container)
        self.assertEqual(strategy.event_bus, self.mock_event_bus)
        self.assertEqual(strategy.subscription_manager, self.mock_subscription_manager)
        self.assertEqual(strategy._regime_detector_key, 'TestRegimeDetector')
        
        # Verify parameters were loaded
        self.assertIn('trending_up', strategy._regime_specific_params)
    
    def test_setup(self):
        """Test setup method."""
        strategy = RegimeAdaptiveStrategy('TestRegimeAdaptive', 'regime_adaptive_strategy')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config,
            'data_handler_key': 'TestDataHandler',
            'portfolio_key': 'TestPortfolio'
        }
        
        # Skip file loading for this test
        with patch.object(strategy, '_load_parameters_from_file'):
            strategy.initialize(context)
            strategy.setup()
        
        # Verify regime detector was resolved
        self.assertEqual(strategy._regime_detector, self.mock_regime_detector)
        self.assertEqual(strategy._current_regime, 'trending_up')
    
    def test_event_subscriptions(self):
        """Test event subscription setup."""
        strategy = RegimeAdaptiveStrategy('TestRegimeAdaptive', 'regime_adaptive_strategy')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config,
            'data_handler_key': 'TestDataHandler',
            'portfolio_key': 'TestPortfolio'
        }
        
        with patch.object(strategy, '_load_parameters_from_file'):
            strategy.initialize(context)
            strategy.initialize_event_subscriptions()
        
        # Verify CLASSIFICATION subscription was added
        self.mock_subscription_manager.subscribe.assert_called()
        calls = self.mock_subscription_manager.subscribe.call_args_list
        
        # Find CLASSIFICATION subscription
        classification_subscribed = False
        for call in calls:
            if call[0][0] == EventType.CLASSIFICATION:
                classification_subscribed = True
                break
        
        self.assertTrue(classification_subscribed)
    
    def test_lifecycle_methods(self):
        """Test start, stop, and teardown methods."""
        strategy = RegimeAdaptiveStrategy('TestRegimeAdaptive', 'regime_adaptive_strategy')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config,
            'data_handler_key': 'TestDataHandler',
            'portfolio_key': 'TestPortfolio'
        }
        
        with patch.object(strategy, '_load_parameters_from_file'):
            strategy.initialize(context)
            
            # Test start
            strategy.start()
            self.assertEqual(strategy.state, strategy.ComponentState.RUNNING)
            
            # Test stop
            strategy.stop()
            self.assertEqual(strategy.state, strategy.ComponentState.STOPPED)
            
            # Test teardown
            strategy.teardown()
            self.assertIsNone(strategy._regime_detector)
            self.assertEqual(len(strategy._regime_specific_params), 0)


if __name__ == '__main__':
    unittest.main()