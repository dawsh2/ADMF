#!/usr/bin/env python3
"""Test script to verify RegimeDetector refactoring."""

import unittest
from unittest.mock import Mock, MagicMock, patch
from src.strategy.regime_detector import RegimeDetector
from src.core.event import Event, EventType


class TestRegimeDetectorRefactor(unittest.TestCase):
    """Test the refactored RegimeDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = {
            'regime_detector': {
                'regime_thresholds': {
                    'trending_up': {
                        'ma_trend': {'min': 0.5},
                        'rsi': {'min': 50, 'max': 70}
                    },
                    'trending_down': {
                        'ma_trend': {'max': -0.5},
                        'rsi': {'min': 30, 'max': 50}
                    }
                },
                'min_regime_duration': 3,
                'verbose_logging': False,
                'summary_interval': 100,
                'debug_mode': False,
                'indicators': {
                    'ma_trend': {
                        'type': 'simple_ma_trend',
                        'parameters': {'period': 20}
                    },
                    'rsi': {
                        'type': 'rsi',
                        'parameters': {'period': 14}
                    }
                }
            }
        }
        
        # Create mock container
        self.mock_container = Mock()
        self.mock_container.get_config.return_value = self.mock_config
        
        # Create mock event bus
        self.mock_event_bus = Mock()
        self.mock_event_bus._subscribers = {EventType.CLASSIFICATION: []}
        
        # Create mock subscription manager
        self.mock_subscription_manager = Mock()
    
    def test_constructor(self):
        """Test minimal constructor pattern."""
        detector = RegimeDetector('TestRegimeDetector', 'regime_detector')
        
        # Verify basic initialization
        self.assertEqual(detector.instance_name, 'TestRegimeDetector')
        self.assertEqual(detector.config_key, 'regime_detector')
        self.assertEqual(len(detector._regime_indicators), 0)
        self.assertEqual(detector._min_regime_duration, 1)
        self.assertIsNone(detector._current_classification)
    
    def test_initialize(self):
        """Test initialization with context."""
        detector = RegimeDetector('TestRegimeDetector', 'regime_detector')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config
        }
        
        detector.initialize(context)
        
        # Verify initialization
        self.assertEqual(detector.container, self.mock_container)
        self.assertEqual(detector.event_bus, self.mock_event_bus)
        self.assertEqual(detector.subscription_manager, self.mock_subscription_manager)
        self.assertEqual(detector._min_regime_duration, 3)
        self.assertIn('trending_up', detector._regime_thresholds)
        self.assertIn('trending_down', detector._regime_thresholds)
    
    @patch('src.strategy.regime_detector.SimpleMATrendIndicator')
    @patch('src.strategy.regime_detector.RSIIndicator')
    def test_setup(self, mock_rsi_class, mock_ma_class):
        """Test setup method."""
        # Create mock indicators
        mock_ma_indicator = Mock()
        mock_rsi_indicator = Mock()
        mock_ma_class.return_value = mock_ma_indicator
        mock_rsi_class.return_value = mock_rsi_indicator
        
        detector = RegimeDetector('TestRegimeDetector', 'regime_detector')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config
        }
        
        detector.initialize(context)
        detector.setup()
        
        # Verify indicators were created
        self.assertEqual(len(detector._regime_indicators), 2)
        self.assertIn('ma_trend', detector._regime_indicators)
        self.assertIn('rsi', detector._regime_indicators)
    
    def test_classify_initial_state(self):
        """Test classification in initial state."""
        detector = RegimeDetector('TestRegimeDetector', 'regime_detector')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config
        }
        
        detector.initialize(context)
        
        # Test classification without indicators
        data = {'timestamp': '2024-01-01 09:30:00'}
        result = detector.classify(data)
        
        # Should return default initially
        self.assertEqual(result, 'default')
        self.assertEqual(detector._current_classification, 'default')
    
    def test_lifecycle_methods(self):
        """Test start, stop, and teardown methods."""
        detector = RegimeDetector('TestRegimeDetector', 'regime_detector')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config
        }
        
        detector.initialize(context)
        
        # Test start
        detector.start()
        self.assertEqual(detector.state, detector.ComponentState.RUNNING)
        
        # Test stop
        detector.stop()
        self.assertEqual(detector.state, detector.ComponentState.STOPPED)
        
        # Test teardown
        detector.teardown()
        self.assertEqual(len(detector._regime_indicators), 0)
        self.assertEqual(len(detector._regime_thresholds), 0)
    
    def test_reset(self):
        """Test reset method."""
        detector = RegimeDetector('TestRegimeDetector', 'regime_detector')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config
        }
        
        detector.initialize(context)
        
        # Set some state
        detector._current_classification = 'trending_up'
        detector._current_regime_duration = 10
        detector._total_checks = 100
        detector._regime_counts = {'trending_up': 50, 'default': 50}
        
        # Reset
        detector.reset()
        
        # Verify reset
        self.assertIsNone(detector._current_classification)
        self.assertEqual(detector._current_regime_duration, 0)
        self.assertEqual(detector._total_checks, 0)
        self.assertEqual(len(detector._regime_counts), 0)
    
    def test_publish_classification_event(self):
        """Test publishing classification events."""
        detector = RegimeDetector('TestRegimeDetector', 'regime_detector')
        
        # Initialize with context
        context = {
            'container': self.mock_container,
            'event_bus': self.mock_event_bus,
            'subscription_manager': self.mock_subscription_manager,
            'config': self.mock_config
        }
        
        detector.initialize(context)
        
        # Test publishing event
        detector._publish_classification_event('trending_up', '2024-01-01 09:30:00')
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        published_event = self.mock_event_bus.publish.call_args[0][0]
        self.assertEqual(published_event.event_type, EventType.CLASSIFICATION)
        self.assertEqual(published_event.payload['classification'], 'trending_up')
        self.assertEqual(published_event.payload['detector_name'], 'TestRegimeDetector')


if __name__ == '__main__':
    unittest.main()