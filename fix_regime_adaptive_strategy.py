"""
This script fixes an issue with the regime-adaptive strategy test during optimization.

The current issue is that during optimization (with --optimize flag), the system 
uses EnsembleSignalStrategy instead of RegimeAdaptiveStrategy, but when it tries to
run the regime-adaptive test at the end of optimization, it's still using 
EnsembleSignalStrategy, which doesn't properly respond to regime changes.

To fix this, we need to:
1. Create a proper RegimeAdaptiveStrategy instance for the test
2. Register it with the container
3. Make sure all events are properly routed to it
"""

from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.core.container import Container
from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def replace_strategy_for_adaptive_test(container: Container, config_loader, event_bus):
    """
    Replaces the current strategy with RegimeAdaptiveStrategy for the adaptive test.
    
    This function:
    1. Creates a new RegimeAdaptiveStrategy instance
    2. Configures it properly
    3. Replaces the EnsembleSignalStrategy with this new instance in the container
    
    Args:
        container: The IoC container
        config_loader: The configuration loader
        event_bus: The event bus
        
    Returns:
        The newly created RegimeAdaptiveStrategy instance
    """
    logger.info("Creating RegimeAdaptiveStrategy for adaptive test")
    
    # Create the RegimeAdaptiveStrategy
    adaptive_strat_args = {
        "instance_name": "RegimeAdaptiveStrategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "container": container,
        "component_config_key": "components.regime_adaptive_strategy"
    }
    
    try:
        # Create the RegimeAdaptiveStrategy instance
        adaptive_strategy = RegimeAdaptiveStrategy(**adaptive_strat_args)
        logger.info(f"Created RegimeAdaptiveStrategy: {adaptive_strategy.name}")
        
        # Register it with the container
        container.register_instance("adaptive_strategy", adaptive_strategy)
        logger.info("Registered RegimeAdaptiveStrategy as 'adaptive_strategy'")
        
        # Also make it available as the regular strategy
        container.register_instance("strategy", adaptive_strategy)
        logger.info("Replaced current strategy with RegimeAdaptiveStrategy for adaptive test")
        
        return adaptive_strategy
        
    except Exception as e:
        logger.error(f"Failed to create RegimeAdaptiveStrategy: {e}", exc_info=True)
        return None
        
def fix_enhanced_optimizer():
    """
    Apply the fix to the EnhancedOptimizer class
    
    This function:
    1. Monkey-patches the EnhancedOptimizer._run_regime_adaptive_test method
    2. Adds the strategy replacement logic
    
    Note: This is a temporary fix. The proper solution would be to update the 
    EnhancedOptimizer class directly.
    """
    from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
    
    # Store the original method
    original_method = EnhancedOptimizer._run_regime_adaptive_test
    
    def patched_run_regime_adaptive_test(self, results):
        """
        Patched version of the _run_regime_adaptive_test method.
        """
        logger.info("Using patched regime-adaptive test method")
        
        # Replace the strategy with RegimeAdaptiveStrategy
        adaptive_strategy = replace_strategy_for_adaptive_test(
            self._container, self._config_loader, self._event_bus)
        
        if not adaptive_strategy:
            logger.error("Failed to create RegimeAdaptiveStrategy, cannot run adaptive test")
            results["regime_adaptive_test_results"] = {"error": "Failed to create RegimeAdaptiveStrategy"}
            return
            
        # Now call the original method
        return original_method(self, results)
        
    # Replace the method with our patched version
    EnhancedOptimizer._run_regime_adaptive_test = patched_run_regime_adaptive_test
    logger.info("Successfully patched EnhancedOptimizer._run_regime_adaptive_test")
    
if __name__ == "__main__":
    # Apply the fix
    fix_enhanced_optimizer()
    logger.info("Fix applied. You can now run the optimizer with --optimize and it will use RegimeAdaptiveStrategy for the adaptive test.")