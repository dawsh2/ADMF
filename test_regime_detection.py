#!/usr/bin/env python3
# test_regime_detection.py
import argparse
import logging
import sys
from typing import Dict, Any, List

# Core imports
from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.event_bus import EventBus
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.regime_detector import RegimeDetector

def main():
    parser = argparse.ArgumentParser(description="Test Regime Detection")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--bars", type=int, default=1000, help="Number of bars to process. Default is 1000."
    )
    args = parser.parse_args()
    config_path = args.config
    max_bars_to_process = args.bars

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger("test_regime_detection")
    
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config_loader = SimpleConfigLoader(config_file_path=config_path)
    setup_logging(config_loader)
    
    # Create event bus
    event_bus = EventBus()
    
    # Initialize data handler
    logger.info("Initializing data handler")
    data_handler = CSVDataHandler(
        instance_name="SPY_CSV_Loader",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.data_handler_csv",
        max_bars=max_bars_to_process
    )
    
    # Initialize regime detector
    logger.info("Initializing regime detector")
    regime_detector = RegimeDetector(
        instance_name="TestRegimeDetector",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.MyPrimaryRegimeDetector"
    )
    
    # Setup and start components
    data_handler.setup()
    regime_detector.setup()
    regime_detector.start()
    
    # Process bars and track regimes
    logger.info("Beginning regime detection test")
    regime_counts: Dict[str, int] = {}
    regime_transitions: List[Dict[str, Any]] = []
    last_regime = None
    
    # Set active dataset to "full"
    logger.info("Setting active dataset to 'full'")
    data_handler.set_active_dataset("full")
    
    # We need to manually process each bar since we don't want to use the event bus
    if data_handler._active_df is not None and not data_handler._active_df.empty:
        bar_count = 0
        current_regime = None
        
        # Reset data iterator to make sure we start from the beginning
        data_handler._data_iterator = data_handler._active_df.iterrows()
        
        # Process each bar
        for index, row in data_handler._active_df.iterrows():
            bar_count += 1
            
            # Create bar data
            bar_data = {
                "symbol": data_handler._symbol,
                "timestamp": row[data_handler._timestamp_column]
            }
            
            # Add OHLCV data
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in row:
                    bar_data[col.lower()] = row[col]
            
            # Classify the current regime for this bar
            current_regime = regime_detector.classify(bar_data)
            
            # Track regime statistics
            if current_regime:
                regime_counts[current_regime] = regime_counts.get(current_regime, 0) + 1
                
                # Record regime transitions
                if last_regime is not None and last_regime != current_regime:
                    transition = {
                        "bar_index": bar_count,
                        "timestamp": bar_data.get("timestamp"),
                        "from_regime": last_regime,
                        "to_regime": current_regime,
                        "indicator_values": regime_detector.get_regime_data().get("indicators", {})
                    }
                    regime_transitions.append(transition)
                    logger.info(f"Regime transition at bar {bar_count}: {last_regime} -> {current_regime}")
                    logger.info(f"Indicator values: {transition['indicator_values']}")
                
                last_regime = current_regime
                
            # Log progress every 100 bars
            if bar_count % 100 == 0:
                logger.info(f"Processed {bar_count} bars. Current regime: {current_regime}")
    else:
        logger.error("No data available in the active dataset")
        bar_count = 0
    
    # Print regime statistics
    logger.info("\n---- REGIME DETECTION RESULTS ----")
    logger.info(f"Processed {bar_count} bars")
    logger.info(f"Regime distribution:")
    for regime, count in regime_counts.items():
        percentage = (count / bar_count) * 100 if bar_count > 0 else 0
        logger.info(f"  - {regime}: {count} bars ({percentage:.2f}%)")
    
    logger.info(f"\nDetected {len(regime_transitions)} regime transitions:")
    for idx, transition in enumerate(regime_transitions, 1):
        logger.info(f"  {idx}. Bar {transition['bar_index']} ({transition['timestamp']}): {transition['from_regime']} -> {transition['to_regime']}")
    
    # Clean up
    data_handler.stop()
    regime_detector.stop()
    logger.info("Test completed")

if __name__ == "__main__":
    main()