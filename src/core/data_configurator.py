#!/usr/bin/env python3
"""
DataConfigurator - Centralized data configuration logic for backtests.

This class ensures consistent data configuration across different execution paths
(optimization vs standard backtest), preventing issues with operation ordering.
"""

from typing import Optional, Dict, Any
import logging


class DataConfigurator:
    """
    Centralizes data handler configuration logic to ensure consistency
    between optimization and standard backtest runs.
    
    Key responsibilities:
    1. Apply max_bars limit
    2. Apply train/test split
    3. Set active dataset
    
    IMPORTANT: Operations must be applied in this order:
    1. max_bars FIRST (to limit the data)
    2. train/test split SECOND (to split the limited data)
    3. set active dataset LAST (to select train/test/full)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the data configurator."""
        self.logger = logger or logging.getLogger(__name__)
        
    def configure(self, 
                  data_handler: Any,
                  max_bars: Optional[int] = None,
                  train_test_split_ratio: Optional[float] = None,
                  dataset: Optional[str] = None,
                  use_test_dataset: Optional[bool] = None) -> None:
        """
        Configure the data handler with consistent ordering.
        
        Args:
            data_handler: The data handler to configure
            max_bars: Maximum number of bars to use (applied FIRST)
            train_test_split_ratio: Ratio for train/test split (applied SECOND)
            dataset: Dataset to use ('train', 'test', 'full', or None)
            use_test_dataset: Legacy flag for test dataset (used if dataset is None)
        """
        # Step 1: Apply max_bars limit FIRST
        if max_bars and hasattr(data_handler, 'set_max_bars'):
            self.logger.info(f"DataConfigurator: Limiting to {max_bars} bars BEFORE train/test split")
            data_handler.set_max_bars(max_bars)
            
        # Step 2: Apply train/test split SECOND (if not already applied)
        if train_test_split_ratio and train_test_split_ratio > 0:
            # CRITICAL: Check if we need to reapply the split after bars limit
            # The data handler may have already split the FULL dataset during setup,
            # but we need to split the LIMITED dataset instead
            needs_split = True
            
            # Check if data handler has the correct split (after bars limit)
            if (hasattr(data_handler, '_train_test_split_ratio') and 
                data_handler._train_test_split_ratio == train_test_split_ratio):
                # Check if the split was applied to the correct dataset size
                if (hasattr(data_handler, '_train_df') and hasattr(data_handler, '_test_df') and
                    data_handler._train_df is not None and data_handler._test_df is not None):
                    total_split_size = len(data_handler._train_df) + len(data_handler._test_df)
                    
                    # If max_bars was set, check if split size matches
                    if max_bars and max_bars > 0:
                        if total_split_size == max_bars:
                            self.logger.info(f"DataConfigurator: Train/test split already correctly applied to {max_bars} bars")
                            needs_split = False
                        else:
                            self.logger.warning(
                                f"DataConfigurator: Split exists but for wrong dataset size "
                                f"(current: {total_split_size}, expected: {max_bars}). Reapplying split."
                            )
                    else:
                        # No bars limit, existing split is fine
                        self.logger.info("DataConfigurator: Train/test split already applied")
                        needs_split = False
                    
            if needs_split and hasattr(data_handler, 'apply_train_test_split'):
                self.logger.info(f"DataConfigurator: Applying train/test split with ratio {train_test_split_ratio}")
                data_handler.apply_train_test_split(train_test_split_ratio)
        
        # Step 3: Set active dataset LAST
        if hasattr(data_handler, 'set_active_dataset'):
            # Determine which dataset to use
            target_dataset = self._determine_target_dataset(
                data_handler, dataset, use_test_dataset
            )
            
            self.logger.info(f"DataConfigurator: Setting active dataset to '{target_dataset}'")
            data_handler.set_active_dataset(target_dataset)
            
    def _determine_target_dataset(self,
                                 data_handler: Any,
                                 dataset: Optional[str],
                                 use_test_dataset: Optional[bool]) -> str:
        """
        Determine which dataset to use based on configuration and availability.
        
        Args:
            data_handler: The data handler
            dataset: Explicit dataset choice ('train', 'test', 'full', or None)
            use_test_dataset: Legacy flag (used if dataset is None)
            
        Returns:
            The dataset to use ('train', 'test', or 'full')
        """
        # Check if train/test split is available
        has_split = False
        if (hasattr(data_handler, 'train_df_exists_and_is_not_empty') and 
            hasattr(data_handler, 'test_df_exists_and_is_not_empty')):
            has_split = (data_handler.train_df_exists_and_is_not_empty and 
                        data_handler.test_df_exists_and_is_not_empty)
        
        # If explicit dataset is specified
        if dataset:
            if dataset in ['train', 'test'] and not has_split:
                self.logger.warning(
                    f"DataConfigurator: Requested dataset '{dataset}' but no train/test split available, "
                    f"using 'full' dataset"
                )
                return 'full'
            return dataset
            
        # Use legacy flag if no explicit dataset
        if use_test_dataset is not None and has_split:
            return 'test' if use_test_dataset else 'train'
            
        # Default to full dataset
        return 'full'
        
    def ensure_train_test_split(self,
                               data_handler: Any,
                               train_test_split_ratio: float) -> None:
        """
        Ensure train/test split is applied to the data handler.
        
        This is useful when you need to guarantee the split is applied,
        regardless of the data handler's current state.
        
        Args:
            data_handler: The data handler
            train_test_split_ratio: The split ratio to apply
        """
        if hasattr(data_handler, 'apply_train_test_split'):
            self.logger.info(f"DataConfigurator: Ensuring train/test split with ratio {train_test_split_ratio}")
            data_handler.apply_train_test_split(train_test_split_ratio)
        else:
            self.logger.warning("DataConfigurator: Data handler does not support train/test split")