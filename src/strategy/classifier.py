# src/strategy/classifier.py
from abc import abstractmethod
from typing import Any, Dict, Optional, List 

from ..core.component import BaseComponent
from ..core.event import Event, EventType 

class Classifier(BaseComponent):
    """
    Base class for all market data classifiers.
    
    Classifiers analyze market data and produce categorical labels without
    directly generating trading signals. Examples include regime detectors,
    market state classifiers, and volatility classifiers.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader, 
                event_bus,
                component_config_key: str = None):
        super().__init__(instance_name, config_loader, component_config_key) # Pass config_loader here
        self._event_bus = event_bus
        self._current_classification: Optional[str] = None 
        self._classification_history: List[Dict[str, Any]] = [] 
    
    @abstractmethod
    def classify(self, data: Dict[str, Any]) -> str:
        pass
    
    def get_current_classification(self) -> Optional[str]:
        return self._current_classification
    
    def get_classification_history(self) -> List[Dict[str, Any]]:
        return self._classification_history
    
    def setup(self):
        super().setup() # Call BaseComponent's setup
        self.logger.info(f"Setting up classifier '{self.name}'")
        if self._event_bus:
            self._event_bus.subscribe(EventType.BAR, self.on_bar) 
        # self.state already set by BaseComponent.setup() or should be set here if BaseComponent.setup is abstract
        if self.state != BaseComponent.STATE_INITIALIZED: # Ensure state is set if not by super
            self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"Classifier '{self.name}' setup complete. State: {self.state}. Subscribed to BAR events.")
    
    def on_bar(self, event: Event): 
        self.logger.warning(f"CLASSIFIER_DEBUG: {self.name} received BAR event (state: {self.state})")
        data: Dict[str, Any] = event.payload 
        
        if not isinstance(data, dict):
            self.logger.warning(f"Received BAR event with non-dict payload for {self.name}. Skipping classification.")
            return

        try:
            new_classification = self.classify(data)
        except Exception as e:
            self.logger.error(f"Error during classification in {self.name} for data {data}: {e}", exc_info=True)
            return

        if not isinstance(new_classification, str):
            self.logger.error(f"Classifier {self.name} returned non-string classification: {new_classification}. Skipping update.")
            return
            
        classification_changed = new_classification != self._current_classification
        previous_classification_for_event = self._current_classification
        
        # Update current classification before appending to history for consistency
        self._current_classification = new_classification
        
        timestamp = data.get('timestamp')
        if timestamp is None:
            self.logger.warning(f"Timestamp missing in BAR event data for {self.name}. History record may be incomplete.")

        self._classification_history.append({
            'timestamp': timestamp, 
            'classification': new_classification,
            'changed': classification_changed,
            'bar_data': data # Optionally include the bar data that led to this classification state
        })
        
        # Always publish classification event during optimization to ensure regimes are detected
        # This ensures that even if the classification hasn't changed, the event is published during optimization
        if self._event_bus:
            try:
                # Pass the bar data to _create_classification_event
                classification_event = self._create_classification_event(data, new_classification, previous_classification_for_event)
                self._event_bus.publish(classification_event)
                
                if classification_changed:
                    self.logger.info(f"Classifier '{self.name}' published CLASSIFICATION event: New='{new_classification}', Prev='{previous_classification_for_event}' at {timestamp}")
                else:
                    self.logger.debug(f"Classifier '{self.name}' published repeat CLASSIFICATION event: '{new_classification}' at {timestamp}")
            except Exception as e:
                self.logger.error(f"Error creating or publishing CLASSIFICATION event in {self.name}: {e}", exc_info=True)

    def _create_classification_event(self, bar_data: Dict[str, Any], classification: str, previous_classification: Optional[str]) -> Event:
        """
        Create a classification event.
        
        Args:
            bar_data: The market data (bar) that triggered this classification state.
            classification: The new classification label.
            previous_classification: The classification label before this change.
            
        Returns:
            Event object for the CLASSIFICATION event type.
        """
        timestamp = bar_data.get('timestamp') 
        
        payload = {
            'timestamp': timestamp, # Timestamp of the bar causing the classification
            'classifier_name': self.name,
            'classification': classification, # The new (current) classification
            'previous_classification': previous_classification,
            # --- ADD THE BAR DATA (OR KEY PARTS LIKE PRICE) TO THE PAYLOAD ---
            'bar_close_price': bar_data.get('close'), # Important for P&L segmentation
            'full_bar_data': bar_data # Optional: send the whole bar if needed by consumer
            # ----------------------------------------------------------------
        }
        return Event(EventType.CLASSIFICATION, payload) 
    
    def start(self):
        super().start() 
        if self.state in [BaseComponent.STATE_INITIALIZED, BaseComponent.STATE_STOPPED]: # Check state before forcing
            self.state = BaseComponent.STATE_STARTED
            
            # Ensure we're subscribed to BAR events (needed for restarts)
            if self._event_bus:
                self._event_bus.subscribe(EventType.BAR, self.on_bar)
                self.logger.debug(f"Classifier '{self.name}' re-subscribed to BAR events on start/restart")
                
            self.logger.info(f"Classifier '{self.name}' successfully started/restarted. State: {self.state}")
        elif self.state == BaseComponent.STATE_STARTED:
             self.logger.info(f"Classifier '{self.name}' already started.")
        else:
            self.logger.warning(f"Classifier '{self.name}' was not in expected state (was {self.state}) before attempting to start. Expected INITIALIZED or STOPPED. State not changed by Classifier.start().")

    def stop(self):
        super().stop() # Call parent's stop, it should handle state if it's not abstract
        self.logger.info(f"Stopping classifier '{self.name}'...")
        if self._event_bus and hasattr(self._event_bus, 'unsubscribe'):
             try:
                 self._event_bus.unsubscribe(EventType.BAR, self.on_bar)
                 self.logger.info(f"Classifier '{self.name}' unsubscribed from BAR events.")
             except Exception as e: 
                 self.logger.error(f"Error unsubscribing {self.name} from BAR events: {e}", exc_info=True)
        
        if self.state != BaseComponent.STATE_STOPPED: # Check state before forcing
            self.state = BaseComponent.STATE_STOPPED
            self.logger.info(f"Classifier '{self.name}' stopped. State: {self.state}")
        elif self.state == BaseComponent.STATE_STOPPED:
            self.logger.info(f"Classifier '{self.name}' already stopped.")
