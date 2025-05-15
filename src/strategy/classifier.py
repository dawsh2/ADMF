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
        """
        Initialize the Classifier.
        
        Args:
            instance_name (str): The unique name of this classifier instance.
            config_loader: Configuration loader instance.
            event_bus: Event bus for publishing classification events and subscribing to data.
            component_config_key (str, optional): Configuration key for this component.
        """
        super().__init__(instance_name, config_loader, component_config_key)
        self._event_bus = event_bus
        self._current_classification: Optional[str] = None 
        self._classification_history: List[Dict[str, Any]] = [] 
    
    @abstractmethod
    def classify(self, data: Dict[str, Any]) -> str:
        """
        Classify market data into a categorical label.
        
        This method must be implemented by subclasses.
        
        Args:
            data: Market data (e.g., a bar) to classify.
            
        Returns:
            str: Classification label (e.g., "trending_up", "high_volatility").
        """
        pass
    
    def get_current_classification(self) -> Optional[str]:
        """
        Returns the current classification label.
        
        Returns:
            str: Current classification label or None if not yet classified.
        """
        return self._current_classification
    
    def get_classification_history(self) -> List[Dict[str, Any]]:
        """
        Returns the history of classifications with timestamps and change flags.
        
        Returns:
            list: List of classification records. Each record is a dictionary
                  containing 'timestamp', 'classification', and 'changed' (bool).
        """
        return self._classification_history
    
    def setup(self):
        """
        Set up classifier resources.
        This typically involves subscribing to relevant data events (e.g., BAR events).
        """
        self.logger.info(f"Setting up classifier '{self.name}'")
        if self._event_bus:
            self._event_bus.subscribe(EventType.BAR, self.on_bar) 
        self.state = BaseComponent.STATE_INITIALIZED # Correctly set by BaseComponent or here
        self.logger.info(f"Classifier '{self.name}' initialized and subscribed to BAR events.")
    
    def on_bar(self, event: Event): 
        """
        Process bar event, update classification, and publish if changed.
        
        Args:
            event: Bar event containing market data.
        """
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
        self._current_classification = new_classification
        
        timestamp = data.get('timestamp')
        if timestamp is None:
            self.logger.warning(f"Timestamp missing in BAR event data for {self.name}. History record may be incomplete.")

        self._classification_history.append({
            'timestamp': timestamp, 
            'classification': new_classification,
            'changed': classification_changed
        })
        
        if classification_changed and self._event_bus:
            try:
                classification_event = self._create_classification_event(data, new_classification, previous_classification_for_event)
                self._event_bus.publish(classification_event)
                self.logger.debug(f"Classifier '{self.name}' published CLASSIFICATION event: New='{new_classification}', Prev='{previous_classification_for_event}'")
            except Exception as e:
                self.logger.error(f"Error creating or publishing CLASSIFICATION event in {self.name}: {e}", exc_info=True)

    def _create_classification_event(self, data: Dict[str, Any], classification: str, previous_classification: Optional[str]) -> Event:
        """
        Create a classification event.
        """
        timestamp = data.get('timestamp') 
        payload = {
            'timestamp': timestamp,
            'classifier_name': self.name,
            'classification': classification,
            'previous_classification': previous_classification
        }
        # Ensure EventType.CLASSIFICATION exists in your src.core.event.EventType enum
        return Event(EventType.CLASSIFICATION, payload) 
    
    def start(self):
        """Start the classifier."""
        # Call the abstract parent's start (which just logs in your case)
        super().start() # Calls BaseComponent.start() which logs "Starting component..."
        
        # Explicitly set the state for this component if parent doesn't
        if self.state == BaseComponent.STATE_INITIALIZED:
            self.state = BaseComponent.STATE_STARTED
            self.logger.info(f"Classifier '{self.name}' successfully started. State: {self.state}")
        else:
            self.logger.warning(f"Classifier '{self.name}' was not in INITIALIZED state (was {self.state}) before attempting to start. State not changed by Classifier.start().")

    def stop(self):
        """Stop the classifier."""
        # Call the abstract parent's stop (if it has one, or implement logic here)
        # super().stop() # Uncomment if BaseComponent has a concrete stop or if it's also abstract
        
        self.logger.info(f"Stopping classifier '{self.name}'...")
        # Add any specific cleanup for the classifier here, e.g., unsubscribing
        if self._event_bus and hasattr(self._event_bus, 'unsubscribe'):
             try:
                 self._event_bus.unsubscribe(EventType.BAR, self.on_bar)
                 self.logger.info(f"Classifier '{self.name}' unsubscribed from BAR events.")
             except Exception as e: # Be specific about expected exceptions if possible
                 self.logger.error(f"Error unsubscribing {self.name} from BAR events: {e}", exc_info=True)

        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"Classifier '{self.name}' stopped. State: {self.state}")
        # If BaseComponent.stop() is abstract and meant to be overridden,
        # ensure super().stop() is called if it exists and does something.
        # If BaseComponent.stop() is concrete and sets state, call it at the end.
        # For now, assuming Classifier handles its own state transition to STOPPED here.

