# src/core/dummy_component.py
from typing import Optional, Dict, Any
import logging
from .component_base import ComponentBase, ComponentState
from .event import Event, EventType

class DummyComponent(ComponentBase):
    """
    A dummy component for testing and debugging.
    Can be configured to listen to specific event types.
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Component-specific attributes initialized later
        self._event_bus = None
        self._listen_to_event_type: Optional[EventType] = None
        self._subscribed_handler = None
        
    def _initialize(self):
        """Component-specific initialization logic."""
        # Get event bus from context
        self._event_bus = self._context.get('event_bus')
        if not self._event_bus:
            self.logger.warning(f"{self.name}: No event bus provided in context")
            
        # Get event type to listen to from config
        listen_to_event_type_str = self.get_specific_config("listen_to_event_type")
        if listen_to_event_type_str:
            try:
                self._listen_to_event_type = EventType[listen_to_event_type_str.upper()]
                self.logger.info(f"{self.name}: Configured to listen to {self._listen_to_event_type.name} events")
            except KeyError:
                self.logger.error(f"{self.name}: Invalid event type '{listen_to_event_type_str}'")
                
        # Log dummy setting
        dummy_setting = self.get_specific_config("some_setting", "default_dummy_value")
        self.logger.info(f"{self.name}: Setting 'some_setting' = {dummy_setting}")
        
    def _start(self):
        """Start component - subscribe to events."""
        if self._event_bus and self._listen_to_event_type:
            # Select appropriate handler
            handler_map = {
                EventType.SIGNAL: self._handle_signal_event,
                EventType.ORDER: self._handle_order_event,
                EventType.FILL: self._handle_fill_event,
                EventType.BAR: self._generic_event_handler
            }
            
            self._subscribed_handler = handler_map.get(
                self._listen_to_event_type, 
                self._generic_event_handler
            )
            
            self._event_bus.subscribe(self._listen_to_event_type, self._subscribed_handler)
            self.logger.info(f"{self.name}: Subscribed to {self._listen_to_event_type.name} events")
            
    def _stop(self):
        """Stop component - unsubscribe from events."""
        if self._event_bus and self._listen_to_event_type and self._subscribed_handler:
            try:
                self._event_bus.unsubscribe(self._listen_to_event_type, self._subscribed_handler)
                self.logger.info(f"{self.name}: Unsubscribed from {self._listen_to_event_type.name} events")
            except Exception as e:
                self.logger.error(f"{self.name}: Error unsubscribing from events: {e}")
                
    def _generic_event_handler(self, event: Event):
        """Generic event handler for any event type."""
        self.logger.info(f"{self.name}: Received {event.event_type.name} event: {event.payload}")
        
    def _handle_signal_event(self, event: Event):
        """Specific handler for SIGNAL events."""
        if event.event_type == EventType.SIGNAL:
            self.logger.info(f"{self.name} (Signal Consumer): Received SIGNAL: {event.payload}")
            
    def _handle_order_event(self, event: Event):
        """Specific handler for ORDER events."""
        if event.event_type == EventType.ORDER:
            self.logger.info(f"{self.name} (Order Logger): Received ORDER: {event.payload}")
            
    def _handle_fill_event(self, event: Event):
        """Specific handler for FILL events."""
        if event.event_type == EventType.FILL:
            self.logger.info(f"{self.name} (Fill Logger): Received FILL: {event.payload}")