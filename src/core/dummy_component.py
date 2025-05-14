# src/core/dummy_component.py
from .component import BaseComponent
from .exceptions import ComponentError
from .event import Event, EventType # Import Event and EventType
# from .event_bus import EventBus # Not strictly needed for type hint in Python 3.9+ if using string

class DummyComponent(BaseComponent):
    def __init__(self, 
                 instance_name: str, 
                 config_loader,         
                 event_bus,             
                 component_config_key: str = None,
                 listen_to_event_type_str: str = None): # New param to specify event type
        
        super().__init__(
            instance_name=instance_name, 
            config_loader=config_loader, 
            component_config_key=component_config_key
        )
        
        self._event_bus = event_bus
        self.listen_to_event_type: EventType = None
        if listen_to_event_type_str:
            try:
                self.listen_to_event_type = EventType[listen_to_event_type_str.upper()]
            except KeyError:
                self.logger.error(f"Invalid event type string '{listen_to_event_type_str}' for {self.name}. Will not subscribe.")

        self.logger.info(f"DummyComponent '{self.name}' specific initialization. EventBus assigned. Listening to: {self.listen_to_event_type.name if self.listen_to_event_type else 'None'}")

    def _generic_event_handler(self, event: Event):
        self.logger.info(f"'{self.name}' received an event of type '{event.event_type.name}': {event.payload}")

    # Keep specific handlers if needed for more detailed logging or different instances
    def _handle_signal_event(self, event: Event):
        if event.event_type == EventType.SIGNAL:
            self.logger.info(f"'{self.name}' (Signal Consumer) received SIGNAL: {event.payload}")

    def _handle_order_event(self, event: Event):
        if event.event_type == EventType.ORDER:
            self.logger.info(f"'{self.name}' (Order/Fill Logger) received ORDER: {event.payload}")

    def _handle_fill_event(self, event: Event):
        if event.event_type == EventType.FILL:
            self.logger.info(f"'{self.name}' (Order/Fill Logger) received FILL: {event.payload}")

    def setup(self):
        self.logger.info(f"DummyComponent '{self.name}' performing its specific setup.")
        dummy_setting = self.get_specific_config("some_setting", "default_dummy_value")
        self.logger.info(f"DummyComponent specific setting 'some_setting': {dummy_setting}")

        if self._event_bus and self.listen_to_event_type:
            handler_map = {
                EventType.SIGNAL: self._handle_signal_event,
                EventType.ORDER: self._handle_order_event,
                EventType.FILL: self._handle_fill_event,
                EventType.BAR: self._generic_event_handler # Example for BAR if needed
            }
            handler_to_subscribe = handler_map.get(self.listen_to_event_type)
            
            if handler_to_subscribe:
                self._event_bus.subscribe(self.listen_to_event_type, handler_to_subscribe)
                self.logger.info(f"'{self.name}' subscribed to {self.listen_to_event_type.name} events.")
            else:
                self.logger.warning(f"'{self.name}' no specific handler for {self.listen_to_event_type.name}, will not subscribe with default generic handler unless explicitly chosen.")
        elif self._event_bus and not self.listen_to_event_type:
             self.logger.info(f"'{self.name}' not configured to listen to a specific event type during setup.")
        else:
            self.logger.warning(f"'{self.name}' has no event bus or no event type specified, cannot subscribe to events.")
            
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"DummyComponent '{self.name}' setup complete. State: {self.state}")

    def start(self):
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start DummyComponent '{self.name}' from state '{self.state}'. Expected INITIALIZED.")
            return
        self.logger.info(f"DummyComponent '{self.name}' started. Listening for configured events...")
        self.state = BaseComponent.STATE_STARTED

    def stop(self):
        self.logger.info(f"DummyComponent '{self.name}' performing its specific stop actions.")
        if self._event_bus and self.listen_to_event_type:
            handler_map = {
                EventType.SIGNAL: self._handle_signal_event,
                EventType.ORDER: self._handle_order_event,
                EventType.FILL: self._handle_fill_event,
                EventType.BAR: self._generic_event_handler
            }
            handler_to_unsubscribe = handler_map.get(self.listen_to_event_type)
            if handler_to_unsubscribe:
                try:
                    self._event_bus.unsubscribe(self.listen_to_event_type, handler_to_unsubscribe)
                    self.logger.info(f"'{self.name}' attempted to unsubscribe from {self.listen_to_event_type.name} events.")
                except Exception as e: 
                    self.logger.error(f"Error unsubscribing '{self.name}' from {self.listen_to_event_type.name} events: {e}")
        
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"DummyComponent '{self.name}' stopped. State: {self.state}")
