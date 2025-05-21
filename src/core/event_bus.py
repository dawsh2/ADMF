# src/core/event_bus.py
import logging
import threading
from collections import defaultdict
from typing import Callable, List

from .event import Event, EventType # Import your Event and EventType

logger = logging.getLogger(__name__)

class EventBus:
    """
    A simple, thread-safe event bus for dispatching events to subscribed listeners.
    """
    def __init__(self):
        self._subscribers: defaultdict[EventType, List[Callable[[Event], None]]] = defaultdict(list)
        self._lock: threading.Lock = threading.Lock()
        logger.info("EventBus initialized.")

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """
        Subscribes a handler to a specific event type.

        Args:
            event_type (EventType): The type of event to subscribe to.
            handler (Callable[[Event], None]): The function/method to call when the event occurs.
                                              It should accept a single argument: the event object.
        """
        if not callable(handler):
            logger.error(f"Attempted to subscribe with a non-callable handler for event type {event_type.name}.")
            return

        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(f"Handler '{getattr(handler, '__name__', repr(handler))}' subscribed to event type '{event_type.name}'.")
            else:
                logger.debug(f"Handler '{getattr(handler, '__name__', repr(handler))}' already subscribed to event type '{event_type.name}'. Skipping duplicate subscription.")

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """
        Unsubscribes a handler from a specific event type.

        Args:
            event_type (EventType): The type of event to unsubscribe from.
            handler (Callable[[Event], None]): The handler to remove.
        """
        # Use a debug-level log for not found handlers to reduce noise
        with self._lock:
            try:
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)
                    logger.debug(f"Handler '{getattr(handler, '__name__', repr(handler))}' unsubscribed from event type '{event_type.name}'.")
                else:
                    logger.debug(f"Handler '{getattr(handler, '__name__', repr(handler))}' not found for event type '{event_type.name}' during unsubscribe.")
            except ValueError: # Should not happen if 'in' check is done, but as a safeguard
                logger.debug(f"Handler '{getattr(handler, '__name__', repr(handler))}' not found for event type '{event_type.name}' during unsubscribe (ValueError).")
            except KeyError: # EventType not in subscribers
                logger.debug(f"No subscribers found for event type '{event_type.name}' during unsubscribe.")


    def publish(self, event: Event):
        """
        Publishes an event to all subscribed handlers for its type.

        Args:
            event (Event): The event object to publish.
        """
        if not isinstance(event, Event):
            logger.error(f"Attempted to publish a non-Event object: {event}")
            return

        logger.debug(f"Publishing event: {event}")

        handlers_to_call: List[Callable[[Event], None]] = []
        with self._lock:
            # Iterate over a copy of the list of handlers to avoid issues if a handler
            # tries to subscribe/unsubscribe during event dispatch.
            handlers_to_call = list(self._subscribers.get(event.event_type, []))

        if not handlers_to_call:
            logger.debug(f"No handlers subscribed for event type '{event.event_type.name}'.")
            return

        for handler in handlers_to_call:
            try:
                # logger.debug(f"Dispatching event type '{event.event_type.name}' to handler '{getattr(handler, '__name__', repr(handler))}'.")
                handler(event)
            except Exception as e:
                logger.error(f"Error in handler '{getattr(handler, '__name__', repr(handler))}' for event '{event.event_type.name}': {e}", exc_info=True)
                # Depending on policy, you might want to unregister faulty handlers
                # or implement more sophisticated error handling for handlers.
