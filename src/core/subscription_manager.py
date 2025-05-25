#!/usr/bin/env python3
"""
Subscription manager for managing event subscriptions.

Based on EVENT_SYSTEM.md, this class helps components manage their
event subscriptions in a clean and trackable way.
"""

from typing import List, Tuple, Any, Optional


class SubscriptionManager:
    """
    Manages event subscriptions for a component.
    
    This class tracks all subscriptions made by a component and provides
    a clean way to unsubscribe all handlers during component teardown.
    """
    
    def __init__(self, event_bus):
        """
        Initialize with event bus.
        
        Args:
            event_bus: The event bus instance to subscribe to
        """
        self.event_bus = event_bus
        self.subscriptions: List[Tuple[Any, Any, Optional[Any]]] = []
        
    def subscribe(self, event_type: Any, handler: Any, context: Optional[Any] = None) -> 'SubscriptionManager':
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Function to call when events occur
            context: Optional context to associate with the subscription
            
        Returns:
            self: For method chaining
        """
        self.event_bus.subscribe(event_type, handler)
        self.subscriptions.append((event_type, handler, context))
        return self
        
    def unsubscribe(self, event_type: Any, handler: Any, context: Optional[Any] = None) -> bool:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler to unsubscribe
            context: Optional context associated with the subscription
            
        Returns:
            bool: Whether the handler was successfully unsubscribed
        """
        result = self.event_bus.unsubscribe(event_type, handler)
        if result:
            try:
                self.subscriptions.remove((event_type, handler, context))
            except ValueError:
                # Subscription not found in our list, but was removed from event bus
                pass
        return result
        
    def unsubscribe_all(self) -> None:
        """
        Unsubscribe from all subscribed events.
        
        This method iterates through all tracked subscriptions and
        unsubscribes each one from the event bus.
        """
        # Make a copy to avoid modifying list while iterating
        subscriptions_copy = self.subscriptions.copy()
        
        for event_type, handler, context in subscriptions_copy:
            self.event_bus.unsubscribe(event_type, handler)
            
        # Clear the subscriptions list
        self.subscriptions.clear()
        
    def get_subscription_count(self) -> int:
        """
        Get the number of active subscriptions.
        
        Returns:
            int: Number of subscriptions being managed
        """
        return len(self.subscriptions)
        
    def has_subscription(self, event_type: Any, handler: Any = None) -> bool:
        """
        Check if a subscription exists.
        
        Args:
            event_type: Event type to check
            handler: Optional specific handler to check for
            
        Returns:
            bool: Whether a matching subscription exists
        """
        for sub_type, sub_handler, _ in self.subscriptions:
            if sub_type == event_type:
                if handler is None or sub_handler == handler:
                    return True
        return False