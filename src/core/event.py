# src/core/event.py
from enum import Enum, auto
import datetime # For timestamping events, as suggested in detailed docs

class EventType(Enum):
    """
    Defines the types of events that can occur in the system.
    Based on PLAN.MD Phase 1.
    """
    BAR = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    # Add other generic system events if needed, e.g., SYSTEM_START, SYSTEM_SHUTDOWN
    # For now, keeping it to the trading-specific ones from PLAN.MD

class Event:
    """
    Base class for all events in the system.
    """
    def __init__(self, event_type: EventType, payload: dict = None):
        """
        Initializes an Event.

        Args:
            event_type (EventType): The type of the event.
            payload (dict, optional): A dictionary containing event-specific data.
                                      Defaults to None, which becomes an empty dict.
        """
        self.event_type: EventType = event_type
        self.timestamp: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
        self.payload: dict = payload if payload is not None else {}

    def __str__(self):
        return f"Event(type={self.event_type.name}, ts={self.timestamp}, payload={self.payload})"

    def __repr__(self):
        return f"<Event type={self.event_type.name} payload_keys={list(self.payload.keys())}>"

# Example of specific event classes (optional for MVP, but good practice for clarity)
# If we don't define specific classes, the payload dict needs to be well-documented.
# For MVP, we might just use the base Event class and rely on payload structure.
#
# class BarEvent(Event):
#     def __init__(self, symbol: str, open_price: float, high_price: float, low_price: float, close_price: float, volume: int, bar_timestamp: datetime.datetime):
#         payload = {
#             "symbol": symbol,
#             "open": open_price,
#             "high": high_price,
#             "low": low_price,
#             "close": close_price,
#             "volume": volume,
#             "bar_timestamp": bar_timestamp
#         }
#         super().__init__(EventType.BAR, payload)
