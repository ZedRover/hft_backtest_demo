"""
Event logging system.

Provides detailed audit trail of backtest execution.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class EventType(Enum):
    """Event types for audit trail"""
    ORDER_SUBMITTED = "order_submitted"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    POSITION_UPDATED = "position_updated"
    STRATEGY_ACTION = "strategy_action"
    QUEUE_UPDATED = "queue_updated"
    SNAPSHOT_PROCESSED = "snapshot_processed"


@dataclass
class Event:
    """
    Backtest event for audit trail.

    Captures all significant events during backtest execution.
    """
    timestamp: int  # milliseconds
    event_type: EventType
    description: str
    data: Optional[dict] = None


class EventLogger:
    """
    Logs events during backtest execution.

    Provides both Python logging and detailed event tracking.
    """

    def __init__(self, log_level: str = "INFO", save_events: bool = False):
        """
        Initialize event logger.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            save_events: Whether to save detailed event history
        """
        self.save_events = save_events
        self.events: List[Event] = [] if save_events else []

        # Configure Python logger
        self.logger = logging.getLogger("hft_backtest")
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_event(
        self,
        timestamp: int,
        event_type: EventType,
        description: str,
        data: Optional[dict] = None
    ) -> None:
        """
        Log an event.

        Args:
            timestamp: Event timestamp
            event_type: Type of event
            description: Event description
            data: Optional event data
        """
        if self.save_events:
            event = Event(
                timestamp=timestamp,
                event_type=event_type,
                description=description,
                data=data
            )
            self.events.append(event)

        # Log to Python logger
        self.logger.debug(f"[{timestamp}] {event_type.value}: {description}")

    def log_order_submitted(
        self,
        timestamp: int,
        order_id: int,
        side: int,
        price: float,
        quantity: float
    ) -> None:
        """Log order submission"""
        side_str = "BUY" if side == 1 else "SELL"
        description = f"Order {order_id} submitted: {side_str} {quantity} @ {price}"
        self.log_event(
            timestamp,
            EventType.ORDER_SUBMITTED,
            description,
            {"order_id": order_id, "side": side, "price": price, "quantity": quantity}
        )

    def log_order_filled(
        self,
        timestamp: int,
        order_id: int,
        fill_price: float,
        fill_quantity: float
    ) -> None:
        """Log order fill"""
        description = f"Order {order_id} filled: {fill_quantity} @ {fill_price}"
        self.log_event(
            timestamp,
            EventType.ORDER_FILLED,
            description,
            {"order_id": order_id, "fill_price": fill_price, "fill_quantity": fill_quantity}
        )

    def log_order_cancelled(self, timestamp: int, order_id: int) -> None:
        """Log order cancellation"""
        description = f"Order {order_id} cancelled"
        self.log_event(timestamp, EventType.ORDER_CANCELLED, description, {"order_id": order_id})

    def log_order_rejected(self, timestamp: int, order_id: int, reason: str) -> None:
        """Log order rejection"""
        description = f"Order {order_id} rejected: {reason}"
        self.log_event(
            timestamp,
            EventType.ORDER_REJECTED,
            description,
            {"order_id": order_id, "reason": reason}
        )

    def log_position_updated(
        self,
        timestamp: int,
        quantity: float,
        avg_entry: float,
        realized_pnl: float
    ) -> None:
        """Log position update"""
        description = f"Position updated: qty={quantity:.2f}, entry={avg_entry:.2f}, pnl={realized_pnl:.2f}"
        self.log_event(
            timestamp,
            EventType.POSITION_UPDATED,
            description,
            {"quantity": quantity, "avg_entry": avg_entry, "realized_pnl": realized_pnl}
        )

    def log_queue_updated(
        self,
        timestamp: int,
        order_id: int,
        queue_position: int,
        volume_ahead: float
    ) -> None:
        """Log queue position update"""
        description = f"Queue updated for order {order_id}: pos={queue_position}, vol_ahead={volume_ahead:.2f}"
        self.log_event(
            timestamp,
            EventType.QUEUE_UPDATED,
            description,
            {"order_id": order_id, "queue_position": queue_position, "volume_ahead": volume_ahead}
        )

    def log_snapshot_processed(self, timestamp: int, snapshot_num: int) -> None:
        """Log snapshot processing"""
        if snapshot_num % 1000 == 0:  # Log every 1000 snapshots
            self.logger.info(f"Processed {snapshot_num} snapshots (timestamp: {timestamp})")

    def get_events(self) -> List[Event]:
        """Get all logged events"""
        return self.events if self.save_events else []

    def clear_events(self) -> None:
        """Clear event history"""
        if self.save_events:
            self.events.clear()