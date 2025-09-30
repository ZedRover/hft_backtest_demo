"""
Protocol definitions for core backtesting components.

Defines abstract interfaces that enable dependency injection and allow
components to be easily mocked, tested, and replaced.
"""

from typing import Protocol, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass


@dataclass
class FillResult:
    """Result of a fill check operation."""
    fill_price: float
    fill_quantity: float


class OrderBookProtocol(Protocol):
    """
    Protocol for order book management.
    
    Maintains clean market state by separating our orders from market volumes.
    """
    
    def update_from_snapshot(self, snapshot: np.ndarray) -> None:
        """
        Update order book from market snapshot.
        
        Args:
            snapshot: Market snapshot containing bid/ask levels
        """
        ...
    
    def add_our_order(
        self,
        order_id: int,
        price: float,
        quantity: float,
        side: int
    ) -> Tuple[float, float]:
        """
        Add our order to the book.
        
        Args:
            order_id: Unique order identifier
            price: Order limit price
            quantity: Order quantity
            side: 1 for bid, -1 for ask
            
        Returns:
            Tuple of (volume_ahead, queue_position)
        """
        ...
    
    def remove_our_order(self, order_id: int, quantity: float) -> None:
        """
        Remove our order from the book.
        
        Args:
            order_id: Order to remove
            quantity: Quantity to remove (for partial fills)
        """
        ...
    
    def get_market_volume_at_price(self, price: float, side: int) -> float:
        """
        Get market volume at price level (excluding our orders).
        
        Args:
            price: Price level
            side: 1 for bid, -1 for ask
            
        Returns:
            Market volume at that price
        """
        ...
    
    def update_order_queue_position(self, order_id: int, new_position: float) -> None:
        """
        Update an order's queue position.
        
        Args:
            order_id: Order to update
            new_position: New volume ahead in queue
        """
        ...


class QueueSimulatorProtocol(Protocol):
    """
    Protocol for queue position simulation.
    
    Tracks and updates queue positions based on market activity.
    """
    
    def update_queue_position_for_order(
        self,
        order: np.ndarray,
        snapshot: np.ndarray,
        cancellation_rate: float,
        volume_traded: float = 0.0
    ) -> Tuple[int, float]:
        """
        Update queue position for a specific order.
        
        Args:
            order: Order record (structured array element)
            snapshot: Current market snapshot
            cancellation_rate: Probability of cancellation per snapshot
            volume_traded: Volume traded at order's price level
            
        Returns:
            Tuple of (new_queue_position, new_volume_ahead)
        """
        ...


class FillSimulatorProtocol(Protocol):
    """
    Protocol for order fill simulation.
    
    Determines when and how orders get filled based on market state.
    """
    
    def check_fill(
        self,
        order_price: float,
        order_quantity: float,
        order_side: int,
        volume_ahead: float,
        volume_traded: float,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray
    ) -> Optional[FillResult]:
        """
        Check if order should be filled.
        
        Args:
            order_price: Order limit price
            order_quantity: Remaining quantity
            order_side: 1 for bid, -1 for ask
            volume_ahead: Volume ahead in queue
            volume_traded: Volume traded at this price
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot
            
        Returns:
            FillResult if filled, None otherwise
        """
        ...
    
    def calculate_volume_traded_at_price(
        self,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray,
        price: float,
        side: int
    ) -> float:
        """
        Calculate volume traded at a specific price level.
        
        Args:
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot
            price: Price level
            side: 1 for bid, -1 for ask
            
        Returns:
            Volume traded (consumed) at that price
        """
        ...


class MicrostructureModelProtocol(Protocol):
    """
    Protocol for market microstructure modeling.
    
    Decomposes volume changes into trades, adds, and cancellations.
    """
    
    def decompose_volume_change(
        self,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray,
        price: float,
        side: int,
        our_position: float,
        random_state: np.random.RandomState
    ) -> 'VolumeChange':
        """
        Decompose volume change at a price level.
        
        Args:
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot
            price: Price level to analyze
            side: 1 for bid, -1 for ask
            our_position: Our position in queue
            random_state: Random state for reproducibility
            
        Returns:
            VolumeChange with decomposed components
        """
        ...
    
    def estimate_queue_advancement(
        self,
        change: 'VolumeChange',
        our_position: float
    ) -> float:
        """
        Estimate how much queue position advances.
        
        Args:
            change: Decomposed volume change
            our_position: Current position in queue
            
        Returns:
            New position (volume ahead)
        """
        ...


class StrategyProtocol(Protocol):
    """
    Protocol for trading strategies.
    
    Strategies must implement on_snapshot to generate trading actions.
    """
    
    def on_snapshot(self, context: 'StrategyContext') -> List['OrderAction']:
        """
        Called for each market snapshot.
        
        Args:
            context: Immutable context with market state
            
        Returns:
            List of order actions to execute
        """
        ...
    
    def on_fill(self, order_id: int, price: float, quantity: float) -> None:
        """
        Optional callback when order is filled.
        
        Args:
            order_id: Filled order ID
            price: Fill price
            quantity: Fill quantity
        """
        ...
    
    def on_order_rejected(self, order_id: int, reason: str) -> None:
        """
        Optional callback when order is rejected.
        
        Args:
            order_id: Rejected order ID
            reason: Rejection reason
        """
        ...


class DataLoaderProtocol(Protocol):
    """
    Protocol for snapshot data loading.
    
    Supports multiple file formats (CSV, Parquet, HDF5).
    """
    
    def load_snapshots(
        self,
        path: str,
        symbol: Optional[str] = None,
        validate: bool = True
    ) -> np.ndarray:
        """
        Load snapshot data from file.
        
        Args:
            path: Path to data file
            symbol: Optional symbol filter
            validate: Whether to validate data
            
        Returns:
            NumPy structured array of snapshots
        """
        ...


class EventLoggerProtocol(Protocol):
    """
    Protocol for event logging during backtest.
    
    Tracks all important events for debugging and analysis.
    """
    
    def log_order_submitted(
        self,
        timestamp: int,
        order_id: int,
        side: int,
        price: float,
        quantity: float
    ) -> None:
        """Log order submission event."""
        ...
    
    def log_order_filled(
        self,
        timestamp: int,
        order_id: int,
        price: float,
        quantity: float
    ) -> None:
        """Log order fill event."""
        ...
    
    def log_order_cancelled(self, timestamp: int, order_id: int) -> None:
        """Log order cancellation event."""
        ...
    
    def log_order_rejected(
        self,
        timestamp: int,
        order_id: int,
        reason: str
    ) -> None:
        """Log order rejection event."""
        ...
    
    def log_position_updated(
        self,
        timestamp: int,
        quantity: float,
        avg_price: float,
        realized_pnl: float
    ) -> None:
        """Log position update event."""
        ...
    
    def log_queue_updated(
        self,
        timestamp: int,
        order_id: int,
        position: int,
        volume_ahead: float
    ) -> None:
        """Log queue position update event."""
        ...
    
    def log_snapshot_processed(self, timestamp: int, count: int) -> None:
        """Log snapshot processing event."""
        ...
