"""
Queue position tracking and simulation.

Implements probabilistic queue model with Numba JIT optimization.
Tracks position in limit order book queue and simulates fills.
"""

import numpy as np
import numba
from typing import Tuple, Optional


@numba.jit(nopython=True, cache=True)
def update_queue_position(
    initial_position: int,
    initial_volume_ahead: float,
    volume_traded: float,
    cancellation_rate: float,
    random_value: float
) -> Tuple[int, float]:
    """
    Update queue position based on volume traded and probabilistic cancellations.

    Args:
        initial_position: Current position in queue
        initial_volume_ahead: Volume ahead in queue
        volume_traded: Volume traded at this price level
        cancellation_rate: Probability of orders ahead cancelling per snapshot
        random_value: Random value in [0, 1) for cancellation sampling

    Returns:
        (new_position, new_volume_ahead)

    Implementation:
        1. Apply deterministic advancement from volume traded
        2. Apply probabilistic cancellations to remaining volume ahead
        3. Ensure position and volume_ahead never negative
    """
    volume_ahead = initial_volume_ahead

    # Deterministic advancement: reduce volume ahead by volume traded
    volume_ahead = max(0.0, volume_ahead - volume_traded)

    # Probabilistic cancellation: approximate binomial with expected value
    # Expected cancellations = n * p
    if volume_ahead > 0 and cancellation_rate > 0:
        n_units = volume_ahead
        # Use random_value to add variance around expected value
        # cancelled ~ Binomial(n, p) ≈ n*p + noise
        expected_cancelled = n_units * cancellation_rate
        # Add variance: use random_value to perturb around mean
        # Simple approximation: cancelled = expected ± random perturbation
        std_dev = np.sqrt(n_units * cancellation_rate * (1 - cancellation_rate))
        # Map random_value [0,1) to z-score approximately
        z_score = (random_value - 0.5) * 4  # ~95% coverage
        cancelled = expected_cancelled + z_score * std_dev
        cancelled = max(0.0, min(n_units, cancelled))
        volume_ahead = max(0.0, volume_ahead - cancelled)

    # Calculate new position (simple: position proportional to volume ahead)
    if volume_ahead <= 0:
        position = 0
    else:
        position = int(volume_ahead)  # Simplified position calculation

    return position, volume_ahead


class QueueSimulator:
    """
    Queue simulator for order fill simulation.

    Implements QueueSimulatorProtocol with Numba-optimized functions.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize queue simulator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_state = np.random.RandomState(random_seed)

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
            snapshot: Current snapshot
            cancellation_rate: Probability of cancellation per snapshot
            volume_traded: Volume traded at order's price level

        Returns:
            (new_queue_position, new_volume_ahead)
        """
        initial_position = int(order['queue_position'])
        initial_volume_ahead = float(order['volume_ahead'])

        # Generate random value for cancellation simulation
        random_value = self.random_state.random()

        new_position, new_volume_ahead = update_queue_position(
            initial_position=initial_position,
            initial_volume_ahead=initial_volume_ahead,
            volume_traded=volume_traded,
            cancellation_rate=cancellation_rate,
            random_value=random_value
        )

        return new_position, new_volume_ahead

    def simulate_fills(
        self,
        order: np.ndarray,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Determine if order should be filled based on queue position.

        Args:
            order: Order record
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot

        Returns:
            (fill_price, fill_quantity) if filled, else None

        Logic:
            - Calculate volume traded at order's price level
            - If volume_ahead < volume_traded, order gets filled
            - Fill quantity = min(order.remaining_quantity, available_volume)
        """
        # Get order details
        remaining_qty = float(order['remaining_quantity'])
        volume_ahead = float(order['volume_ahead'])
        order_price = float(order['price'])
        order_side = int(order['side'])

        if remaining_qty <= 0:
            return None

        # Estimate volume traded at this price level
        # For simplicity, use cumulative volume delta as proxy
        volume_delta = snapshot_curr['cumulative_volume'] - snapshot_prev['cumulative_volume']

        # Check if queue ahead is cleared
        if volume_ahead < volume_delta:
            # Queue is cleared, order can be filled
            available_volume = volume_delta - volume_ahead

            # Fill quantity is minimum of remaining and available
            fill_qty = min(remaining_qty, available_volume)

            return order_price, fill_qty

        return None


def initialize_queue_position(
    order_price: float,
    order_side: int,
    snapshot: np.ndarray
) -> Tuple[int, float]:
    """
    Initialize queue position when order is first placed.

    Args:
        order_price: Order limit price
        order_side: 1 for buy, -1 for sell
        snapshot: Current snapshot

    Returns:
        (initial_queue_position, initial_volume_ahead)

    Assumes order joins the back of the queue at its price level.
    """
    # Get volume at this price level from snapshot
    from .orderbook import get_volume_at_price

    volume_at_level = get_volume_at_price(snapshot, order_price, order_side)

    # Order joins back of queue
    # Initial volume ahead = current volume at this level
    volume_ahead = volume_at_level
    position = int(volume_ahead) if volume_ahead > 0 else 0

    return position, volume_ahead