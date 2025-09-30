"""
Market microstructure model for volume change decomposition.

Decomposes order book volume changes between snapshots into:
- Trades (volume consumed by market orders)
- New orders (volume added to the book)
- Cancellations (volume removed from the book)

Uses probabilistic models and heuristics to estimate unobservable order flow.
"""

import numpy as np
import numba
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from .constants import (
    DEFAULT_TRADE_PROBABILITY,
    DEFAULT_ADD_PROBABILITY, 
    DEFAULT_CANCEL_ALPHA,
    DEFAULT_CANCEL_BETA,
    DEFAULT_TRADE_AGGRESSION,
    PRICE_EPSILON
)


@dataclass
class VolumeChange:
    """
    Decomposition of volume change at a price level.

    Attributes:
        price: Price level
        volume_before: Volume before
        volume_after: Volume after
        traded: Estimated volume traded (consumed by market orders)
        added: Estimated new orders added
        cancelled_before_us: Cancellations before us in queue
        cancelled_after_us: Cancellations after us in queue
    """
    price: float
    volume_before: float
    volume_after: float
    traded: float = 0.0
    added: float = 0.0
    cancelled_before_us: float = 0.0
    cancelled_after_us: float = 0.0


@dataclass
class MicrostructureParams:
    """
    Parameters for microstructure model.

    Attributes:
        trade_probability: P(volume decrease is trade vs cancel) [0, 1]
        add_probability: P(volume increase is new order vs replaced cancel) [0, 1]
        cancel_loc_alpha: Shape parameter for cancel location (beta distribution)
                         Higher alpha = cancels concentrate near front
        cancel_loc_beta: Shape parameter for cancel location
                        Higher beta = cancels concentrate near back
        trade_aggression: How aggressive trades are (0=passive, 1=aggressive)
                         Affects how deep into book trades go
    """
    trade_probability: float = DEFAULT_TRADE_PROBABILITY
    add_probability: float = DEFAULT_ADD_PROBABILITY
    cancel_loc_alpha: float = DEFAULT_CANCEL_ALPHA
    cancel_loc_beta: float = DEFAULT_CANCEL_BETA
    trade_aggression: float = DEFAULT_TRADE_AGGRESSION


class MicrostructureModel:
    """
    Market microstructure model for estimating order flow.

    Uses cumulative volume, order book changes, and probabilistic models
    to decompose volume changes into trades, adds, and cancels.
    """

    def __init__(self, params: Optional[MicrostructureParams] = None):
        """
        Initialize microstructure model.

        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or MicrostructureParams()

    def decompose_volume_change(
        self,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray,
        price: float,
        side: int,
        our_position: float,
        random_state: np.random.RandomState
    ) -> VolumeChange:
        """
        Decompose volume change at a specific price level.

        Args:
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot
            price: Price level to analyze
            side: 1 for bid, -1 for ask
            our_position: Our position in queue (volume ahead of us)
            random_state: Random state for reproducibility

        Returns:
            VolumeChange with decomposed components
        """
        # Get volumes at this price
        vol_before = self._get_volume_at_price(snapshot_prev, price, side)
        vol_after = self._get_volume_at_price(snapshot_curr, price, side)

        change = VolumeChange(
            price=price,
            volume_before=vol_before,
            volume_after=vol_after
        )

        if vol_before == 0:
            # Price didn't exist before
            return change

        # Get cumulative volume change as proxy for total market activity
        cum_vol_delta = snapshot_curr['cumulative_volume'] - snapshot_prev['cumulative_volume']

        if vol_after > vol_before:
            # Volume increased - new orders added
            self._decompose_increase(change, cum_vol_delta, random_state)
        elif vol_after < vol_before:
            # Volume decreased - trades and/or cancels
            self._decompose_decrease(change, cum_vol_delta, our_position, random_state)

        return change

    def _decompose_increase(
        self,
        change: VolumeChange,
        cum_vol_delta: float,
        random_state: np.random.RandomState
    ) -> None:
        """
        Decompose volume increase (new orders added).

        Volume can increase from:
        1. New limit orders joining the book
        2. Orders that were cancelled getting replaced

        Args:
            change: VolumeChange object to update
            cum_vol_delta: Total market volume traded
            random_state: Random state
        """
        increase = change.volume_after - change.volume_before

        # Use add_probability to determine if this is truly new orders
        # or partially replacing cancelled orders
        if random_state.random() < self.params.add_probability:
            # Mostly new orders
            change.added = increase
        else:
            # Mix of new and replaced
            change.added = increase * 0.5
            # Rest is implicit cancellation that got replaced

    def _decompose_decrease(
        self,
        change: VolumeChange,
        cum_vol_delta: float,
        our_position: float,
        random_state: np.random.RandomState
    ) -> None:
        """
        Decompose volume decrease into trades and cancellations.

        Volume can decrease from:
        1. Market orders consuming liquidity (trades)
        2. Limit orders being cancelled

        We use:
        - Cumulative volume as a signal for trade activity
        - trade_probability to split between trades and cancels
        - Beta distribution to locate cancellations in the queue

        Args:
            change: VolumeChange object to update
            cum_vol_delta: Total market volume traded
            our_position: Our position in queue
            random_state: Random state
        """
        decrease = change.volume_before - change.volume_after

        # Estimate trade component based on cumulative volume
        # Higher cumulative volume = more likely to be trades
        if cum_vol_delta > 0:
            # Scale trade probability by market activity
            activity_factor = min(1.0, cum_vol_delta / change.volume_before)
            trade_prob = self.params.trade_probability * (0.5 + 0.5 * activity_factor)
        else:
            # No cumulative volume = likely cancellations
            trade_prob = self.params.trade_probability * 0.3

        # Sample trade vs cancel
        if random_state.random() < trade_prob:
            # Volume decrease is primarily from trades
            trade_fraction = 0.7 + 0.3 * random_state.random()  # 70-100% trades
            change.traded = decrease * trade_fraction
            total_cancelled = decrease * (1 - trade_fraction)
        else:
            # Volume decrease is primarily from cancellations
            cancel_fraction = 0.7 + 0.3 * random_state.random()  # 70-100% cancels
            change.traded = decrease * (1 - cancel_fraction)
            total_cancelled = decrease * cancel_fraction

        # Distribute cancellations before/after our position
        if total_cancelled > 0 and change.volume_before > 0:
            # Use beta distribution to model cancel locations
            # Alpha > Beta means more cancels at front
            # Beta > Alpha means more cancels at back
            cancel_positions = random_state.beta(
                self.params.cancel_loc_alpha,
                self.params.cancel_loc_beta,
                size=int(total_cancelled) + 1
            )

            # Cancels before us = those with position < our_position
            our_ratio = our_position / change.volume_before if change.volume_before > 0 else 0.5
            cancels_before_mask = cancel_positions < our_ratio

            change.cancelled_before_us = np.sum(cancels_before_mask) * (total_cancelled / len(cancel_positions))
            change.cancelled_after_us = total_cancelled - change.cancelled_before_us

    def estimate_queue_advancement(
        self,
        change: VolumeChange,
        our_position: float
    ) -> float:
        """
        Estimate how much our queue position advances.

        Our position advances from:
        1. Trades consuming volume ahead of us
        2. Cancellations ahead of us

        Args:
            change: VolumeChange with decomposed components
            our_position: Current position (volume ahead)

        Returns:
            New position (volume ahead after advancement)
        """
        # Start with current position
        new_position = our_position

        # Trades consume from the front of the queue
        new_position = max(0.0, new_position - change.traded)

        # Cancellations before us also advance our position
        new_position = max(0.0, new_position - change.cancelled_before_us)

        return new_position

    def _get_volume_at_price(
        self,
        snapshot: np.ndarray,
        price: float,
        side: int
    ) -> float:
        """Get volume at specific price from snapshot."""
        if side == 1:  # Bid
            prices = [
                snapshot['bid_price_0'], snapshot['bid_price_1'],
                snapshot['bid_price_2'], snapshot['bid_price_3'],
                snapshot['bid_price_4']
            ]
            volumes = [
                snapshot['bid_volume_0'], snapshot['bid_volume_1'],
                snapshot['bid_volume_2'], snapshot['bid_volume_3'],
                snapshot['bid_volume_4']
            ]
        else:  # Ask
            prices = [
                snapshot['ask_price_0'], snapshot['ask_price_1'],
                snapshot['ask_price_2'], snapshot['ask_price_3'],
                snapshot['ask_price_4']
            ]
            volumes = [
                snapshot['ask_volume_0'], snapshot['ask_volume_1'],
                snapshot['ask_volume_2'], snapshot['ask_volume_3'],
                snapshot['ask_volume_4']
            ]

        for p, v in zip(prices, volumes):
            if abs(p - price) < PRICE_EPSILON:
                return float(v)

        return 0.0


@numba.jit(nopython=True, cache=True)
def calculate_trade_volume_numba(
    volume_decrease: float,
    cumulative_volume: float,
    trade_probability: float,
    random_value: float
) -> float:
    """
    Fast calculation of trade volume from decrease (Numba-optimized).

    Args:
        volume_decrease: Total volume decrease at price
        cumulative_volume: Market cumulative volume
        trade_probability: Base probability that decrease is trade
        random_value: Random value [0, 1)

    Returns:
        Estimated trade volume
    """
    if volume_decrease <= 0:
        return 0.0

    # Adjust trade probability based on market activity
    if cumulative_volume > 0:
        activity_factor = min(1.0, cumulative_volume / max(volume_decrease, 1.0))
        trade_prob = trade_probability * (0.5 + 0.5 * activity_factor)
    else:
        trade_prob = trade_probability * 0.3

    # Determine if this is primarily trade or cancel
    if random_value < trade_prob:
        # 70-100% is trade
        trade_fraction = 0.7 + 0.3 * random_value
    else:
        # 0-30% is trade
        trade_fraction = 0.3 * random_value

    return volume_decrease * trade_fraction


@numba.jit(nopython=True, cache=True)
def estimate_cancellation_before_position(
    total_cancelled: float,
    our_position: float,
    total_volume: float,
    alpha: float,
    beta: float,
    random_value: float
) -> float:
    """
    Estimate cancellations before our position (Numba-optimized).

    Uses simplified beta distribution approximation.

    Args:
        total_cancelled: Total volume cancelled
        our_position: Our position in queue
        total_volume: Total volume at this price
        alpha: Beta distribution alpha
        beta: Beta distribution beta
        random_value: Random value [0, 1)

    Returns:
        Estimated cancellations before us
    """
    if total_cancelled <= 0 or total_volume <= 0:
        return 0.0

    # Our relative position in queue
    our_ratio = our_position / total_volume

    # Simplified beta CDF approximation
    # If alpha > beta: more weight at front (low values)
    # If beta > alpha: more weight at back (high values)
    mean = alpha / (alpha + beta)

    # Estimate what fraction of cancels are before us
    if our_ratio < mean:
        # We're ahead of mean - fewer cancels before us
        cancel_ratio = our_ratio * 0.7
    else:
        # We're behind mean - more cancels before us
        cancel_ratio = our_ratio * 1.3

    cancel_ratio = min(1.0, max(0.0, cancel_ratio))

    return total_cancelled * cancel_ratio