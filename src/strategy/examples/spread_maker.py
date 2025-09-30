"""
Adaptive spread market maker strategy.

Adjusts spread based on recent volatility.
"""

from typing import List
import numpy as np
from ..base import BaseStrategy
from ..context import StrategyContext
from ..actions import SubmitOrder, CancelOrder, OrderAction


class SpreadMaker(BaseStrategy):
    """
    Market maker with adaptive spread based on volatility.

    Strategy:
        - Calculate recent price volatility
        - Adjust spread: wider spread in high volatility
        - Cancel old orders and replace when spread changes significantly

    Args:
        base_spread_bps: Minimum spread in basis points
        max_spread_bps: Maximum spread in basis points
        quote_size: Size of each quote
        max_position: Maximum absolute position size
        lookback: Number of snapshots for volatility calculation
    """

    def __init__(
        self,
        base_spread_bps: int = 10,
        max_spread_bps: int = 50,
        quote_size: float = 1.0,
        max_position: float = 10.0,
        lookback: int = 20
    ):
        self.base_spread_bps = base_spread_bps
        self.max_spread_bps = max_spread_bps
        self.quote_size = quote_size
        self.max_position = max_position
        self.lookback = lookback

        # State tracking
        self.price_history: List[float] = []
        self.current_spread_bps = base_spread_bps

    def on_snapshot(self, context: StrategyContext) -> List[OrderAction]:
        """Generate order actions with adaptive spread"""
        # Update price history
        mid = context.get_mid_price()
        self.price_history.append(mid)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)

        # Calculate volatility (standard deviation of returns)
        if len(self.price_history) >= 2:
            prices = np.array(self.price_history)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.0

            # Adjust spread based on volatility
            # Higher volatility → wider spread
            vol_multiplier = 1.0 + min(volatility * 100, 4.0)  # Cap at 5x
            target_spread = self.base_spread_bps * vol_multiplier
            target_spread = min(target_spread, self.max_spread_bps)
        else:
            target_spread = self.base_spread_bps

        # Check if spread changed significantly (>20%)
        spread_changed = abs(target_spread - self.current_spread_bps) / self.current_spread_bps > 0.2

        actions: List[OrderAction] = []

        # Cancel existing orders if spread changed
        if spread_changed and context.has_active_orders():
            for order in context.active_orders:
                actions.append(CancelOrder(order_id=int(order['order_id'])))

        # Place new orders if no active orders (or just cancelled)
        if not context.has_active_orders() or spread_changed:
            # Don't trade if position limit reached
            if abs(context.position_quantity) < self.max_position:
                spread = mid * target_spread / 10000
                buy_price = mid - spread / 2
                sell_price = mid + spread / 2

                actions.extend([
                    SubmitOrder(side=1, price=buy_price, quantity=self.quote_size),
                    SubmitOrder(side=-1, price=sell_price, quantity=self.quote_size),
                ])

                self.current_spread_bps = target_spread

        return actions