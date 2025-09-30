"""
Aggressive market maker strategy.

Places orders at the best bid/ask to maximize fill rate.
Demonstrates orderbook state tracking with orders inside the book.
"""

from typing import List
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.actions import SubmitOrder, OrderAction


class AggressiveMaker(BaseStrategy):
    """
    Aggressive market-making strategy that quotes at top of book.

    Strategy:
        - If no active orders: place buy at best bid, sell at best ask
        - If have active orders: wait for fills
        - Uses tick_offset to avoid crossing the spread

    This strategy demonstrates:
        - High fill rate by being at front of queue
        - Orderbook state tracking with orders in visible book
        - More realistic maker behavior
    """

    def __init__(self, tick_offset: float = 0.2, quote_size: float = 1.0):
        """
        Initialize strategy.

        Args:
            tick_offset: Offset from best bid/ask in price units (0.2 = stay on our side)
            quote_size: Size of each quote
        """
        self.tick_offset = tick_offset
        self.quote_size = quote_size
        self.order_count = 0

    def on_snapshot(self, context: StrategyContext) -> List[OrderAction]:
        """
        Called for each snapshot.

        Args:
            context: Strategy context with market data

        Returns:
            List of order actions
        """
        # Check position limit
        if abs(context.position_quantity) >= 5.0:
            # Position limit reached, don't add more
            return []

        # Only quote if we don't have too many active orders
        if len(context.active_orders) >= 10:
            return []

        snapshot = context.snapshot

        # Get best bid/ask
        best_bid = float(snapshot['bid_price_0'])
        best_ask = float(snapshot['ask_price_0'])

        # Place orders just inside the spread
        # Buy slightly below best bid to provide liquidity
        # Sell slightly above best ask to provide liquidity
        buy_price = best_bid - self.tick_offset
        sell_price = best_ask + self.tick_offset

        # Sanity check: don't cross the spread
        if buy_price >= sell_price - 0.5:
            # Spread too tight, skip
            return []

        # Place quotes on both sides
        actions = [
            SubmitOrder(side=1, price=buy_price, quantity=self.quote_size),
            SubmitOrder(side=-1, price=sell_price, quantity=self.quote_size),
        ]

        self.order_count += 2
        return actions

    def on_fill(self, order_id: int, price: float, quantity: float) -> None:
        """
        Called when order is filled.

        Args:
            order_id: Filled order ID
            price: Fill price
            quantity: Fill quantity
        """
        # In this strategy, we just let fills happen and requote next snapshot
        pass