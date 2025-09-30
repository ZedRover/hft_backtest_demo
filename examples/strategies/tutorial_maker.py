"""
Tutorial market maker strategy.

Simple example from quickstart.md demonstrating strategy interface.
"""

from typing import List
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.actions import SubmitOrder, OrderAction


class TutorialMaker(BaseStrategy):
    """
    Simple market-making strategy for tutorial purposes.

    Strategy:
        - If no active orders: place buy and sell quotes
        - If have active orders: wait
        - Targets spread_bps basis points around mid price

    This strategy demonstrates:
        - Implementing BaseStrategy
        - Using StrategyContext
        - Generating OrderActions
    """

    def __init__(self, spread_bps: int = 10, quote_size: float = 1.0):
        """
        Initialize strategy.

        Args:
            spread_bps: Target spread in basis points (10 = 0.1%)
            quote_size: Size of each quote
        """
        self.spread_bps = spread_bps
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
        # Skip if we already have active orders
        if context.has_active_orders():
            return []

        # Calculate target prices
        mid = context.get_mid_price()
        spread = mid * self.spread_bps / 10000

        buy_price = mid - spread / 2
        sell_price = mid + spread / 2

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
        # In this simple strategy, we just let the engine handle it
        # A more sophisticated strategy might adjust quotes after fills
        pass