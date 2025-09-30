"""
Base strategy class.

Defines the StrategyProtocol interface that all strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import List
from .context import StrategyContext
from .actions import OrderAction


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies must implement on_snapshot() to generate order actions
    based on market state.
    """

    @abstractmethod
    def on_snapshot(self, context: StrategyContext) -> List[OrderAction]:
        """
        Called for each market snapshot.

        Args:
            context: Immutable context containing snapshot, position, and orders

        Returns:
            List of order actions to execute (can be empty)

        Contract:
            - MUST return a list (not None)
            - MUST NOT modify context (immutable)
            - SHOULD complete quickly (<100μs for performance)
        """
        raise NotImplementedError("Strategy must implement on_snapshot()")

    def on_fill(self, order_id: int, price: float, quantity: float) -> None:
        """
        Optional callback when an order is filled.

        Args:
            order_id: ID of the filled order
            price: Fill price
            quantity: Fill quantity

        Note:
            This is optional and can be used for strategy state updates.
        """
        pass

    def on_order_rejected(self, order_id: int, reason: str) -> None:
        """
        Optional callback when an order is rejected.

        Args:
            order_id: ID of the rejected order
            reason: Rejection reason

        Note:
            This is optional and can be used for error handling.
        """
        pass