"""
Order action types.

Defines actions that strategies can return to the backtesting engine.
"""

from dataclasses import dataclass
from typing import Union


@dataclass
class SubmitOrder:
    """
    Action to submit a new limit order.

    Attributes:
        side: 1 for buy, -1 for sell
        price: Limit price
        quantity: Order quantity
    """
    side: int  # 1=buy, -1=sell
    price: float
    quantity: float

    def __post_init__(self) -> None:
        """Validate fields"""
        if self.side not in (1, -1):
            raise ValueError(f"Invalid side: {self.side}. Must be 1 (buy) or -1 (sell)")
        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}. Must be positive")
        if self.quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.quantity}. Must be positive")


@dataclass
class CancelOrder:
    """
    Action to cancel an existing order.

    Attributes:
        order_id: ID of order to cancel
    """
    order_id: int

    def __post_init__(self) -> None:
        """Validate fields"""
        if self.order_id < 0:
            raise ValueError(f"Invalid order_id: {self.order_id}")


@dataclass
class ModifyOrder:
    """
    Action to modify an existing order (cancel-replace).

    Attributes:
        order_id: ID of order to modify
        new_price: New limit price
        new_quantity: New quantity
    """
    order_id: int
    new_price: float
    new_quantity: float

    def __post_init__(self) -> None:
        """Validate fields"""
        if self.order_id < 0:
            raise ValueError(f"Invalid order_id: {self.order_id}")
        if self.new_price <= 0:
            raise ValueError(f"Invalid new_price: {self.new_price}")
        if self.new_quantity <= 0:
            raise ValueError(f"Invalid new_quantity: {self.new_quantity}")


# Type alias for any order action
OrderAction = Union[SubmitOrder, CancelOrder, ModifyOrder]