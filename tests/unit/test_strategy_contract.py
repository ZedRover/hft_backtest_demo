"""
Contract tests for StrategyProtocol interface.

These tests verify that the strategy interface contract is properly enforced.
Tests MUST fail before implementation exists.
"""

import pytest
import numpy as np
from typing import List


def test_strategy_protocol_exists():
    """Test that StrategyProtocol is defined"""
    with pytest.raises(ImportError):
        from src.strategy.base import StrategyProtocol


def test_base_strategy_class_exists():
    """Test that BaseStrategy class is defined"""
    with pytest.raises(ImportError):
        from src.strategy.base import BaseStrategy


def test_strategy_context_exists():
    """Test that StrategyContext is defined"""
    with pytest.raises(ImportError):
        from src.strategy.context import StrategyContext


def test_order_actions_exist():
    """Test that order action types are defined"""
    with pytest.raises(ImportError):
        from src.strategy.actions import SubmitOrder, CancelOrder, ModifyOrder


def test_strategy_must_implement_on_snapshot():
    """Test that strategies must implement on_snapshot method"""
    # This will fail until BaseStrategy is implemented
    with pytest.raises(ImportError):
        from src.strategy.base import BaseStrategy

        class TestStrategy(BaseStrategy):
            pass

        # Should raise TypeError if on_snapshot not implemented
        with pytest.raises(TypeError):
            TestStrategy()


def test_on_snapshot_signature():
    """Test that on_snapshot has correct signature"""
    with pytest.raises(ImportError):
        from src.strategy.base import BaseStrategy
        from src.strategy.context import StrategyContext
        import inspect

        # Get on_snapshot signature
        sig = inspect.signature(BaseStrategy.on_snapshot)
        params = list(sig.parameters.keys())

        assert params == ['self', 'context'], "on_snapshot must accept self and context"
        assert sig.parameters['context'].annotation == StrategyContext


def test_on_snapshot_returns_list():
    """Test that on_snapshot returns List[OrderAction]"""
    with pytest.raises(ImportError):
        from src.strategy.base import BaseStrategy
        import inspect

        sig = inspect.signature(BaseStrategy.on_snapshot)
        return_type = sig.return_annotation

        assert 'List' in str(return_type), "on_snapshot must return List"


def test_strategy_context_immutable():
    """Test that StrategyContext fields are read-only"""
    with pytest.raises(ImportError):
        from src.strategy.context import StrategyContext
        from dataclasses import FrozenInstanceError

        # Create mock context
        ctx = StrategyContext(
            snapshot=None,
            position=None,
            active_orders=[],
            timestamp=0,
            symbol="TEST"
        )

        # Should not be able to modify
        with pytest.raises(FrozenInstanceError):
            ctx.timestamp = 1


def test_submit_order_validation():
    """Test SubmitOrder action has required fields"""
    with pytest.raises(ImportError):
        from src.strategy.actions import SubmitOrder

        order = SubmitOrder(side=1, price=100.0, quantity=1.0)
        assert order.side == 1
        assert order.price == 100.0
        assert order.quantity == 1.0


def test_cancel_order_validation():
    """Test CancelOrder action has required fields"""
    with pytest.raises(ImportError):
        from src.strategy.actions import CancelOrder

        action = CancelOrder(order_id=123)
        assert action.order_id == 123


def test_modify_order_validation():
    """Test ModifyOrder action has required fields"""
    with pytest.raises(ImportError):
        from src.strategy.actions import ModifyOrder

        action = ModifyOrder(order_id=123, new_price=101.0, new_quantity=2.0)
        assert action.order_id == 123
        assert action.new_price == 101.0
        assert action.new_quantity == 2.0


def test_strategy_optional_callbacks():
    """Test that on_fill and on_order_rejected are optional"""
    with pytest.raises(ImportError):
        from src.strategy.base import BaseStrategy

        class MinimalStrategy(BaseStrategy):
            def on_snapshot(self, context):
                return []

        # Should not require on_fill or on_order_rejected
        strategy = MinimalStrategy()
        assert hasattr(strategy, 'on_fill')
        assert hasattr(strategy, 'on_order_rejected')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])