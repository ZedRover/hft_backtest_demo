"""Unit tests for P&L calculation"""

import pytest
import numpy as np


def test_pnl_long_position():
    """Test P&L calculation for long position"""
    with pytest.raises(ImportError):
        from src.core.pnl import calculate_position_pnl

        # Long 10 @ 100, current price 105
        realized, unrealized = calculate_position_pnl(
            position_quantity=10.0,
            avg_entry_price=100.0,
            current_price=105.0,
            contract_multiplier=1.0
        )

        assert unrealized == 50.0  # (105 - 100) * 10


def test_pnl_short_position():
    """Test P&L calculation for short position"""
    with pytest.raises(ImportError):
        from src.core.pnl import calculate_position_pnl

        # Short 10 @ 100, current price 95
        realized, unrealized = calculate_position_pnl(
            position_quantity=-10.0,
            avg_entry_price=100.0,
            current_price=95.0,
            contract_multiplier=1.0
        )

        assert unrealized == 50.0  # (100 - 95) * 10 for short


def test_pnl_position_update_with_fill():
    """Test position update when order fills"""
    with pytest.raises(ImportError):
        from src.core.pnl import update_position

        # Initial position: long 5 @ 100
        # New fill: buy 5 @ 102
        new_qty, new_avg, realized = update_position(
            current_quantity=5.0,
            current_avg_entry=100.0,
            fill_quantity=5.0,
            fill_price=102.0,
            fill_side=1,  # Buy
            contract_multiplier=1.0
        )

        assert new_qty == 10.0  # 5 + 5
        assert new_avg == 101.0  # (5*100 + 5*102) / 10
        assert realized == 0.0  # No closing, so no realized P&L


def test_pnl_position_reduction():
    """Test position reduction (closing partial)"""
    with pytest.raises(ImportError):
        from src.core.pnl import update_position

        # Initial position: long 10 @ 100
        # Sell 5 @ 105
        new_qty, new_avg, realized = update_position(
            current_quantity=10.0,
            current_avg_entry=100.0,
            fill_quantity=5.0,
            fill_price=105.0,
            fill_side=-1,  # Sell
            contract_multiplier=1.0
        )

        assert new_qty == 5.0  # 10 - 5
        assert new_avg == 100.0  # Entry price unchanged
        assert realized == 25.0  # (105 - 100) * 5


def test_pnl_flat_position():
    """Test flat position has zero unrealized P&L"""
    with pytest.raises(ImportError):
        from src.core.pnl import calculate_position_pnl

        realized, unrealized = calculate_position_pnl(
            position_quantity=0.0,
            avg_entry_price=0.0,
            current_price=100.0,
            contract_multiplier=1.0
        )

        assert unrealized == 0.0


def test_pnl_with_fees():
    """Test P&L includes transaction fees"""
    with pytest.raises(ImportError):
        from src.core.pnl import calculate_fill_cost

        # Buy 10 @ 100 with 0.01% fee
        notional = 10.0 * 100.0 * 1.0  # quantity * price * multiplier
        fee = calculate_fill_cost(
            notional_value=notional,
            fee_rate=0.0001,  # 1 basis point
            is_maker=True
        )

        assert fee == 0.1  # 1000 * 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])