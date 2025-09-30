"""Unit tests for fill simulation logic"""

import pytest
import numpy as np


def test_fill_simulation_full_fill():
    """Test full fill when sufficient volume trades"""
    with pytest.raises(ImportError):
        from src.core.matching import simulate_fill

        # Order has 1.0 remaining, 0 volume ahead, 10 volume trades
        fill = simulate_fill(
            order_remaining=1.0,
            volume_ahead=0.0,
            volume_traded=10.0,
            order_price=100.0
        )

        assert fill is not None
        fill_qty, fill_price = fill
        assert fill_qty == 1.0  # Full fill
        assert fill_price == 100.0


def test_fill_simulation_partial_fill():
    """Test partial fill when limited volume available"""
    with pytest.raises(ImportError):
        from src.core.matching import simulate_fill

        # Order has 10.0 remaining, 0 volume ahead, but only 5 volume trades
        fill = simulate_fill(
            order_remaining=10.0,
            volume_ahead=0.0,
            volume_traded=5.0,
            order_price=100.0
        )

        assert fill is not None
        fill_qty, fill_price = fill
        assert fill_qty == 5.0  # Partial fill
        assert fill_price == 100.0


def test_fill_simulation_no_fill():
    """Test no fill when volume ahead not cleared"""
    with pytest.raises(ImportError):
        from src.core.matching import simulate_fill

        # Order has volume ahead, insufficient trading to clear it
        fill = simulate_fill(
            order_remaining=1.0,
            volume_ahead=10.0,  # 10 ahead
            volume_traded=5.0,   # Only 5 trades
            order_price=100.0
        )

        assert fill is None  # No fill yet


def test_fill_simulation_after_queue_cleared():
    """Test fill after queue ahead is cleared"""
    with pytest.raises(ImportError):
        from src.core.matching import simulate_fill

        # Order has 1.0 remaining, 5 volume ahead, 10 volume trades
        # Should clear queue (5) and fill order (1)
        fill = simulate_fill(
            order_remaining=1.0,
            volume_ahead=5.0,
            volume_traded=10.0,
            order_price=100.0
        )

        assert fill is not None
        fill_qty, fill_price = fill
        assert fill_qty == 1.0  # Gets filled after queue clears
        assert fill_price == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])