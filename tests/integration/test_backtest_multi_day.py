"""Integration test for multi-day backtest"""

import pytest


def test_multi_day_position_carryover():
    """Test that position carries over across days"""
    with pytest.raises(ImportError):
        from src.engine.backtest import BacktestEngine
        # Test will be implemented after engine exists


if __name__ == "__main__":
    pytest.main([__file__, "-v"])