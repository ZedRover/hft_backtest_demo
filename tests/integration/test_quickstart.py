"""Integration test validating quickstart.md works end-to-end"""

import pytest


def test_quickstart_example_runs():
    """Test that quickstart example can execute"""
    with pytest.raises(ImportError):
        from src.engine.backtest import BacktestEngine
        # Test will be implemented after engine exists


if __name__ == "__main__":
    pytest.main([__file__, "-v"])