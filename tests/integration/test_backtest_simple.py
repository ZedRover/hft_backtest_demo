"""Integration test for simple maker strategy with synthetic data"""

import pytest
import numpy as np


def test_simple_backtest_runs():
    """Test that basic backtest can run"""
    with pytest.raises(ImportError):
        from src.engine.backtest import BacktestEngine
        from src.engine.config import BacktestConfig

        # Will fail until implementation exists
        config = BacktestConfig(
            data_path="test_data.parquet",
            symbol="IF2401",
            strategy_class=None,
            strategy_params={},
            max_position=10.0,
            random_seed=42
        )

        engine = BacktestEngine(config)
        result = engine.run()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])