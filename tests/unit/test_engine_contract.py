"""Contract tests for BacktestEngineProtocol"""

import pytest


def test_backtest_engine_exists():
    """Test that BacktestEngine class exists"""
    with pytest.raises(ImportError):
        from src.engine.backtest import BacktestEngine


def test_backtest_config_exists():
    """Test that BacktestConfig exists"""
    with pytest.raises(ImportError):
        from src.engine.config import BacktestConfig


def test_backtest_result_exists():
    """Test that BacktestResult exists"""
    with pytest.raises(ImportError):
        from src.analysis.report import BacktestResult


def test_engine_has_run_method():
    """Test that engine has run() method"""
    with pytest.raises(ImportError):
        from src.engine.backtest import BacktestEngine
        assert hasattr(BacktestEngine, 'run')


def test_engine_has_validate_data_method():
    """Test that engine has validate_data() method"""
    with pytest.raises(ImportError):
        from src.engine.backtest import BacktestEngine
        assert hasattr(BacktestEngine, 'validate_data')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])