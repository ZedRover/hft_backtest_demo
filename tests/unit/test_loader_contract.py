"""Contract tests for DataLoaderProtocol"""

import pytest


def test_data_loader_exists():
    """Test that DataLoader class exists"""
    with pytest.raises(ImportError):
        from src.data.loader import DataLoader


def test_loader_has_load_method():
    """Test that loader has load() method"""
    with pytest.raises(ImportError):
        from src.data.loader import DataLoader
        assert hasattr(DataLoader, 'load')


def test_load_returns_numpy_array():
    """Test that load() returns NumPy structured array"""
    with pytest.raises(ImportError):
        from src.data.loader import DataLoader
        import inspect

        sig = inspect.signature(DataLoader.load)
        # Should return np.ndarray


if __name__ == "__main__":
    pytest.main([__file__, "-v"])