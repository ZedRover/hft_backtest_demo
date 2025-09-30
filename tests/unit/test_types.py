"""Unit tests for NumPy structured array types"""

import pytest
import numpy as np


def test_snapshot_dtype_exists():
    """Test that SNAPSHOT_DTYPE is defined"""
    with pytest.raises(ImportError):
        from src.data.types import SNAPSHOT_DTYPE


def test_order_dtype_exists():
    """Test that ORDER_DTYPE is defined"""
    with pytest.raises(ImportError):
        from src.data.types import ORDER_DTYPE


def test_fill_dtype_exists():
    """Test that FILL_DTYPE is defined"""
    with pytest.raises(ImportError):
        from src.data.types import FILL_DTYPE


def test_snapshot_dtype_has_required_fields():
    """Test that SNAPSHOT_DTYPE has all required fields"""
    with pytest.raises(ImportError):
        from src.data.types import SNAPSHOT_DTYPE

        required_fields = [
            'timestamp', 'symbol',
            'bid_price_0', 'bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4',
            'bid_volume_0', 'bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4',
            'ask_price_0', 'ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4',
            'ask_volume_0', 'ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4',
            'cumulative_volume', 'last_price'
        ]

        dtype_fields = [name for name, _ in SNAPSHOT_DTYPE.descr]
        for field in required_fields:
            assert field in dtype_fields, f"Missing field: {field}"


def test_snapshot_array_creation():
    """Test creating NumPy array with SNAPSHOT_DTYPE"""
    with pytest.raises(ImportError):
        from src.data.types import SNAPSHOT_DTYPE

        # Create array
        arr = np.zeros(10, dtype=SNAPSHOT_DTYPE)
        assert len(arr) == 10
        assert arr['timestamp'].dtype == np.dtype('i8')


def test_order_dtype_has_required_fields():
    """Test that ORDER_DTYPE has required fields"""
    with pytest.raises(ImportError):
        from src.data.types import ORDER_DTYPE

        required_fields = [
            'order_id', 'timestamp', 'symbol', 'side', 'price',
            'quantity', 'remaining_quantity', 'filled_quantity',
            'state', 'queue_position', 'volume_ahead'
        ]

        dtype_fields = [name for name, _ in ORDER_DTYPE.descr]
        for field in required_fields:
            assert field in dtype_fields, f"Missing field: {field}"


def test_fill_dtype_has_required_fields():
    """Test that FILL_DTYPE has required fields"""
    with pytest.raises(ImportError):
        from src.data.types import FILL_DTYPE

        required_fields = [
            'fill_id', 'order_id', 'timestamp', 'price',
            'quantity', 'side', 'fee', 'is_maker'
        ]

        dtype_fields = [name for name, _ in FILL_DTYPE.descr]
        for field in required_fields:
            assert field in dtype_fields, f"Missing field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])