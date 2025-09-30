"""Unit tests for data validation"""

import pytest
import numpy as np
from src.data.validator import validate_snapshots
from src.data.types import SNAPSHOT_DTYPE


def create_minimal_snapshot(timestamp, bid_price, ask_price, bid_vol, ask_vol):
    """Create a minimal snapshot for testing."""
    snapshot = np.zeros(1, dtype=SNAPSHOT_DTYPE)[0]
    snapshot['timestamp'] = timestamp
    snapshot['symbol'] = 'TEST'

    # Fill all 5 levels
    for i in range(5):
        if i == 0:
            # Use provided values for level 0
            snapshot['bid_price_0'] = bid_price
            snapshot['ask_price_0'] = ask_price
            snapshot['bid_volume_0'] = bid_vol
            snapshot['ask_volume_0'] = ask_vol
        else:
            # Fill other levels with valid data
            snapshot[f'bid_price_{i}'] = max(0.1, bid_price - i * 0.5) if not np.isnan(bid_price) and bid_price > 0 else 100.0
            snapshot[f'ask_price_{i}'] = ask_price + i * 0.5 if not np.isnan(ask_price) and ask_price > 0 else 101.0
            snapshot[f'bid_volume_{i}'] = 10.0
            snapshot[f'ask_volume_{i}'] = 10.0

    snapshot['cumulative_volume'] = 1000.0
    snapshot['turnover_volume'] = 5000000.0

    if not np.isnan(bid_price) and not np.isnan(ask_price) and bid_price > 0 and ask_price > 0:
        snapshot['last_price'] = (bid_price + ask_price) / 2
    else:
        snapshot['last_price'] = 100.0

    return np.array([snapshot])


def test_validate_monotonic_timestamps():
    """Test that timestamps must be monotonically increasing"""
    # Create data with non-monotonic timestamps
    data = np.zeros(2, dtype=SNAPSHOT_DTYPE)
    data[0]['timestamp'] = 1000
    data[0]['symbol'] = 'TEST'
    data[0]['bid_price_0'] = 100.0
    data[0]['ask_price_0'] = 101.0
    data[0]['bid_volume_0'] = 10.0
    data[0]['ask_volume_0'] = 10.0
    data[0]['cumulative_volume'] = 1000.0
    data[0]['turnover_volume'] = 500000.0
    data[0]['last_price'] = 100.5

    data[1]['timestamp'] = 999  # Goes backward!
    data[1]['symbol'] = 'TEST'
    data[1]['bid_price_0'] = 100.0
    data[1]['ask_price_0'] = 101.0
    data[1]['bid_volume_0'] = 10.0
    data[1]['ask_volume_0'] = 10.0
    data[1]['cumulative_volume'] = 1000.0
    data[1]['turnover_volume'] = 500000.0
    data[1]['last_price'] = 100.5

    with pytest.raises(ValueError, match="Timestamps must be monotonically increasing"):
        validate_snapshots(data)


def test_validate_no_crossed_book():
    """Test that bid prices must be < ask prices"""
    data = create_minimal_snapshot(1000, 101.0, 100.0, 10.0, 10.0)  # Crossed!

    with pytest.raises(ValueError, match="Crossed book"):
        validate_snapshots(data)


def test_validate_no_nan_values():
    """Test that NaN values are rejected"""
    data = create_minimal_snapshot(1000, np.nan, 101.0, 10.0, 10.0)  # NaN bid

    with pytest.raises(ValueError, match="NaN.*bid_price_0"):
        validate_snapshots(data)


def test_validate_negative_volumes():
    """Test that negative volumes are rejected"""
    data = create_minimal_snapshot(1000, 100.0, 101.0, -10.0, 10.0)  # Negative volume

    with pytest.raises(ValueError, match="Negative volume"):
        validate_snapshots(data)


def test_validate_valid_data_passes():
    """Test that valid data passes validation"""
    data = create_minimal_snapshot(1000, 100.0, 101.0, 10.0, 10.0)

    # Should not raise
    validate_snapshots(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])