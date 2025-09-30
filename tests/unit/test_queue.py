"""
Unit tests for queue position calculation.

Tests deterministic advancement, probabilistic cancellation, and edge cases.
These tests MUST fail before implementation.
"""

import pytest
import numpy as np


def test_queue_deterministic_advancement():
    """Test that volume traded advances queue position deterministically"""
    from src.core.queue import update_queue_position

    # Order at position with 100 volume ahead
    # If 50 volume trades, should have 50 ahead remaining
    position, volume_ahead = update_queue_position(
        initial_position=10,
        initial_volume_ahead=100.0,
        volume_traded=50.0,
        cancellation_rate=0.0,  # No probabilistic cancellation
        random_value=0.5
    )

    assert volume_ahead == 50.0
    assert position >= 0


def test_queue_probabilistic_cancellation():
    """Test that probabilistic cancellations improve queue position"""
    from src.core.queue import update_queue_position

    # With cancellation rate > 0, volume ahead should decrease
    position, volume_ahead = update_queue_position(
        initial_position=10,
        initial_volume_ahead=100.0,
        volume_traded=0.0,  # No volume traded
        cancellation_rate=0.05,  # 5% cancellation rate
        random_value=0.5
    )

    # Volume ahead should be less than initial (probabilistically)
    # With fixed seed, should be deterministic
    assert volume_ahead <= 100.0
    assert volume_ahead >= 0.0


def test_queue_position_never_negative():
    """Test that queue position never goes negative"""
    from src.core.queue import update_queue_position

    position, volume_ahead = update_queue_position(
        initial_position=5,
        initial_volume_ahead=10.0,
        volume_traded=50.0,  # Trade more than volume ahead
        cancellation_rate=0.0,
        random_value=0.5
    )

    assert position >= 0
    assert volume_ahead >= 0.0


def test_queue_zero_volume_ahead_means_front():
    """Test that zero volume ahead means position is at front"""
    from src.core.queue import update_queue_position

    position, volume_ahead = update_queue_position(
        initial_position=0,
        initial_volume_ahead=0.0,
        volume_traded=0.0,
        cancellation_rate=0.0,
        random_value=0.5
    )

    assert position == 0
    assert volume_ahead == 0.0


def test_queue_reproducibility_with_seed():
    """Test that same seed gives same cancellation results"""
    from src.core.queue import update_queue_position

    # Run twice with same random value
    result1 = update_queue_position(
        initial_position=10,
        initial_volume_ahead=100.0,
        volume_traded=0.0,
        cancellation_rate=0.05,
        random_value=0.42
    )

    result2 = update_queue_position(
        initial_position=10,
        initial_volume_ahead=100.0,
        volume_traded=0.0,
        cancellation_rate=0.05,
        random_value=0.42
    )

    assert result1 == result2, "Same random_value should produce same result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])