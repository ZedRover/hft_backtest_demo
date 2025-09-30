"""
Visualization utilities for backtest results.

Provides functions to plot P&L curves, positions, and fill distributions.
"""

import numpy as np
from typing import Optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_pnl_curve(
    timestamps: np.ndarray,
    pnl_series: np.ndarray,
    title: str = "Cumulative P&L",
    figsize: tuple = (12, 6)
) -> Optional[object]:
    """
    Plot cumulative P&L over time.

    Args:
        timestamps: Timestamp array (milliseconds)
        pnl_series: P&L series
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Convert timestamps to seconds
    time_seconds = (timestamps - timestamps[0]) / 1000.0

    ax.plot(time_seconds, pnl_series, linewidth=1.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('P&L ($)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)

    return fig


def plot_position(
    timestamps: np.ndarray,
    position_series: np.ndarray,
    title: str = "Position Over Time",
    figsize: tuple = (12, 6)
) -> Optional[object]:
    """
    Plot position over time.

    Args:
        timestamps: Timestamp array (milliseconds)
        position_series: Position series
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Convert timestamps to seconds
    time_seconds = (timestamps - timestamps[0]) / 1000.0

    ax.plot(time_seconds, position_series, linewidth=1.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Position')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)

    return fig


def plot_fill_distribution(
    fills: np.ndarray,
    title: str = "Fill Price Distribution",
    figsize: tuple = (12, 6)
) -> Optional[object]:
    """
    Plot distribution of fill prices.

    Args:
        fills: Array of fills (FILL_DTYPE)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return None

    if len(fills) == 0:
        print("No fills to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    fill_prices = fills['price']
    ax.hist(fill_prices, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Fill Price')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def plot_backtest_summary(
    timestamps: np.ndarray,
    pnl_series: np.ndarray,
    position_series: np.ndarray,
    fills: np.ndarray,
    figsize: tuple = (14, 10)
) -> Optional[object]:
    """
    Create comprehensive backtest summary plot.

    Args:
        timestamps: Timestamp array
        pnl_series: P&L series
        position_series: Position series
        fills: Array of fills
        figsize: Figure size

    Returns:
        Matplotlib figure or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return None

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Convert timestamps to seconds
    time_seconds = (timestamps - timestamps[0]) / 1000.0

    # P&L curve
    axes[0].plot(time_seconds, pnl_series, linewidth=1.5)
    axes[0].set_ylabel('P&L ($)')
    axes[0].set_title('Cumulative P&L')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3)

    # Position over time
    axes[1].plot(time_seconds, position_series, linewidth=1.5, color='orange')
    axes[1].set_ylabel('Position')
    axes[1].set_title('Position Over Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)

    # Fill distribution
    if len(fills) > 0:
        fill_prices = fills['price']
        axes[2].hist(fill_prices, bins=30, alpha=0.7, edgecolor='black', color='green')
        axes[2].set_xlabel('Fill Price')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Fill Price Distribution')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No fills', ha='center', va='center')
        axes[2].set_title('Fill Price Distribution')

    plt.tight_layout()
    return fig


def save_plot(fig: object, filepath: str, dpi: int = 300) -> None:
    """
    Save matplotlib figure to file.

    Args:
        fig: Matplotlib figure
        filepath: Output file path
        dpi: DPI for saved image
    """
    if not MATPLOTLIB_AVAILABLE or fig is None:
        print("Cannot save plot: matplotlib not available or figure is None")
        return

    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {filepath}")