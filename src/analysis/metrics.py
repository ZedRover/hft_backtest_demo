"""
Performance metrics calculation.

Calculates Sharpe ratio, drawdown, win rate, and other performance metrics.
"""

import numpy as np
from typing import Dict


def calculate_sharpe_ratio(
    pnl_series: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 48  # 500ms snapshots, 48 per day, 252 trading days
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        pnl_series: P&L time series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(pnl_series) < 2:
        return 0.0

    # Calculate returns
    returns = np.diff(pnl_series)

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    # Annualized Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)

    return float(sharpe_annualized)


def calculate_max_drawdown(pnl_series: np.ndarray) -> tuple[float, int]:
    """
    Calculate maximum drawdown.

    Args:
        pnl_series: P&L time series

    Returns:
        (max_drawdown, max_drawdown_duration) where duration is in periods
    """
    if len(pnl_series) == 0:
        return 0.0, 0

    # Calculate running maximum
    running_max = np.maximum.accumulate(pnl_series)

    # Calculate drawdown
    drawdown = pnl_series - running_max

    # Max drawdown
    max_dd = float(np.min(drawdown))

    # Calculate duration
    # Find start and end of max drawdown period
    max_dd_idx = int(np.argmin(drawdown))
    start_idx = int(np.argmax(pnl_series[:max_dd_idx+1])) if max_dd_idx > 0 else 0
    duration = max_dd_idx - start_idx

    return max_dd, duration


def calculate_win_rate(fills: np.ndarray) -> float:
    """
    Calculate win rate from fills.

    Args:
        fills: NumPy array of fills

    Returns:
        Win rate (0-1)
    """
    if len(fills) == 0:
        return 0.0

    # Group fills by unique prices to identify profitable vs unprofitable trades
    # Simplified: assume each fill is a round-trip trade
    winning_trades = 0
    total_trades = 0

    # This is simplified - proper implementation would track position changes
    for fill in fills:
        total_trades += 1
        # Simplified logic - needs proper P&L attribution
        if fill['fee'] < 0:  # Maker rebate
            winning_trades += 1

    return winning_trades / total_trades if total_trades > 0 else 0.0


def calculate_profit_factor(fills: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        fills: NumPy array of fills

    Returns:
        Profit factor
    """
    if len(fills) == 0:
        return 0.0

    # Simplified calculation
    # Proper implementation needs to track P&L per trade
    total_fees = np.sum(fills['fee'])

    # Placeholder logic
    return 1.0 if total_fees < 0 else 0.0


def calculate_all_metrics(
    pnl_series: np.ndarray,
    fills: np.ndarray,
    orders: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all performance metrics.

    Args:
        pnl_series: P&L time series
        fills: Array of fills
        orders: Array of orders

    Returns:
        Dictionary of metrics
    """
    sharpe = calculate_sharpe_ratio(pnl_series)
    max_dd, dd_duration = calculate_max_drawdown(pnl_series)
    win_rate = calculate_win_rate(fills)
    profit_factor = calculate_profit_factor(fills)

    # Count winning/losing trades (simplified)
    total_trades = len(fills)
    winning_trades = int(win_rate * total_trades)
    losing_trades = total_trades - winning_trades

    # Calculate average win/loss (simplified)
    if len(fills) > 0:
        avg_win = 0.0  # Placeholder
        avg_loss = 0.0  # Placeholder
    else:
        avg_win = 0.0
        avg_loss = 0.0

    return {
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'max_drawdown_duration': dd_duration,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }