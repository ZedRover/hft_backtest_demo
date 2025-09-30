"""
High-Frequency Trading Performance Metrics.

Implements HFT-specific metrics including:
- Intraday Sharpe ratios (15min, 30min, 1hour)
- High-frequency risk metrics
- Order execution quality
- Market making specific metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class HFTMetrics:
    """Container for HFT performance metrics."""

    # Time-based Sharpe ratios (annualized)
    sharpe_15min: float
    sharpe_30min: float
    sharpe_1hour: float
    sharpe_daily: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration_seconds: float
    calmar_ratio: float  # Return / Max Drawdown
    sortino_ratio: float  # Downside deviation only

    # High-frequency metrics
    pnl_per_trade: float
    pnl_per_contract: float
    profit_factor: float  # Gross profit / Gross loss

    # Consistency metrics
    win_rate: float
    profit_loss_ratio: float  # Avg win / Avg loss
    consecutive_wins_max: int
    consecutive_losses_max: int

    # Volatility metrics
    pnl_volatility_15min: float
    pnl_volatility_1hour: float
    return_to_volatility_ratio: float

    # Market making specific
    inventory_turnover: float  # How often position flips
    realized_spread: float  # Profit per round-trip
    adverse_selection_cost: float  # Cost of inventory risk

    # Execution quality
    fill_rate: float
    cancel_rate: float
    avg_time_to_fill_ms: float
    slippage_per_fill: float

    # Statistical significance
    t_statistic: float  # For mean PnL
    p_value: float

    # Risk-adjusted returns
    information_ratio: float  # Excess return / tracking error
    omega_ratio: float  # Prob weighted gains / losses


def calculate_rolling_sharpe(
    pnl_series: np.ndarray,
    timestamps: np.ndarray,
    window_minutes: int,
    annualization_factor: float = None
) -> float:
    """
    Calculate Sharpe ratio over rolling windows.

    Args:
        pnl_series: P&L time series
        timestamps: Timestamps in milliseconds
        window_minutes: Rolling window in minutes
        annualization_factor: Custom annualization (default: inferred)

    Returns:
        Annualized Sharpe ratio
    """
    if len(pnl_series) < 2:
        return 0.0

    # Convert to DataFrame for easier time-based operations
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='ms'),
        'pnl': pnl_series
    })
    df.set_index('timestamp', inplace=True)

    # Calculate returns
    df['returns'] = df['pnl'].diff()

    # Rolling window
    window = f'{window_minutes}min'  # min = minutes
    rolling_returns = df['returns'].rolling(window=window, min_periods=1)

    # Calculate Sharpe for each window
    mean_returns = rolling_returns.mean()
    std_returns = rolling_returns.std()

    # Average Sharpe across all windows
    sharpe_series = mean_returns / std_returns.replace(0, np.nan)
    avg_sharpe = sharpe_series.mean()

    if np.isnan(avg_sharpe):
        return 0.0

    # Annualize
    if annualization_factor is None:
        # Infer from data frequency
        time_diffs = np.diff(timestamps) / 1000.0  # seconds
        avg_interval_seconds = np.median(time_diffs)
        periods_per_year = (365.25 * 24 * 3600) / avg_interval_seconds
        annualization_factor = np.sqrt(periods_per_year)

    return float(avg_sharpe * annualization_factor)


def calculate_sortino_ratio(
    pnl_series: np.ndarray,
    timestamps: np.ndarray,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility).

    Args:
        pnl_series: P&L time series
        timestamps: Timestamps in milliseconds
        target_return: Minimum acceptable return

    Returns:
        Annualized Sortino ratio
    """
    if len(pnl_series) < 2:
        return 0.0

    returns = np.diff(pnl_series)

    # Downside returns only
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return np.inf if np.mean(returns) > 0 else 0.0

    mean_return = np.mean(returns)
    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return np.inf if mean_return > 0 else 0.0

    # Annualize
    time_diffs = np.diff(timestamps) / 1000.0
    avg_interval_seconds = np.median(time_diffs)
    periods_per_year = (365.25 * 24 * 3600) / avg_interval_seconds
    annualization = np.sqrt(periods_per_year)

    sortino = (mean_return / downside_std) * annualization
    return float(sortino)


def calculate_calmar_ratio(
    pnl_series: np.ndarray,
    max_drawdown: float,
    duration_days: float
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        pnl_series: P&L time series
        max_drawdown: Maximum drawdown
        duration_days: Duration in days

    Returns:
        Calmar ratio
    """
    if len(pnl_series) < 2 or max_drawdown == 0:
        return 0.0

    total_return = pnl_series[-1] - pnl_series[0]
    annual_return = total_return * (365.25 / duration_days)

    return float(annual_return / abs(max_drawdown))


def calculate_profit_factor(fills: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate profit factor and related metrics.

    Args:
        fills: Array of fills with 'side', 'price', 'quantity', 'fee'

    Returns:
        (profit_factor, gross_profit, gross_loss)
    """
    if len(fills) == 0:
        return 0.0, 0.0, 0.0

    # Calculate P&L for each fill
    # Simplified: assume alternating buy/sell creates round-trips
    position = 0.0
    avg_price = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    for fill in fills:
        side = fill['side']
        price = fill['price']
        qty = fill['quantity']
        fee = fill['fee']

        if position == 0:
            # Opening position
            position = qty * side
            avg_price = price
        else:
            # Check if closing
            if np.sign(position) != np.sign(side):
                # Closing trade
                pnl = (price - avg_price) * np.sign(position) * min(abs(position), qty)
                pnl -= fee

                if pnl > 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)

                # Update position
                position += qty * side

                if abs(position) < 1e-9:
                    position = 0.0
                    avg_price = 0.0
                else:
                    avg_price = price
            else:
                # Adding to position
                total_qty = abs(position) + qty
                avg_price = (avg_price * abs(position) + price * qty) / total_qty
                position += qty * side

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    return float(profit_factor), float(gross_profit), float(gross_loss)


def calculate_consecutive_streaks(fills: np.ndarray) -> Tuple[int, int]:
    """
    Calculate max consecutive wins and losses.

    Args:
        fills: Array of fills

    Returns:
        (max_consecutive_wins, max_consecutive_losses)
    """
    if len(fills) == 0:
        return 0, 0

    # Simple implementation: track sign changes in cumulative PnL
    pnl_changes = []
    position = 0.0
    avg_price = 0.0

    for fill in fills:
        side = fill['side']
        price = fill['price']
        qty = fill['quantity']

        if position != 0 and np.sign(position) != np.sign(side):
            # Closing trade
            pnl = (price - avg_price) * np.sign(position) * min(abs(position), qty)
            pnl_changes.append(1 if pnl > 0 else -1)

        position += qty * side
        if abs(position) > 1e-9:
            avg_price = price
        else:
            position = 0.0
            avg_price = 0.0

    if len(pnl_changes) == 0:
        return 0, 0

    # Count consecutive
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for change in pnl_changes:
        if change > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    return max_wins, max_losses


def calculate_inventory_turnover(
    position_series: np.ndarray,
    timestamps: np.ndarray
) -> float:
    """
    Calculate how often position flips sign (inventory turnover).

    Args:
        position_series: Position time series
        timestamps: Timestamps in milliseconds

    Returns:
        Turnovers per day
    """
    if len(position_series) < 2:
        return 0.0

    # Count sign changes
    sign_changes = np.sum(np.diff(np.sign(position_series)) != 0)

    # Duration in days
    duration_days = (timestamps[-1] - timestamps[0]) / (1000.0 * 86400)

    return float(sign_changes / duration_days) if duration_days > 0 else 0.0


def calculate_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio (probability weighted gains / losses).

    Args:
        returns: Return series
        threshold: Return threshold

    Returns:
        Omega ratio
    """
    if len(returns) == 0:
        return 0.0

    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]

    if len(losses) == 0:
        return np.inf

    return float(np.sum(gains) / np.sum(losses)) if np.sum(losses) > 0 else 0.0


def calculate_t_statistic(returns: np.ndarray) -> Tuple[float, float]:
    """
    Calculate t-statistic for mean return (statistical significance).

    Args:
        returns: Return series

    Returns:
        (t_statistic, p_value)
    """
    if len(returns) < 2:
        return 0.0, 1.0

    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    n = len(returns)

    if std == 0:
        return np.inf if mean > 0 else 0.0, 0.0

    t_stat = mean / (std / np.sqrt(n))

    # Two-tailed p-value (approximation)
    from scipy import stats
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

    return float(t_stat), float(p_value)


def calculate_execution_metrics(
    orders: np.ndarray,
    fills: np.ndarray,
    timestamps: np.ndarray
) -> Dict[str, float]:
    """
    Calculate order execution quality metrics.

    Args:
        orders: Array of orders
        fills: Array of fills
        timestamps: Timestamps

    Returns:
        Dictionary of execution metrics
    """
    if len(orders) == 0:
        return {
            'fill_rate': 0.0,
            'cancel_rate': 0.0,
            'avg_time_to_fill_ms': 0.0,
            'slippage_per_fill': 0.0,
        }

    # Fill rate (state: 0=pending, 1=partial, 2=filled, 3=cancelled, 4=rejected)
    filled = np.sum(orders['state'] == 2)  # ORDER_STATE_FILLED
    cancelled = np.sum(orders['state'] == 3)  # ORDER_STATE_CANCELLED
    total = len(orders)

    fill_rate = filled / total
    cancel_rate = cancelled / total

    # Time to fill
    filled_orders = orders[orders['state'] == 2]  # ORDER_STATE_FILLED
    if len(filled_orders) > 0 and len(fills) > 0:
        # Match fills to orders
        time_to_fills = []
        for order in filled_orders:
            order_id = order['order_id']
            order_time = order['timestamp']

            # Find corresponding fills
            order_fills = fills[fills['order_id'] == order_id]
            if len(order_fills) > 0:
                first_fill_time = order_fills['timestamp'][0]
                time_to_fills.append(first_fill_time - order_time)

        avg_time_to_fill = np.mean(time_to_fills) if len(time_to_fills) > 0 else 0.0
    else:
        avg_time_to_fill = 0.0

    # Slippage (simplified: difference between order price and fill price)
    if len(fills) > 0 and len(orders) > 0:
        slippages = []
        for fill in fills:
            order_id = fill['order_id']
            matching_orders = orders[orders['order_id'] == order_id]

            if len(matching_orders) > 0:
                order_price = matching_orders[0]['price']
                fill_price = fill['price']
                side = fill['side']

                slippage = (fill_price - order_price) * side
                slippages.append(slippage)

        avg_slippage = np.mean(slippages) if len(slippages) > 0 else 0.0
    else:
        avg_slippage = 0.0

    return {
        'fill_rate': float(fill_rate),
        'cancel_rate': float(cancel_rate),
        'avg_time_to_fill_ms': float(avg_time_to_fill),
        'slippage_per_fill': float(avg_slippage),
    }


def calculate_all_hft_metrics(
    pnl_series: np.ndarray,
    position_series: np.ndarray,
    timestamps: np.ndarray,
    fills: np.ndarray,
    orders: np.ndarray
) -> HFTMetrics:
    """
    Calculate all HFT performance metrics.

    Args:
        pnl_series: P&L time series
        position_series: Position time series
        timestamps: Timestamps in milliseconds
        fills: Array of fills
        orders: Array of orders

    Returns:
        HFTMetrics dataclass with all metrics
    """
    # Time-based Sharpe ratios
    sharpe_15min = calculate_rolling_sharpe(pnl_series, timestamps, 15)
    sharpe_30min = calculate_rolling_sharpe(pnl_series, timestamps, 30)
    sharpe_1hour = calculate_rolling_sharpe(pnl_series, timestamps, 60)
    sharpe_daily = calculate_rolling_sharpe(pnl_series, timestamps, 1440)  # 24*60

    # Drawdown
    running_max = np.maximum.accumulate(pnl_series)
    drawdown = pnl_series - running_max
    max_drawdown = float(np.min(drawdown))

    max_dd_idx = int(np.argmin(drawdown))
    start_idx = int(np.argmax(pnl_series[:max_dd_idx+1])) if max_dd_idx > 0 else 0
    dd_duration_seconds = (timestamps[max_dd_idx] - timestamps[start_idx]) / 1000.0

    # Calmar ratio
    duration_days = (timestamps[-1] - timestamps[0]) / (1000.0 * 86400)
    calmar = calculate_calmar_ratio(pnl_series, max_drawdown, duration_days)

    # Sortino ratio
    sortino = calculate_sortino_ratio(pnl_series, timestamps)

    # P&L metrics
    total_pnl = pnl_series[-1] - pnl_series[0] if len(pnl_series) > 0 else 0.0
    pnl_per_trade = total_pnl / len(fills) if len(fills) > 0 else 0.0

    total_volume = np.sum(fills['quantity']) if len(fills) > 0 else 0.0
    pnl_per_contract = total_pnl / total_volume if total_volume > 0 else 0.0

    # Profit factor
    profit_factor, gross_profit, gross_loss = calculate_profit_factor(fills)

    # Win rate
    winning_fills = gross_profit / (abs(pnl_per_trade) + 1e-9) if pnl_per_trade != 0 else 0
    win_rate = winning_fills / len(fills) if len(fills) > 0 else 0.0
    win_rate = min(1.0, max(0.0, win_rate))  # Clamp

    # Profit/Loss ratio
    avg_win = gross_profit / winning_fills if winning_fills > 0 else 0.0
    losing_fills = len(fills) - winning_fills
    avg_loss = gross_loss / losing_fills if losing_fills > 0 else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf

    # Consecutive streaks
    max_wins, max_losses = calculate_consecutive_streaks(fills)

    # Volatility
    returns = np.diff(pnl_series)
    if len(returns) > 0:
        # 15-min volatility
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps[1:], unit='ms'),
            'returns': returns
        })
        df.set_index('timestamp', inplace=True)
        vol_15min = df['returns'].rolling('15min').std().mean()
        vol_1hour = df['returns'].rolling('1h').std().mean()

        overall_vol = np.std(returns)
        mean_return = np.mean(returns)
        return_vol_ratio = mean_return / overall_vol if overall_vol > 0 else 0.0
    else:
        vol_15min = 0.0
        vol_1hour = 0.0
        return_vol_ratio = 0.0

    # Market making metrics
    inventory_turnover = calculate_inventory_turnover(position_series, timestamps)

    # Realized spread (simplified)
    realized_spread = pnl_per_trade * 2 if pnl_per_trade > 0 else 0.0

    # Adverse selection (simplified: volatility of unrealized PnL)
    adverse_selection = np.std(position_series) if len(position_series) > 0 else 0.0

    # Execution metrics
    exec_metrics = calculate_execution_metrics(orders, fills, timestamps)

    # Statistical significance
    t_stat, p_val = calculate_t_statistic(returns) if len(returns) > 0 else (0.0, 1.0)

    # Information ratio (simplified: Sharpe with no benchmark)
    info_ratio = sharpe_1hour

    # Omega ratio
    omega = calculate_omega_ratio(returns) if len(returns) > 0 else 0.0

    return HFTMetrics(
        sharpe_15min=sharpe_15min,
        sharpe_30min=sharpe_30min,
        sharpe_1hour=sharpe_1hour,
        sharpe_daily=sharpe_daily,
        max_drawdown=max_drawdown,
        max_drawdown_duration_seconds=dd_duration_seconds,
        calmar_ratio=calmar,
        sortino_ratio=sortino,
        pnl_per_trade=pnl_per_trade,
        pnl_per_contract=pnl_per_contract,
        profit_factor=profit_factor,
        win_rate=win_rate,
        profit_loss_ratio=pl_ratio,
        consecutive_wins_max=max_wins,
        consecutive_losses_max=max_losses,
        pnl_volatility_15min=float(vol_15min) if not np.isnan(vol_15min) else 0.0,
        pnl_volatility_1hour=float(vol_1hour) if not np.isnan(vol_1hour) else 0.0,
        return_to_volatility_ratio=return_vol_ratio,
        inventory_turnover=inventory_turnover,
        realized_spread=realized_spread,
        adverse_selection_cost=adverse_selection,
        fill_rate=exec_metrics['fill_rate'],
        cancel_rate=exec_metrics['cancel_rate'],
        avg_time_to_fill_ms=exec_metrics['avg_time_to_fill_ms'],
        slippage_per_fill=exec_metrics['slippage_per_fill'],
        t_statistic=t_stat,
        p_value=p_val,
        information_ratio=info_ratio,
        omega_ratio=omega,
    )


def print_hft_metrics(metrics: HFTMetrics) -> str:
    """
    Format HFT metrics for display.

    Args:
        metrics: HFTMetrics dataclass

    Returns:
        Formatted string
    """
    lines = [
        "",
        "=" * 70,
        "HIGH-FREQUENCY TRADING PERFORMANCE METRICS",
        "=" * 70,
        "",
        "Risk-Adjusted Returns (Annualized):",
        f"  Sharpe Ratio (15min): {metrics.sharpe_15min:.3f}",
        f"  Sharpe Ratio (30min): {metrics.sharpe_30min:.3f}",
        f"  Sharpe Ratio (1hour): {metrics.sharpe_1hour:.3f}",
        f"  Sharpe Ratio (Daily): {metrics.sharpe_daily:.3f}",
        f"  Sortino Ratio: {metrics.sortino_ratio:.3f}",
        f"  Calmar Ratio: {metrics.calmar_ratio:.3f}",
        f"  Information Ratio: {metrics.information_ratio:.3f}",
        f"  Omega Ratio: {metrics.omega_ratio:.3f}",
        "",
        "Risk Metrics:",
        f"  Max Drawdown: ${metrics.max_drawdown:.2f}",
        f"  Max DD Duration: {metrics.max_drawdown_duration_seconds:.1f}s",
        f"  PnL Volatility (15min): {metrics.pnl_volatility_15min:.4f}",
        f"  PnL Volatility (1hour): {metrics.pnl_volatility_1hour:.4f}",
        f"  Return/Vol Ratio: {metrics.return_to_volatility_ratio:.3f}",
        "",
        "Profitability:",
        f"  PnL per Trade: ${metrics.pnl_per_trade:.4f}",
        f"  PnL per Contract: ${metrics.pnl_per_contract:.4f}",
        f"  Profit Factor: {metrics.profit_factor:.3f}",
        f"  Win Rate: {metrics.win_rate*100:.1f}%",
        f"  Profit/Loss Ratio: {metrics.profit_loss_ratio:.3f}",
        "",
        "Consistency:",
        f"  Max Consecutive Wins: {metrics.consecutive_wins_max}",
        f"  Max Consecutive Losses: {metrics.consecutive_losses_max}",
        "",
        "Market Making:",
        f"  Inventory Turnover: {metrics.inventory_turnover:.2f} /day",
        f"  Realized Spread: ${metrics.realized_spread:.4f}",
        f"  Adverse Selection Cost: ${metrics.adverse_selection_cost:.4f}",
        "",
        "Execution Quality:",
        f"  Fill Rate: {metrics.fill_rate*100:.1f}%",
        f"  Cancel Rate: {metrics.cancel_rate*100:.1f}%",
        f"  Avg Time to Fill: {metrics.avg_time_to_fill_ms:.1f}ms",
        f"  Slippage per Fill: ${metrics.slippage_per_fill:.4f}",
        "",
        "Statistical Significance:",
        f"  T-Statistic: {metrics.t_statistic:.3f}",
        f"  P-Value: {metrics.p_value:.4f}",
        f"  {'**SIGNIFICANT**' if metrics.p_value < 0.05 else 'Not significant'} at 5% level",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)