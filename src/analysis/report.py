"""
Backtest result and reporting.

Defines BacktestResult dataclass and result processing.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hft_metrics import HFTMetrics


@dataclass
class BacktestResult:
    """
    Complete backtest results.

    Contains all data from backtest execution including trades,
    P&L series, and performance metrics.
    """

    # Metadata
    strategy_name: str
    symbol: str
    start_time: int  # milliseconds
    end_time: int  # milliseconds

    # Performance
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    final_position: float

    # Trade statistics
    total_trades: int
    total_orders: int
    filled_orders: int
    cancelled_orders: int
    rejected_orders: int

    # Execution statistics
    snapshots_processed: int
    processing_time_ms: float
    throughput_snapshots_per_sec: float

    # Time series data
    pnl_series: np.ndarray  # P&L at each snapshot
    position_series: np.ndarray  # Position at each snapshot
    timestamps: np.ndarray  # Timestamp for each snapshot

    # Detailed records
    fills: np.ndarray  # Structured array of fills (FILL_DTYPE)
    orders: np.ndarray  # Structured array of orders (ORDER_DTYPE)

    # Metrics
    metrics: Dict[str, float]  # Performance metrics
    hft_metrics: Optional['HFTMetrics'] = None  # Detailed HFT metrics

    @property
    def duration_seconds(self) -> float:
        """Get backtest duration in seconds"""
        return (self.end_time - self.start_time) / 1000.0

    @property
    def fill_rate(self) -> float:
        """Get order fill rate"""
        return self.filled_orders / self.total_orders if self.total_orders > 0 else 0.0

    def summary(self) -> str:
        """
        Generate summary string.

        Returns:
            Multi-line summary of backtest results
        """
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Strategy: {self.strategy_name}",
            f"Symbol: {self.symbol}",
            f"Duration: {self.duration_seconds:.1f} seconds",
            "",
            "P&L Summary:",
            f"  Total P&L: ${self.total_pnl:.2f}",
            f"  Realized P&L: ${self.realized_pnl:.2f}",
            f"  Unrealized P&L: ${self.unrealized_pnl:.2f}",
            f"  Final Position: {self.final_position:.2f}",
            "",
            "Trade Statistics:",
            f"  Total Orders: {self.total_orders}",
            f"  Filled Orders: {self.filled_orders}",
            f"  Total Fills: {self.total_trades}",
            f"  Cancelled Orders: {self.cancelled_orders}",
            f"  Fill Rate: {self.fill_rate*100:.1f}%",
            "",
            "Performance Metrics:",
            f"  Sharpe Ratio (15min): {self.metrics.get('sharpe_15min', 0):.3f}",
            f"  Sharpe Ratio (1hour): {self.metrics.get('sharpe_1hour', 0):.3f}",
            f"  Sortino Ratio: {self.metrics.get('sortino_ratio', 0):.3f}",
            f"  Max Drawdown: ${self.metrics.get('max_drawdown', 0):.2f}",
            f"  Win Rate: {self.metrics.get('win_rate', 0)*100:.1f}%",
            f"  Profit Factor: {self.metrics.get('profit_factor', 0):.3f}",
            f"  P-Value: {self.metrics.get('p_value', 1.0):.4f}",
            "",
            "Execution Performance:",
            f"  Snapshots Processed: {self.snapshots_processed}",
            f"  Processing Time: {self.processing_time_ms:.1f} ms",
            f"  Throughput: {self.throughput_snapshots_per_sec:.0f} snapshots/sec",
            "=" * 60,
        ]
        return "\n".join(lines)

    def summary_with_hft_metrics(self) -> str:
        """
        Generate summary with detailed HFT metrics.

        Returns:
            Multi-line summary including HFT metrics
        """
        basic_summary = self.summary()

        if self.hft_metrics is not None:
            from .hft_metrics import print_hft_metrics
            hft_summary = print_hft_metrics(self.hft_metrics)
            return basic_summary + "\n" + hft_summary
        else:
            return basic_summary