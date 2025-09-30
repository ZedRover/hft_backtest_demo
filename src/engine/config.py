"""
Backtest configuration.

Defines BacktestConfig dataclass with all parameters needed for backtesting.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.

    Contains all parameters needed to execute a backtest:
    - Data source
    - Strategy specification
    - Risk limits
    - Transaction costs
    - Queue simulation parameters
    - Execution settings
    """

    # Data source
    data_path: str
    symbol: str

    # Strategy
    strategy_class: Type
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Risk limits
    max_position: float = 10.0
    max_loss: Optional[float] = None  # Stop loss threshold

    # Transaction costs
    maker_fee_rate: float = 0.0001  # 1 basis point default
    taker_fee_rate: float = 0.0002  # 2 basis points default
    contract_multiplier: float = 1.0

    # Queue simulation
    queue_cancellation_rate: float = 0.02  # 2% per snapshot (legacy, for simple model)
    queue_model: str = "probabilistic"  # or "deterministic"

    # Microstructure model parameters
    microstructure_trade_probability: float = 0.7  # P(decrease is trade vs cancel)
    microstructure_add_probability: float = 0.6    # P(increase is new order vs replaced)
    microstructure_cancel_alpha: float = 2.0       # Beta dist alpha (front weight)
    microstructure_cancel_beta: float = 3.0        # Beta dist beta (back weight)
    microstructure_trade_aggression: float = 0.8   # Trade aggression level

    # Execution
    initial_capital: Optional[float] = None  # Optional capital tracking
    random_seed: int = 42  # For reproducibility

    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    save_events: bool = False  # Save detailed event log

    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        if self.max_position <= 0:
            raise ValueError(f"max_position must be positive: {self.max_position}")

        if self.queue_cancellation_rate < 0 or self.queue_cancellation_rate > 1:
            raise ValueError(
                f"queue_cancellation_rate must be in [0, 1]: {self.queue_cancellation_rate}"
            )

        if self.queue_model not in ("probabilistic", "deterministic"):
            raise ValueError(
                f"queue_model must be 'probabilistic' or 'deterministic': {self.queue_model}"
            )

        if self.contract_multiplier <= 0:
            raise ValueError(
                f"contract_multiplier must be positive: {self.contract_multiplier}"
            )

        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError(f"Invalid log_level: {self.log_level}")


@dataclass
class BacktestMetrics:
    """
    Performance metrics for a backtest.

    Calculated from backtest results.
    """
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # snapshots
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0