"""
Run backtest example.

Demonstrates full backtest workflow from quickstart.md.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.backtest import BacktestEngine
from src.engine.config import BacktestConfig
from src.strategy.examples.simple_maker import SimpleMaker


def main():
    """Run example backtest"""
    # Path to sample data
    data_path = Path(__file__).parent / 'data' / 'sample_snapshots.parquet'

    if not data_path.exists():
        print(f"Error: Sample data not found at {data_path}")
        print("Run examples/create_sample_data.py first to generate sample data")
        return 1

    # Configure backtest
    print("Configuring backtest...")
    config = BacktestConfig(
        data_path=str(data_path),
        symbol='IF2401',
        strategy_class=SimpleMaker,
        strategy_params={'spread_bps': 10, 'quote_size': 1.0},
        max_position=10.0,
        maker_fee_rate=0.0001,  # 1 basis point maker fee
        taker_fee_rate=0.0002,  # 2 basis point taker fee
        queue_cancellation_rate=0.02,  # 2% cancellation per snapshot
        random_seed=42,
        log_level='INFO',
    )

    # Run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine(config)
    result = engine.run()

    # Print results with HFT metrics
    print(result.summary_with_hft_metrics())

    # Additional analysis
    print("\nAdditional Statistics:")
    print(f"  Average P&L per snapshot: ${result.total_pnl / result.snapshots_processed:.4f}")

    if result.total_trades > 0:
        print(f"  Average fill price: ${result.fills['price'].mean():.2f}")
        print(f"  Fill price std dev: ${result.fills['price'].std():.2f}")

    print("\nBacktest complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())