"""
Visualize backtest results.

Creates plots from quickstart.md example.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.backtest import BacktestEngine
from src.engine.config import BacktestConfig
from src.strategy.examples.simple_maker import SimpleMaker
from src.analysis.metrics import calculate_all_metrics
from src.analysis.visualization import plot_backtest_summary, save_plot


def main():
    """Run backtest and visualize results"""
    # Path to sample data
    data_path = Path(__file__).parent / 'data' / 'sample_snapshots.parquet'

    if not data_path.exists():
        print(f"Error: Sample data not found at {data_path}")
        print("Run examples/create_sample_data.py first to generate sample data")
        return 1

    # Configure and run backtest
    print("Running backtest...")
    config = BacktestConfig(
        data_path=str(data_path),
        symbol='IF2401',
        strategy_class=SimpleMaker,
        strategy_params={'spread_bps': 10, 'quote_size': 1.0},
        max_position=10.0,
        maker_fee_rate=0.0001,
        queue_cancellation_rate=0.02,
        random_seed=42,
        log_level='WARNING',  # Less verbose for plotting
    )

    engine = BacktestEngine(config)
    result = engine.run()

    # Calculate metrics
    result.metrics = calculate_all_metrics(
        result.pnl_series,
        result.fills,
        result.orders
    )

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_backtest_summary(
        result.timestamps,
        result.pnl_series,
        result.position_series,
        result.fills
    )

    if fig is not None:
        # Save plot
        output_path = Path(__file__).parent / 'backtest_results.png'
        save_plot(fig, str(output_path))
        print(f"\nVisualization saved to {output_path}")

        # Try to display (will work in interactive environments)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except:
            pass
    else:
        print("\nError: Could not create visualization (matplotlib not available)")
        print("Install with: pip install matplotlib")

    return 0


if __name__ == '__main__':
    sys.exit(main())