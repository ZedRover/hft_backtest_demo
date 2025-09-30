"""
Command-line interface for HFT backtest framework.

Provides CLI for running backtests and parameter sweeps.
"""

import argparse
import sys
from pathlib import Path
from ..engine.backtest import BacktestEngine
from ..engine.config import BacktestConfig
from ..strategy.examples.simple_maker import SimpleMaker
from ..strategy.examples.spread_maker import SpreadMaker
from ..analysis.metrics import calculate_all_metrics
from ..analysis.visualization import plot_backtest_summary, save_plot


STRATEGIES = {
    'simple_maker': SimpleMaker,
    'spread_maker': SpreadMaker,
}


def run_backtest_command(args: argparse.Namespace) -> int:
    """
    Run a backtest from command line.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success)
    """
    # Get strategy class
    if args.strategy not in STRATEGIES:
        print(f"Error: Unknown strategy '{args.strategy}'")
        print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
        return 1

    strategy_class = STRATEGIES[args.strategy]

    # Parse strategy parameters
    strategy_params = {}
    if args.params:
        for param in args.params:
            key, value = param.split('=')
            # Try to convert to appropriate type
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            strategy_params[key] = value

    # Create config
    config = BacktestConfig(
        data_path=args.data,
        symbol=args.symbol,
        strategy_class=strategy_class,
        strategy_params=strategy_params,
        max_position=args.max_position,
        maker_fee_rate=args.maker_fee,
        taker_fee_rate=args.taker_fee,
        queue_cancellation_rate=args.cancel_rate,
        random_seed=args.seed,
        log_level=args.log_level,
        save_events=args.save_events
    )

    # Run backtest
    print(f"Running backtest: {strategy_class.__name__} on {args.symbol}")
    engine = BacktestEngine(config)
    result = engine.run()

    # Calculate metrics
    result.metrics = calculate_all_metrics(
        result.pnl_series,
        result.fills,
        result.orders
    )

    # Print summary
    print(result.summary())

    # Save visualization if requested
    if args.plot:
        fig = plot_backtest_summary(
            result.timestamps,
            result.pnl_series,
            result.position_series,
            result.fills
        )
        if fig is not None:
            save_plot(fig, args.plot)

    return 0


def list_strategies_command(args: argparse.Namespace) -> int:
    """List available strategies"""
    print("Available strategies:")
    for name, cls in STRATEGIES.items():
        doc = cls.__doc__ or "No description"
        print(f"  {name}: {doc.strip().split('.')[0]}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="HFT Maker Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run backtest command
    run_parser = subparsers.add_parser('run', help='Run a backtest')
    run_parser.add_argument('data', help='Path to snapshot data file')
    run_parser.add_argument('symbol', help='Trading symbol')
    run_parser.add_argument('strategy', help='Strategy name')
    run_parser.add_argument('--params', nargs='*', help='Strategy parameters (key=value)')
    run_parser.add_argument('--max-position', type=float, default=10.0,
                           help='Maximum position size (default: 10.0)')
    run_parser.add_argument('--maker-fee', type=float, default=0.0001,
                           help='Maker fee rate (default: 0.0001)')
    run_parser.add_argument('--taker-fee', type=float, default=0.0002,
                           help='Taker fee rate (default: 0.0002)')
    run_parser.add_argument('--cancel-rate', type=float, default=0.02,
                           help='Queue cancellation rate (default: 0.02)')
    run_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    run_parser.add_argument('--log-level', default='INFO',
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                           help='Logging level (default: INFO)')
    run_parser.add_argument('--save-events', action='store_true',
                           help='Save detailed event log')
    run_parser.add_argument('--plot', metavar='FILE',
                           help='Save visualization to file')

    # List strategies command
    list_parser = subparsers.add_parser('list', help='List available strategies')

    return parser


def main() -> int:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'run':
        return run_backtest_command(args)
    elif args.command == 'list':
        return list_strategies_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())