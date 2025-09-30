"""
Create sample snapshot data for testing.

Generates synthetic order book snapshots for backtesting.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_snapshots(
    n_snapshots: int = 1000,
    base_price: float = 5000.0,
    spread: float = 2.0,
    volatility: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic snapshot data.

    Args:
        n_snapshots: Number of snapshots to generate
        base_price: Base price for random walk
        spread: Bid-ask spread
        volatility: Price volatility
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with snapshot data
    """
    np.random.seed(random_seed)

    # Time vector (500ms intervals)
    timestamps = np.arange(n_snapshots) * 500  # milliseconds

    data = []
    cumulative_vol = 0.0
    cumulative_turnover = 0.0

    for i, ts in enumerate(timestamps):
        # Random walk for mid price
        if i == 0:
            mid = base_price
        else:
            mid = data[-1]['mid'] + np.random.randn() * volatility

        # Generate order book
        snapshot = {
            'timestamp': ts,
            'symbol': 'IF2401',
            # Bid side (best to 5th level)
            'bid_price_0': mid - spread / 2,
            'bid_price_1': mid - spread / 2 - 0.5,
            'bid_price_2': mid - spread / 2 - 1.0,
            'bid_price_3': mid - spread / 2 - 1.5,
            'bid_price_4': mid - spread / 2 - 2.0,
            'bid_volume_0': 10 + np.random.randint(0, 20),
            'bid_volume_1': 8 + np.random.randint(0, 15),
            'bid_volume_2': 6 + np.random.randint(0, 10),
            'bid_volume_3': 4 + np.random.randint(0, 8),
            'bid_volume_4': 2 + np.random.randint(0, 5),
            # Ask side (best to 5th level)
            'ask_price_0': mid + spread / 2,
            'ask_price_1': mid + spread / 2 + 0.5,
            'ask_price_2': mid + spread / 2 + 1.0,
            'ask_price_3': mid + spread / 2 + 1.5,
            'ask_price_4': mid + spread / 2 + 2.0,
            'ask_volume_0': 10 + np.random.randint(0, 20),
            'ask_volume_1': 8 + np.random.randint(0, 15),
            'ask_volume_2': 6 + np.random.randint(0, 10),
            'ask_volume_3': 4 + np.random.randint(0, 8),
            'ask_volume_4': 2 + np.random.randint(0, 5),
            # Additional fields
            'cumulative_volume': cumulative_vol,
            'turnover_volume': cumulative_turnover,
            'last_price': mid,
            'mid': mid  # For tracking
        }

        # Update cumulative volume and turnover
        vol_delta = 100 + np.random.randint(0, 50)
        cumulative_vol += vol_delta
        cumulative_turnover += vol_delta * mid

        data.append(snapshot)

    df = pd.DataFrame(data)
    # Remove temporary 'mid' column
    df = df.drop(columns=['mid'])

    return df


def main():
    """Generate and save sample data"""
    # Ensure data directory exists
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)

    # Generate 50000 snapshots (~7 hours of data at 500ms intervals)
    print("Generating synthetic snapshot data...")
    df = generate_sample_snapshots(n_snapshots=50000)

    # Save as Parquet (recommended)
    parquet_path = data_dir / 'sample_snapshots.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} snapshots to {parquet_path}")

    # Also save as CSV for compatibility
    csv_path = data_dir / 'sample_snapshots.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} snapshots to {csv_path}")

    # Print sample
    print("\nSample snapshots (first 3):")
    print(df.head(3))

    print("\nData statistics:")
    print(f"  Duration: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1000:.1f} seconds")
    print(f"  Price range: {df['bid_price_0'].min():.2f} - {df['ask_price_0'].max():.2f}")
    print(f"  File size: {parquet_path.stat().st_size / 1024:.1f} KB (Parquet)")


if __name__ == '__main__':
    main()