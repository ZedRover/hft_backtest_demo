"""
Constants and configuration values for the backtesting framework.

Extracts magic numbers into well-named constants for better maintainability.
"""

from enum import IntEnum


# ============================================================================
# Order States
# ============================================================================

class OrderState(IntEnum):
    """Order lifecycle states."""
    PENDING = 0      # Order submitted but not yet confirmed
    PARTIAL = 1      # Order partially filled
    FILLED = 2       # Order completely filled
    CANCELLED = 3    # Order cancelled
    REJECTED = 4     # Order rejected


# ============================================================================
# Order Sides
# ============================================================================

class OrderSide(IntEnum):
    """Order side constants."""
    BID = 1          # Buy order (bid side)
    ASK = -1         # Sell order (ask side)


# ============================================================================
# Microstructure Model Default Parameters
# ============================================================================

# Probability that a volume decrease is a trade vs cancellation
DEFAULT_TRADE_PROBABILITY = 0.7

# Probability that a volume increase is a new order vs replaced order
DEFAULT_ADD_PROBABILITY = 0.6

# Beta distribution alpha parameter for cancellation location
# Higher alpha = cancellations concentrate near front of queue
DEFAULT_CANCEL_ALPHA = 2.0

# Beta distribution beta parameter for cancellation location
# Higher beta = cancellations concentrate near back of queue
DEFAULT_CANCEL_BETA = 3.0

# Trade aggression level (0 = passive, 1 = aggressive)
# Affects how deep into the book trades penetrate
DEFAULT_TRADE_AGGRESSION = 0.8


# ============================================================================
# Queue Position Simulation
# ============================================================================

# Z-score range for random cancellation variance
# Maps [0,1) random value to approximately ±2 standard deviations
CANCELLATION_Z_SCORE_RANGE = 4.0

# Minimum epsilon for floating point comparisons
PRICE_EPSILON = 1e-9

# Fraction of volume decrease attributed to trades (when primarily trades)
TRADE_VOLUME_MIN_FRACTION = 0.7
TRADE_VOLUME_MAX_FRACTION = 1.0

# Fraction of volume decrease attributed to cancellations (when primarily cancels)
CANCEL_VOLUME_MIN_FRACTION = 0.7
CANCEL_VOLUME_MAX_FRACTION = 1.0


# ============================================================================
# Transaction Cost Defaults
# ============================================================================

# Default maker fee rate (1 basis point)
DEFAULT_MAKER_FEE_RATE = 0.0001

# Default taker fee rate (2 basis points)
DEFAULT_TAKER_FEE_RATE = 0.0002

# Default contract multiplier
DEFAULT_CONTRACT_MULTIPLIER = 1.0


# ============================================================================
# Risk Limits
# ============================================================================

# Default maximum position size (contracts)
DEFAULT_MAX_POSITION = 10.0

# Default random seed for reproducibility
DEFAULT_RANDOM_SEED = 42


# ============================================================================
# Order Book Levels
# ============================================================================

# Number of visible price levels on each side
ORDER_BOOK_DEPTH = 5

# Index of best bid/ask (level 0)
BEST_LEVEL_INDEX = 0


# ============================================================================
# Performance Metrics
# ============================================================================

# Time windows for rolling Sharpe ratio calculation (minutes)
SHARPE_WINDOW_15MIN = 15
SHARPE_WINDOW_30MIN = 30
SHARPE_WINDOW_1HOUR = 60
SHARPE_WINDOW_DAILY = 1440  # 24 * 60

# Seconds per day for time conversions
SECONDS_PER_DAY = 86400

# Days per year for annualization
DAYS_PER_YEAR = 365.25

# Milliseconds per second
MILLISECONDS_PER_SECOND = 1000.0


# ============================================================================
# Statistical Significance
# ============================================================================

# P-value threshold for statistical significance
SIGNIFICANCE_LEVEL = 0.05


# ============================================================================
# Microstructure Trade Probability Adjustments
# ============================================================================

# Factor to boost trade probability when cumulative volume is high
TRADE_PROB_HIGH_ACTIVITY_FACTOR = 0.5

# Factor to reduce trade probability when cumulative volume is zero
TRADE_PROB_NO_ACTIVITY_FACTOR = 0.3


# ============================================================================
# Queue Position Ratio Adjustments
# ============================================================================

# Factor when position is ahead of mean (fewer cancels before us)
CANCEL_RATIO_AHEAD_OF_MEAN = 0.7

# Factor when position is behind mean (more cancels before us)
CANCEL_RATIO_BEHIND_MEAN = 1.3


# ============================================================================
# Logging
# ============================================================================

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Whether to save events by default
DEFAULT_SAVE_EVENTS = False


# ============================================================================
# Data Validation
# ============================================================================

# Maximum symbol length
MAX_SYMBOL_LENGTH = 10

# Maximum reasonable spread (as multiple of tick size)
MAX_REASONABLE_SPREAD_TICKS = 1000


# ============================================================================
# Fill Simulation
# ============================================================================

# Minimum fill quantity threshold (to avoid rounding errors)
MIN_FILL_QUANTITY = 1e-9
