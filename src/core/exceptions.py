"""
Custom exception classes for the backtesting framework.

Provides specific error types for better error handling and debugging.
"""


class BacktestError(Exception):
    """Base exception for all backtesting errors."""
    pass


class DataError(BacktestError):
    """Raised when data loading or validation fails."""
    pass


class DataLoadError(DataError):
    """Raised when data file cannot be loaded."""
    pass


class DataValidationError(DataError):
    """Raised when data fails validation checks."""
    
    def __init__(self, message: str, field: str = None, value=None):
        """
        Initialize validation error.
        
        Args:
            message: Error description
            field: Field that failed validation
            value: Invalid value
        """
        self.field = field
        self.value = value
        
        if field:
            full_message = f"Validation failed for field '{field}': {message}"
            if value is not None:
                full_message += f" (value: {value})"
        else:
            full_message = message
            
        super().__init__(full_message)


class ConfigurationError(BacktestError):
    """Raised when configuration is invalid."""
    pass


class OrderError(BacktestError):
    """Base exception for order-related errors."""
    pass


class OrderNotFoundError(OrderError):
    """Raised when order ID is not found."""
    
    def __init__(self, order_id: int):
        self.order_id = order_id
        super().__init__(f"Order {order_id} not found")


class InvalidOrderStateError(OrderError):
    """Raised when order operation is invalid for current state."""
    
    def __init__(self, order_id: int, current_state: str, operation: str):
        self.order_id = order_id
        self.current_state = current_state
        self.operation = operation
        super().__init__(
            f"Cannot {operation} order {order_id} in state {current_state}"
        )


class PositionLimitExceededError(BacktestError):
    """Raised when position limit would be exceeded."""
    
    def __init__(self, current_position: float, new_position: float, limit: float):
        self.current_position = current_position
        self.new_position = new_position
        self.limit = limit
        super().__init__(
            f"Position limit exceeded: {new_position:.2f} > {limit:.2f} "
            f"(current: {current_position:.2f})"
        )


class InsufficientLiquidityError(BacktestError):
    """Raised when market liquidity is insufficient for operation."""
    
    def __init__(self, price: float, side: int, available: float, required: float):
        self.price = price
        self.side = side
        self.available = available
        self.required = required
        side_str = "BID" if side == 1 else "ASK"
        super().__init__(
            f"Insufficient liquidity at {price} {side_str}: "
            f"required {required:.2f}, available {available:.2f}"
        )


class StrategyError(BacktestError):
    """Raised when strategy execution fails."""
    
    def __init__(self, strategy_name: str, message: str):
        self.strategy_name = strategy_name
        super().__init__(f"Strategy '{strategy_name}' error: {message}")


class MicrostructureError(BacktestError):
    """Raised when microstructure model encounters an error."""
    pass


class QueueSimulationError(BacktestError):
    """Raised when queue simulation encounters an error."""
    
    def __init__(self, order_id: int, message: str):
        self.order_id = order_id
        super().__init__(f"Queue simulation error for order {order_id}: {message}")


class FillSimulationError(BacktestError):
    """Raised when fill simulation encounters an error."""
    pass


class MetricsCalculationError(BacktestError):
    """Raised when performance metrics calculation fails."""
    
    def __init__(self, metric_name: str, message: str):
        self.metric_name = metric_name
        super().__init__(f"Failed to calculate metric '{metric_name}': {message}")


class InvalidSnapshotError(DataError):
    """Raised when snapshot data is malformed."""
    
    def __init__(self, timestamp: int, reason: str):
        self.timestamp = timestamp
        self.reason = reason
        super().__init__(f"Invalid snapshot at timestamp {timestamp}: {reason}")


class OrderBookError(BacktestError):
    """Raised when order book operation fails."""
    pass


class PriceNotFoundError(OrderBookError):
    """Raised when price level is not found in order book."""
    
    def __init__(self, price: float, side: int):
        self.price = price
        self.side = side
        side_str = "BID" if side == 1 else "ASK"
        super().__init__(f"Price {price} not found in {side_str} side")
