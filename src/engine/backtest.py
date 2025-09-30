"""
Main backtest engine.

Integrates all components to execute backtests with realistic queue simulation.
"""

import time
import numpy as np
from typing import Optional
from .config import BacktestConfig
from .state import BacktestState
from .events import EventLogger
from ..data.loader import load_snapshots
from ..data.validator import validate_snapshots
from ..strategy.context import StrategyContext
from ..strategy.actions import SubmitOrder, CancelOrder, ModifyOrder
from ..core.queue import QueueSimulator, initialize_queue_position
from ..core.orderbook import OrderBook, get_mid_price, calculate_volume_traded_at_price
from ..core.orderbook_state import OrderBookState
from ..core.microstructure import MicrostructureParams
from ..core.pnl import update_position_with_fill, calculate_fill_cost
from ..analysis.report import BacktestResult
from ..analysis.hft_metrics import calculate_all_hft_metrics


class BacktestEngine:
    """
    Main backtesting engine.

    Executes backtest with realistic queue simulation and comprehensive
    state tracking.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.logger = EventLogger(config.log_level, config.save_events)
        self.queue_sim = QueueSimulator(config.random_seed)

        # Create microstructure params from config
        microstructure_params = MicrostructureParams(
            trade_probability=config.microstructure_trade_probability,
            add_probability=config.microstructure_add_probability,
            cancel_loc_alpha=config.microstructure_cancel_alpha,
            cancel_loc_beta=config.microstructure_cancel_beta,
            trade_aggression=config.microstructure_trade_aggression
        )

        # NEW: OrderBook that maintains clean market state (without our orders)
        self.orderbook = OrderBook()

        # OrderBookState for microstructure model
        self.orderbook_state = OrderBookState(
            microstructure_params=microstructure_params,
            random_seed=config.random_seed
        )

        # Initialize state
        self.state: Optional[BacktestState] = None
        self.strategy: Optional[object] = None
        self.snapshots: Optional[np.ndarray] = None

    def run(self) -> BacktestResult:
        """
        Execute backtest.

        Returns:
            BacktestResult with all performance metrics

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If data file not found
            RuntimeError: If backtest execution fails
        """
        start_time = time.time()

        try:
            # Load and validate data
            self.logger.logger.info(f"Loading data from {self.config.data_path}")
            self.snapshots = load_snapshots(
                self.config.data_path,
                symbol=self.config.symbol,
                validate=True
            )
            self.logger.logger.info(f"Loaded {len(self.snapshots)} snapshots")

            # Initialize components
            self.state = BacktestState(self.config.symbol)
            self.strategy = self.config.strategy_class(**self.config.strategy_params)

            # Run simulation
            self._execute_simulation()

            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000
            throughput = len(self.snapshots) / (processing_time_ms / 1000) if processing_time_ms > 0 else 0

            # Calculate HFT metrics
            pnl_series = np.array(self.state.pnl_series)
            position_series = np.array(self.state.position_series)
            timestamps = np.array(self.state.timestamps)
            fills = self.state.get_fills_array()
            orders = self.state.get_orders_array()

            hft_metrics = calculate_all_hft_metrics(
                pnl_series, position_series, timestamps, fills, orders
            )

            # Convert HFT metrics to dict
            metrics_dict = {
                'sharpe_15min': hft_metrics.sharpe_15min,
                'sharpe_30min': hft_metrics.sharpe_30min,
                'sharpe_1hour': hft_metrics.sharpe_1hour,
                'sharpe_daily': hft_metrics.sharpe_daily,
                'sortino_ratio': hft_metrics.sortino_ratio,
                'calmar_ratio': hft_metrics.calmar_ratio,
                'max_drawdown': hft_metrics.max_drawdown,
                'profit_factor': hft_metrics.profit_factor,
                'win_rate': hft_metrics.win_rate,
                'pnl_per_trade': hft_metrics.pnl_per_trade,
                'inventory_turnover': hft_metrics.inventory_turnover,
                't_statistic': hft_metrics.t_statistic,
                'p_value': hft_metrics.p_value,
            }

            # Build result
            result = BacktestResult(
                strategy_name=self.config.strategy_class.__name__,
                symbol=self.config.symbol,
                start_time=int(self.snapshots[0]['timestamp']),
                end_time=int(self.snapshots[-1]['timestamp']),
                total_pnl=self.state.position.total_pnl,
                realized_pnl=self.state.position.realized_pnl,
                unrealized_pnl=self.state.position.unrealized_pnl,
                final_position=self.state.position.quantity,
                total_trades=len(self.state.fills_list),
                total_orders=self.state.total_orders,
                filled_orders=self.state.filled_orders,
                cancelled_orders=self.state.cancelled_orders,
                rejected_orders=self.state.rejected_orders,
                snapshots_processed=len(self.snapshots),
                processing_time_ms=processing_time_ms,
                throughput_snapshots_per_sec=throughput,
                pnl_series=pnl_series,
                position_series=position_series,
                timestamps=timestamps,
                fills=fills,
                orders=orders,
                metrics=metrics_dict,
                hft_metrics=hft_metrics  # Store full HFT metrics
            )

            self.logger.logger.info(f"Backtest complete: {throughput:.0f} snapshots/sec")
            return result

        except Exception as e:
            self.logger.logger.error(f"Backtest failed: {e}")
            raise RuntimeError(f"Backtest execution failed: {e}") from e

    def _execute_simulation(self) -> None:
        """Execute main simulation loop"""
        for i in range(len(self.snapshots)):
            snapshot = self.snapshots[i]
            timestamp = int(snapshot['timestamp'])

            # Get previous snapshot for fill simulation
            snapshot_prev = self.snapshots[i-1] if i > 0 else snapshot

            # Update orderbook: extract clean market volumes (snapshot - our orders)
            self.orderbook.update_from_snapshot(snapshot)

            # Also update orderbook_state for microstructure model
            self.orderbook_state.update_from_snapshot(snapshot)

            # Update queue positions for active orders
            self._update_queue_positions(snapshot, snapshot_prev)

            # Simulate fills
            self._simulate_fills(snapshot, snapshot_prev, timestamp)

            # Update unrealized P&L
            mid_price = get_mid_price(snapshot)
            self.state.update_unrealized_pnl(mid_price, self.config.contract_multiplier)

            # Create strategy context
            context = self._create_strategy_context(snapshot, timestamp)

            # Call strategy
            actions = self.strategy.on_snapshot(context)

            # Process strategy actions
            self._process_actions(actions, snapshot, timestamp)

            # Record snapshot
            self.state.record_snapshot(timestamp)

            # Log progress
            self.logger.log_snapshot_processed(timestamp, i + 1)

    def _update_queue_positions(
        self,
        snapshot: np.ndarray,
        snapshot_prev: np.ndarray
    ) -> None:
        """Update queue positions for all active orders using microstructure model"""
        for order in self.state.get_active_orders():
            order_id = int(order['order_id'])
            order_price = float(order['price'])
            order_side = int(order['side'])
            current_vol_ahead = float(order['volume_ahead'])

            # Update queue position using microstructure decomposition
            new_vol_ahead, _, change = self.orderbook_state.update_queue_position(
                order_price,
                order_side,
                current_vol_ahead,
                snapshot_prev,
                snapshot
            )

            new_pos = int(new_vol_ahead) if new_vol_ahead > 0 else 0
            self.state.update_order_queue(order_id, new_pos, new_vol_ahead)

            # Log queue update with decomposition details
            timestamp = int(snapshot['timestamp'])
            if change is not None and self.logger.log_level == 'DEBUG':
                self.logger.logger.debug(
                    f"Queue update for order {order_id}: "
                    f"pos {current_vol_ahead:.1f} -> {new_vol_ahead:.1f}, "
                    f"traded={change.traded:.1f}, "
                    f"cancel_before={change.cancelled_before_us:.1f}, "
                    f"cancel_after={change.cancelled_after_us:.1f}, "
                    f"added={change.added:.1f}"
                )
            self.logger.log_queue_updated(timestamp, order_id, new_pos, new_vol_ahead)

    def _simulate_fills(
        self,
        snapshot: np.ndarray,
        snapshot_prev: np.ndarray,
        timestamp: int
    ) -> None:
        """Simulate fills for active orders using orderbook state"""
        for order in self.state.get_active_orders():
            order_id = int(order['order_id'])
            order_price = float(order['price'])
            order_quantity = float(order['remaining_quantity'])
            order_side = int(order['side'])
            volume_ahead = float(order['volume_ahead'])

            # Calculate volume traded at this price
            volume_traded = self.orderbook_state.calculate_volume_traded_at_price(
                snapshot_prev,
                snapshot,
                order_price,
                order_side
            )

            # Check if order gets filled using orderbook state
            fill_result = self.orderbook_state.check_fill(
                order_price,
                order_quantity,
                order_side,
                volume_ahead,
                volume_traded,
                snapshot_prev,
                snapshot
            )

            if fill_result is not None:
                fill_price, fill_quantity = fill_result

                # Calculate fee
                notional = fill_quantity * fill_price * self.config.contract_multiplier
                fee = calculate_fill_cost(
                    notional,
                    self.config.maker_fee_rate,
                    is_maker=True
                )

                # Record fill
                self.state.record_fill(
                    order_id,
                    timestamp,
                    fill_price,
                    fill_quantity,
                    order_side,
                    fee,
                    is_maker=True
                )

                # Update position
                new_qty, new_avg, new_realized = update_position_with_fill(
                    self.state.position.quantity,
                    self.state.position.average_entry_price,
                    self.state.position.realized_pnl,
                    fill_quantity,
                    fill_price,
                    order_side,
                    fee,
                    self.config.contract_multiplier
                )

                self.state.update_position(new_qty, new_avg, new_realized)

                # Remove from both orderbooks
                self.orderbook.remove_our_order(order_id, fill_quantity)
                self.orderbook_state.remove_order(order_price, fill_quantity, order_side)

                # Log events
                self.logger.log_order_filled(timestamp, order_id, fill_price, fill_quantity)
                self.logger.log_position_updated(timestamp, new_qty, new_avg, new_realized)

                # Notify strategy
                if hasattr(self.strategy, 'on_fill'):
                    self.strategy.on_fill(order_id, fill_price, fill_quantity)

    def _create_strategy_context(
        self,
        snapshot: np.ndarray,
        timestamp: int
    ) -> StrategyContext:
        """Create strategy context for current snapshot"""
        return StrategyContext(
            snapshot=snapshot,
            position_quantity=self.state.position.quantity,
            position_avg_entry=self.state.position.average_entry_price,
            position_realized_pnl=self.state.position.realized_pnl,
            position_unrealized_pnl=self.state.position.unrealized_pnl,
            active_orders=self.state.get_active_orders(),
            timestamp=timestamp,
            symbol=self.config.symbol
        )

    def _process_actions(
        self,
        actions: list,
        snapshot: np.ndarray,
        timestamp: int
    ) -> None:
        """Process strategy actions"""
        for action in actions:
            if isinstance(action, SubmitOrder):
                self._process_submit_order(action, snapshot, timestamp)
            elif isinstance(action, CancelOrder):
                self._process_cancel_order(action, timestamp)
            elif isinstance(action, ModifyOrder):
                self._process_modify_order(action, snapshot, timestamp)

    def _process_submit_order(
        self,
        action: SubmitOrder,
        snapshot: np.ndarray,
        timestamp: int
    ) -> None:
        """Process submit order action"""
        # Check position limits
        new_position = self.state.position.quantity + (action.quantity * action.side)
        if abs(new_position) > self.config.max_position:
            self.logger.log_order_rejected(
                timestamp,
                -1,
                f"Position limit exceeded: {abs(new_position)} > {self.config.max_position}"
            )
            return

        # Create order first to get order_id
        order_id = self.state.create_order(
            timestamp,
            action.side,
            action.price,
            action.quantity,
            0,  # Will be updated
            0.0  # Will be updated
        )

        # Add to orderbook (tracks our orders and calculates clean market volume)
        vol_ahead, queue_pos = self.orderbook.add_our_order(
            order_id,
            action.price,
            action.quantity,
            action.side
        )

        # Also add to orderbook_state for microstructure model
        self.orderbook_state.add_our_order(
            action.price,
            action.quantity,
            action.side
        )

        # Update order with correct queue position
        self.state.update_order_queue(order_id, int(queue_pos), vol_ahead)

        self.logger.log_order_submitted(
            timestamp,
            order_id,
            action.side,
            action.price,
            action.quantity
        )

    def _process_cancel_order(self, action: CancelOrder, timestamp: int) -> None:
        """Process cancel order action"""
        # Get order info before canceling
        order = self.state.orders.get(action.order_id)
        success = self.state.cancel_order(action.order_id)

        if success and order is not None:
            # Remove from orderbook
            self.orderbook.remove_our_order(action.order_id, float(order['remaining_quantity']))

            self.logger.log_order_cancelled(timestamp, action.order_id)
        else:
            self.logger.log_order_rejected(
                timestamp,
                action.order_id,
                "Order not found or already terminal"
            )

    def _process_modify_order(
        self,
        action: ModifyOrder,
        snapshot: np.ndarray,
        timestamp: int
    ) -> None:
        """Process modify order action (cancel-replace)"""
        # Cancel old order
        old_order = self.state.orders.get(action.order_id)
        if old_order is not None:
            self.state.cancel_order(action.order_id)

            # Submit new order
            side = int(old_order['side'])
            submit = SubmitOrder(side, action.new_price, action.new_quantity)
            self._process_submit_order(submit, snapshot, timestamp)

    def validate_data(self, snapshots: np.ndarray) -> None:
        """
        Validate snapshot data.

        Args:
            snapshots: NumPy array of snapshots

        Raises:
            ValueError: If data validation fails
        """
        validate_snapshots(snapshots)