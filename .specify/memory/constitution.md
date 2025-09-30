<!--
Sync Impact Report:
Version: 0.0.0 → 1.0.0
Change Type: Initial Constitution Creation
Modified Principles: N/A (new document)
Added Sections: All sections (new constitution)
Removed Sections: N/A
Templates Status:
  ✅ plan-template.md: Reviewed - Constitution Check section present
  ✅ spec-template.md: Reviewed - No updates required (tech-agnostic)
  ✅ tasks-template.md: Reviewed - Task categories align with principles
  ✅ agent-file-template.md: Reviewed - No updates required
Follow-up TODOs: None
-->

# HFT Maker Backtesting Framework Constitution

## Core Principles

### I. Performance First (NON-NEGOTIABLE)
High-frequency trading simulation demands extreme computational efficiency. Every component MUST be optimized for speed:
- Core simulation loops MUST use Numba JIT compilation for critical paths
- Data structures MUST use NumPy arrays for vectorized operations
- Memory allocations MUST be minimized within simulation loops (pre-allocate buffers)
- NO Python loops for per-tick operations; use vectorized operations or Numba
- Benchmark performance targets: Process 500ms snapshots at >10,000 ticks/second

**Rationale**: HFT backtesting requires processing millions of ticks. Sub-optimal performance makes the framework unusable for realistic strategy evaluation.

### II. Market Microstructure Realism
Simulations MUST accurately model Chinese futures market mechanics:
- Queue position tracking for each order at each price level
- Order cancellation probability based on queue dynamics
- Fill simulation considering volume ahead in queue
- Tick-by-tick or snapshot-based order matching (500ms intervals)
- Transaction costs (fees, slippage) modeled explicitly

**Rationale**: Inaccurate market simulation leads to over-optimistic backtest results and failed live strategies. Queue position is critical for maker strategies.

### III. Data-Driven Design
All functionality MUST work with the specified data format:
- Input: 500ms snapshots with 5-level order book (bid/ask prices, volumes)
- Cumulative volume fields for volume delta calculations
- Support for multiple contracts/symbols
- Timestamp handling with millisecond precision
- Memory-efficient data loading (chunked/streaming for large datasets)

**Rationale**: Framework exists to process real market data. Design must fit the data format, not force data transformation overhead.

### IV. Strategy Interface Simplicity
Strategy implementation MUST be simple and focused:
- Strategies define: signal generation, order placement logic, risk limits
- Framework handles: order matching, queue simulation, P&L calculation, position tracking
- Clear separation: Strategy logic vs. market simulation vs. execution simulation
- Strategies receive: current snapshot, position state, order state
- Strategies return: order instructions (submit/cancel/modify)

**Rationale**: Complexity in strategy interface leads to errors and limits iteration speed. Researchers should focus on alpha generation, not simulation mechanics.

### V. Reproducibility & Debuggability
Every backtest run MUST be reproducible and debuggable:
- Deterministic simulation (fixed random seed for queue cancellations)
- Detailed event logging (configurable verbosity)
- Trade-by-trade audit trail with timestamps
- Position and P&L snapshots at each step
- Ability to replay specific time windows

**Rationale**: Non-reproducible backtests are worthless for strategy development. Debugging requires detailed event history.

### VI. Vectorized Architecture
Prefer batch/vectorized operations over iterative processing:
- Process multiple snapshots in batches where possible
- Use NumPy broadcasting for calculations across price levels
- Numba parallel execution for independent symbol backtests
- Avoid Python object creation in hot loops

**Rationale**: Vectorization leverages modern CPU SIMD instructions and reduces Python interpreter overhead, critical for performance targets.

## Technical Standards

### Performance Requirements
- **Throughput**: Process ≥10,000 snapshots/second per strategy (single core)
- **Memory**: Support full-day backtests (≥28,800 snapshots @ 500ms) in <8GB RAM
- **Latency**: Snapshot processing time <100μs (95th percentile)
- **Scalability**: Support parallel backtesting of multiple strategies/parameters

### Code Quality Standards
- Type hints required for all public APIs (enforced with mypy)
- Numba-decorated functions must have explicit signatures
- NumPy array operations preferred over pandas for hot paths
- Profile-guided optimization: Benchmark before and after changes
- Unit tests for all simulation logic (queue, matching, P&L)
- Integration tests with synthetic and historical data

### Data Standards
- Input data format: Structured NumPy arrays or Pandas DataFrame (converted to NumPy)
- Required fields: timestamp, bid_prices[5], bid_volumes[5], ask_prices[5], ask_volumes[5], cumulative_volume
- Optional fields: open_interest, last_price, total_volume
- Missing data handling: Explicit forward-fill or skip strategy
- Data validation on load (check monotonic timestamps, price sanity)

## Development Workflow

### Implementation Order
1. **Core Simulation Engine** (highest priority)
   - Order book snapshot representation
   - Queue position model
   - Fill simulation logic
   - P&L calculator
2. **Data Pipeline**
   - Snapshot data loader
   - Data validation
   - Memory-efficient iteration
3. **Strategy Interface**
   - Base strategy class
   - Signal generation hooks
   - Order management interface
4. **Backtesting Framework**
   - Event loop
   - State management
   - Results collection
5. **Analysis & Reporting**
   - Performance metrics
   - Trade analysis
   - Visualization utilities

### Testing Requirements
- TDD for all simulation logic (write tests before implementation)
- Test cases MUST include:
  - Queue position calculation edge cases
  - Fill simulation with partial fills
  - P&L calculation with fees
  - Order state transitions
- Performance regression tests (benchmark suite)
- Integration test with 1-hour sample data

### Optimization Workflow
1. Implement correct logic (Python, readable)
2. Write comprehensive tests
3. Profile to identify bottlenecks
4. Apply Numba JIT to hot functions
5. Vectorize operations where possible
6. Verify performance improvement (>5x minimum)
7. Ensure tests still pass

## Governance

### Amendment Process
- Constitution changes require documented rationale and impact analysis
- Performance principle violations MUST include benchmark justification
- Major version bump for principle changes
- Minor version bump for new sections or significant clarifications
- Patch version bump for typos, wording improvements

### Compliance Requirements
- All code reviews MUST verify compliance with Performance First and Market Microstructure Realism principles
- New features MUST NOT degrade simulation throughput by >10%
- Simulation accuracy changes MUST be validated against known market scenarios
- Optimization changes MUST include before/after benchmarks

### Decision Authority
- Performance targets and market model accuracy are non-negotiable
- Strategy interface changes require validation with example strategies
- Data format extensions must maintain backward compatibility
- Framework complexity must be justified against user value

**Version**: 1.0.0 | **Ratified**: 2025-09-29 | **Last Amended**: 2025-09-29