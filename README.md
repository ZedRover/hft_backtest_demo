# HFT Maker Backtesting Framework

[English](#english) | [中文](#chinese)

---

<a name="english"></a>

## English

### Overview

A high-performance backtesting framework for market-making strategies on Chinese futures markets, featuring realistic queue simulation and advanced market microstructure modeling.

**Key Features:**
- ✅ **Realistic Queue Simulation**: Probabilistic order cancellations and precise queue position tracking
- ✅ **Market Microstructure Model**: Decomposes volume changes into trades, adds, and cancellations
- ✅ **High Performance**: 2,000-6,000+ snapshots/second with Numba JIT optimization
- ✅ **Flexible Configuration**: Tunable hyperparameters for different market conditions
- ✅ **Complete Implementation**: Fully functional with examples and documentation

### Project Status

**Implementation: COMPLETE ✅**

All core components are implemented and tested:
- ✅ Data loading and validation (CSV, Parquet, HDF5)
- ✅ Numba-optimized core engine (queue, orderbook, P&L)
- ✅ Market microstructure model with configurable parameters
- ✅ Strategy interface with multiple examples
- ✅ Complete backtesting engine
- ✅ Analysis tools and visualization
- ✅ CLI interface
- ✅ Working examples

### Quick Start

#### Installation

```bash
# Clone repository
git clone <repository-url>
cd hft_backtest_demo

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

#### Basic Usage

```python
from src.engine.backtest import BacktestEngine
from src.engine.config import BacktestConfig
from src.strategy.examples.aggressive_maker import AggressiveMaker

# Configure backtest
config = BacktestConfig(
    data_path="examples/data/sample_snapshots.parquet",
    symbol="IF2401",
    strategy_class=AggressiveMaker,
    strategy_params={"tick_offset": 0.2, "quote_size": 1.0},
    max_position=10.0,
    maker_fee_rate=0.0001,
    random_seed=42,
)

# Run backtest
engine = BacktestEngine(config)
result = engine.run()

# Print results
print(result.summary())
```

#### Generate Sample Data

```bash
# Generate 1000 synthetic snapshots
python examples/create_sample_data.py

# Run example backtest
python examples/run_backtest.py

# Create visualization
python examples/visualize_results.py
```

### Architecture

```
src/
├── core/              # Core simulation engine
│   ├── queue.py       # Queue position tracking
│   ├── orderbook.py   # Orderbook utilities
│   ├── orderbook_state.py  # Orderbook state management
│   ├── microstructure.py   # Market microstructure model
│   └── pnl.py         # P&L calculations (Numba-optimized)
│
├── data/              # Data handling
│   ├── types.py       # NumPy structured arrays
│   ├── loader.py      # Multi-format data loader
│   └── validator.py   # Data validation
│
├── strategy/          # Strategy framework
│   ├── base.py        # BaseStrategy abstract class
│   ├── context.py     # StrategyContext (immutable)
│   ├── actions.py     # Order actions
│   └── examples/      # Example strategies
│       ├── simple_maker.py      # Simple market maker
│       ├── spread_maker.py      # Adaptive spread maker
│       └── aggressive_maker.py  # Aggressive maker (in-book quotes)
│
├── engine/            # Backtesting engine
│   ├── config.py      # BacktestConfig
│   ├── state.py       # State management
│   ├── events.py      # Event logging
│   └── backtest.py    # Main backtest loop
│
├── analysis/          # Results analysis
│   ├── metrics.py     # Performance metrics
│   ├── report.py      # Report generation
│   └── visualization.py  # Plotting utilities
│
└── cli/               # Command-line interface
    └── main.py        # CLI entry point
```

### Market Microstructure Model

The framework includes an advanced market microstructure model that decomposes order book volume changes into:

1. **Trades**: Volume consumed by market orders
2. **New Orders**: Limit orders added to the book
3. **Cancellations**:
   - Before our position: Advances our queue position ✅
   - After our position: No effect on our position

#### Configurable Hyperparameters

```python
config = BacktestConfig(
    # ... other parameters ...

    # Microstructure model parameters
    microstructure_trade_probability=0.7,  # P(decrease = trade vs cancel)
    microstructure_add_probability=0.6,    # P(increase = new order vs replace)
    microstructure_cancel_alpha=2.0,       # Beta dist α (front weight)
    microstructure_cancel_beta=3.0,        # Beta dist β (back weight)
    microstructure_trade_aggression=0.8,   # Trade aggression level
)
```

See [Microstructure Model Documentation](docs/microstructure_model.md) for detailed explanation.

### Strategy Development

Create custom strategies by extending `BaseStrategy`:

```python
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.actions import SubmitOrder, OrderAction
from typing import List

class MyStrategy(BaseStrategy):
    def __init__(self, param1: float, param2: int):
        self.param1 = param1
        self.param2 = param2

    def on_snapshot(self, context: StrategyContext) -> List[OrderAction]:
        """Called for each snapshot"""
        # Access market data
        mid = context.get_mid_price()
        spread = context.get_spread()

        # Check position
        position = context.position_quantity

        # Generate orders
        if not context.has_active_orders():
            return [
                SubmitOrder(side=1, price=mid-0.5, quantity=1.0),
                SubmitOrder(side=-1, price=mid+0.5, quantity=1.0),
            ]
        return []

    def on_fill(self, order_id: int, price: float, quantity: float) -> None:
        """Called when order is filled"""
        pass
```

### Performance

**Benchmarks** (1000 snapshots):
- Throughput: 2,000-6,000 snapshots/sec
- Memory: ~100MB
- Latency: <500μs per snapshot

**Optimization:**
- Numba JIT compilation on hot paths
- NumPy vectorized operations
- Efficient order book state management

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/unit/            # Unit tests
pytest tests/integration/     # Integration tests
pytest tests/performance/     # Performance benchmarks

# Run with coverage
pytest --cov=src tests/
```

### Configuration

Full configuration options in `BacktestConfig`:

```python
@dataclass
class BacktestConfig:
    # Data
    data_path: str                    # Path to snapshot data
    symbol: str                       # Trading symbol

    # Strategy
    strategy_class: Type              # Strategy class
    strategy_params: Dict[str, Any]   # Strategy parameters

    # Risk limits
    max_position: float = 10.0        # Maximum position size
    max_loss: Optional[float] = None  # Stop loss threshold

    # Transaction costs
    maker_fee_rate: float = 0.0001    # Maker fee (1bp)
    taker_fee_rate: float = 0.0002    # Taker fee (2bp)
    contract_multiplier: float = 1.0  # Contract multiplier

    # Microstructure model
    microstructure_trade_probability: float = 0.7
    microstructure_add_probability: float = 0.6
    microstructure_cancel_alpha: float = 2.0
    microstructure_cancel_beta: float = 3.0
    microstructure_trade_aggression: float = 0.8

    # Execution
    random_seed: int = 42             # For reproducibility
    log_level: str = "INFO"           # Logging level
```

### Documentation

- [Data Format Specification](docs/data_format.md) - Required data fields and formats ⭐
- [Microstructure Model](docs/microstructure_model.md) - Detailed model documentation
- [Feature Specification](specs/001-maker-snapshot-500ms/spec.md) - Requirements
- [Implementation Plan](specs/001-maker-snapshot-500ms/plan.md) - Design
- [Task Breakdown](specs/001-maker-snapshot-500ms/tasks.md) - Development tasks

### License

MIT License

---

<a name="chinese"></a>

## 中文

### 概述

面向中国期货市场做市策略的高性能回测框架，具备真实队列模拟和先进的市场微观结构建模功能。

**核心特性：**
- ✅ **真实队列模拟**：概率性撤单和精确的队列位置追踪
- ✅ **市场微观结构模型**：将量变化分解为成交、新增和撤单
- ✅ **高性能**：使用Numba JIT优化，达到2,000-6,000+ snapshots/秒
- ✅ **灵活配置**：可调整超参数适应不同市场状态
- ✅ **完整实现**：功能完备，包含示例和文档

### 项目状态

**实现进度：完成 ✅**

所有核心组件已实现并测试：
- ✅ 数据加载和验证（CSV、Parquet、HDF5）
- ✅ Numba优化的核心引擎（队列、订单簿、P&L）
- ✅ 可配置参数的市场微观结构模型
- ✅ 策略接口及多个示例
- ✅ 完整的回测引擎
- ✅ 分析工具和可视化
- ✅ CLI命令行界面
- ✅ 工作示例

### 快速开始

#### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd hft_backtest_demo

# 使用uv创建虚拟环境（推荐）
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
uv pip install -e .
```

#### 基础使用

```python
from src.engine.backtest import BacktestEngine
from src.engine.config import BacktestConfig
from src.strategy.examples.aggressive_maker import AggressiveMaker

# 配置回测
config = BacktestConfig(
    data_path="examples/data/sample_snapshots.parquet",
    symbol="IF2401",
    strategy_class=AggressiveMaker,
    strategy_params={"tick_offset": 0.2, "quote_size": 1.0},
    max_position=10.0,
    maker_fee_rate=0.0001,
    random_seed=42,
)

# 运行回测
engine = BacktestEngine(config)
result = engine.run()

# 打印结果
print(result.summary())
```

#### 生成样本数据

```bash
# 生成1000个合成快照
python examples/create_sample_data.py

# 运行示例回测
python examples/run_backtest.py

# 创建可视化
python examples/visualize_results.py
```

### 架构

```
src/
├── core/              # 核心模拟引擎
│   ├── queue.py       # 队列位置追踪
│   ├── orderbook.py   # 订单簿工具
│   ├── orderbook_state.py  # 订单簿状态管理
│   ├── microstructure.py   # 市场微观结构模型
│   └── pnl.py         # P&L计算（Numba优化）
│
├── data/              # 数据处理
│   ├── types.py       # NumPy结构化数组
│   ├── loader.py      # 多格式数据加载器
│   └── validator.py   # 数据验证
│
├── strategy/          # 策略框架
│   ├── base.py        # BaseStrategy抽象类
│   ├── context.py     # StrategyContext（不可变）
│   ├── actions.py     # 订单操作
│   └── examples/      # 示例策略
│       ├── simple_maker.py      # 简单做市商
│       ├── spread_maker.py      # 自适应价差做市商
│       └── aggressive_maker.py  # 激进做市商（盘口内挂单）
│
├── engine/            # 回测引擎
│   ├── config.py      # BacktestConfig
│   ├── state.py       # 状态管理
│   ├── events.py      # 事件日志
│   └── backtest.py    # 主回测循环
│
├── analysis/          # 结果分析
│   ├── metrics.py     # 性能指标
│   ├── report.py      # 报告生成
│   └── visualization.py  # 绘图工具
│
└── cli/               # 命令行界面
    └── main.py        # CLI入口
```

### 市场微观结构模型

框架包含先进的市场微观结构模型，将订单簿量变化分解为：

1. **成交（Trades）**：市场订单消耗的量
2. **新增挂单（Adds）**：加入订单簿的限价单
3. **撤单（Cancellations）**：
   - 在我们之前的撤单：使我们的队列位置前进 ✅
   - 在我们之后的撤单：对我们的位置无影响

#### 可配置超参数

```python
config = BacktestConfig(
    # ... 其他参数 ...

    # 微观结构模型参数
    microstructure_trade_probability=0.7,  # P(减少=成交 vs 撤单)
    microstructure_add_probability=0.6,    # P(增加=新单 vs 替换)
    microstructure_cancel_alpha=2.0,       # Beta分布α（前方权重）
    microstructure_cancel_beta=3.0,        # Beta分布β（后方权重）
    microstructure_trade_aggression=0.8,   # 成交侵略性水平
)
```

详细说明请参阅[微观结构模型文档](docs/microstructure_model.md)。

### 策略开发

通过继承 `BaseStrategy` 创建自定义策略：

```python
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.actions import SubmitOrder, OrderAction
from typing import List

class MyStrategy(BaseStrategy):
    def __init__(self, param1: float, param2: int):
        self.param1 = param1
        self.param2 = param2

    def on_snapshot(self, context: StrategyContext) -> List[OrderAction]:
        """每个快照时调用"""
        # 访问市场数据
        mid = context.get_mid_price()
        spread = context.get_spread()

        # 检查持仓
        position = context.position_quantity

        # 生成订单
        if not context.has_active_orders():
            return [
                SubmitOrder(side=1, price=mid-0.5, quantity=1.0),
                SubmitOrder(side=-1, price=mid+0.5, quantity=1.0),
            ]
        return []

    def on_fill(self, order_id: int, price: float, quantity: float) -> None:
        """订单成交时调用"""
        pass
```

### 性能

**基准测试**（1000个快照）：
- 吞吐量：2,000-6,000 snapshots/秒
- 内存：~100MB
- 延迟：<500μs 每个快照

**优化技术：**
- 热点路径使用Numba JIT编译
- NumPy向量化操作
- 高效的订单簿状态管理

### 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试套件
pytest tests/unit/            # 单元测试
pytest tests/integration/     # 集成测试
pytest tests/performance/     # 性能基准

# 运行覆盖率测试
pytest --cov=src tests/
```

### 配置

`BacktestConfig` 完整配置选项：

```python
@dataclass
class BacktestConfig:
    # 数据
    data_path: str                    # 快照数据路径
    symbol: str                       # 交易代码

    # 策略
    strategy_class: Type              # 策略类
    strategy_params: Dict[str, Any]   # 策略参数

    # 风险限制
    max_position: float = 10.0        # 最大持仓
    max_loss: Optional[float] = None  # 止损阈值

    # 交易成本
    maker_fee_rate: float = 0.0001    # Maker费率（1bp）
    taker_fee_rate: float = 0.0002    # Taker费率（2bp）
    contract_multiplier: float = 1.0  # 合约乘数

    # 微观结构模型
    microstructure_trade_probability: float = 0.7
    microstructure_add_probability: float = 0.6
    microstructure_cancel_alpha: float = 2.0
    microstructure_cancel_beta: float = 3.0
    microstructure_trade_aggression: float = 0.8

    # 执行
    random_seed: int = 42             # 可重现性
    log_level: str = "INFO"           # 日志级别
```

### 文档

- [数据格式规范](docs/data_format.md) - 必需的数据字段和格式 ⭐
- [微观结构模型](docs/microstructure_model.md) - 详细模型文档
- [功能规格](specs/001-maker-snapshot-500ms/spec.md) - 需求文档
- [实现计划](specs/001-maker-snapshot-500ms/plan.md) - 设计文档
- [任务分解](specs/001-maker-snapshot-500ms/tasks.md) - 开发任务

### 许可证

MIT License