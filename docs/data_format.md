# Data Format Specification

[English](#english) | [中文](#chinese)

---

<a name="english"></a>

## English

### Snapshot Data Format

The backtesting framework requires order book snapshot data with the following structure:

#### Required Fields

**Timestamp & Symbol**
- `timestamp` (int64): Unix timestamp in milliseconds
- `symbol` (string): Contract symbol (max 10 characters), e.g., "IF2401"

**Bid Levels** (5 levels, 0 = best)
- `bid_price_0` to `bid_price_4` (float64): Bid prices in descending order
- `bid_volume_0` to `bid_volume_4` (float64): Bid volumes at each price

**Ask Levels** (5 levels, 0 = best)
- `ask_price_0` to `ask_price_4` (float64): Ask prices in ascending order
- `ask_volume_0` to `ask_volume_4` (float64): Ask volumes at each price

**Market Activity** (REQUIRED)
- `cumulative_volume` (float64): **Total traded volume** since session start (monotonically increasing)
- `turnover_volume` (float64): **Total turnover (volume × price)** since session start (monotonically increasing)
- `last_price` (float64): Last trade price

#### Data Validation Rules

The framework automatically validates:

1. **Timestamps**: Must be monotonically increasing
2. **No Crossed Book**: All bid prices < all ask prices
3. **Price Ordering**:
   - Bids: Descending order (best_bid > bid_1 > ... > bid_4)
   - Asks: Ascending order (best_ask < ask_1 < ... < ask_4)
4. **Non-negative Volumes**: All volumes >= 0
5. **Cumulative Fields**: Both `cumulative_volume` and `turnover_volume` must be monotonically increasing
6. **No NaN/Inf**: All numeric fields must be valid numbers

#### Supported File Formats

1. **Parquet** (recommended)
   - Best performance and compression
   - Preserves data types

2. **CSV**
   - Human-readable
   - Must include header row

3. **HDF5**
   - Efficient for large datasets
   - Random access support

#### Example Data

```python
import pandas as pd
import numpy as np

# Generate sample snapshot
snapshot = {
    'timestamp': 1640000000000,  # Unix timestamp in ms
    'symbol': 'IF2401',

    # Bid side (best to 5th level)
    'bid_price_0': 4999.0,
    'bid_price_1': 4998.5,
    'bid_price_2': 4998.0,
    'bid_price_3': 4997.5,
    'bid_price_4': 4997.0,
    'bid_volume_0': 15.0,
    'bid_volume_1': 12.0,
    'bid_volume_2': 10.0,
    'bid_volume_3': 8.0,
    'bid_volume_4': 5.0,

    # Ask side (best to 5th level)
    'ask_price_0': 5001.0,
    'ask_price_1': 5001.5,
    'ask_price_2': 5002.0,
    'ask_price_3': 5002.5,
    'ask_price_4': 5003.0,
    'ask_volume_0': 15.0,
    'ask_volume_1': 12.0,
    'ask_volume_2': 10.0,
    'ask_volume_3': 8.0,
    'ask_volume_4': 5.0,

    # Market activity (REQUIRED)
    'cumulative_volume': 125000.0,  # Total volume traded
    'turnover_volume': 625000000.0,  # Total value traded (volume * price)
    'last_price': 5000.0,
}

df = pd.DataFrame([snapshot])
df.to_parquet('snapshots.parquet', index=False)
```

#### Loading Data

```python
from src.data.loader import load_snapshots

# Load snapshots (auto-detects format)
snapshots = load_snapshots('snapshots.parquet', symbol='IF2401', validate=True)

# snapshots is a NumPy structured array ready for backtesting
print(f"Loaded {len(snapshots)} snapshots")
```

#### Generating Sample Data

Use the provided script to generate synthetic test data:

```bash
python examples/create_sample_data.py
```

This creates sample data with:
- 1000 snapshots (500ms intervals)
- Random walk price simulation
- Realistic order book structure
- Proper `cumulative_volume` and `turnover_volume` fields

---

<a name="chinese"></a>

## 中文

### Snapshot 数据格式

回测框架需要订单簿快照数据，包含以下结构：

#### 必需字段

**时间戳和代码**
- `timestamp` (int64): Unix时间戳（毫秒）
- `symbol` (字符串): 合约代码（最多10字符），例如 "IF2401"

**买方档位**（5档，0 = 最佳）
- `bid_price_0` 到 `bid_price_4` (float64): 买价，降序排列
- `bid_volume_0` 到 `bid_volume_4` (float64): 每个价格的买量

**卖方档位**（5档，0 = 最佳）
- `ask_price_0` 到 `ask_price_4` (float64): 卖价，升序排列
- `ask_volume_0` 到 `ask_volume_4` (float64): 每个价格的卖量

**市场活动**（必需）
- `cumulative_volume` (float64): **累计成交量**，从开盘至今（单调递增）
- `turnover_volume` (float64): **累计成交额（量×价）**，从开盘至今（单调递增）
- `last_price` (float64): 最新成交价

#### 数据验证规则

框架自动验证：

1. **时间戳**: 必须单调递增
2. **不交叉盘口**: 所有买价 < 所有卖价
3. **价格排序**:
   - 买价: 降序排列（最佳买价 > 第2档 > ... > 第5档）
   - 卖价: 升序排列（最佳卖价 < 第2档 < ... < 第5档）
4. **非负量**: 所有量 >= 0
5. **累计字段**: `cumulative_volume` 和 `turnover_volume` 必须单调递增
6. **无NaN/Inf**: 所有数值字段必须是有效数字

#### 支持的文件格式

1. **Parquet**（推荐）
   - 最佳性能和压缩率
   - 保留数据类型

2. **CSV**
   - 人类可读
   - 必须包含表头行

3. **HDF5**
   - 适合大数据集
   - 支持随机访问

#### 数据示例

```python
import pandas as pd
import numpy as np

# 生成示例快照
snapshot = {
    'timestamp': 1640000000000,  # Unix时间戳（毫秒）
    'symbol': 'IF2401',

    # 买方（最佳到第5档）
    'bid_price_0': 4999.0,
    'bid_price_1': 4998.5,
    'bid_price_2': 4998.0,
    'bid_price_3': 4997.5,
    'bid_price_4': 4997.0,
    'bid_volume_0': 15.0,
    'bid_volume_1': 12.0,
    'bid_volume_2': 10.0,
    'bid_volume_3': 8.0,
    'bid_volume_4': 5.0,

    # 卖方（最佳到第5档）
    'ask_price_0': 5001.0,
    'ask_price_1': 5001.5,
    'ask_price_2': 5002.0,
    'ask_price_3': 5002.5,
    'ask_price_4': 5003.0,
    'ask_volume_0': 15.0,
    'ask_volume_1': 12.0,
    'ask_volume_2': 10.0,
    'ask_volume_3': 8.0,
    'ask_volume_4': 5.0,

    # 市场活动（必需）
    'cumulative_volume': 125000.0,  # 累计成交量
    'turnover_volume': 625000000.0,  # 累计成交额（量×价）
    'last_price': 5000.0,
}

df = pd.DataFrame([snapshot])
df.to_parquet('snapshots.parquet', index=False)
```

#### 加载数据

```python
from src.data.loader import load_snapshots

# 加载快照（自动检测格式）
snapshots = load_snapshots('snapshots.parquet', symbol='IF2401', validate=True)

# snapshots 是 NumPy 结构化数组，可用于回测
print(f"加载了 {len(snapshots)} 个快照")
```

#### 生成示例数据

使用提供的脚本生成合成测试数据：

```bash
python examples/create_sample_data.py
```

生成的数据包含：
- 1000个快照（500ms间隔）
- 随机游走价格模拟
- 真实的订单簿结构
- 正确的 `cumulative_volume` 和 `turnover_volume` 字段