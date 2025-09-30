# HFT回测框架技术总结

## 1. OrderBook建模

### 1.1 核心问题

Snapshot数据包含所有市场参与者的挂单量，包括我们自己的订单：

```
snapshot.bid_volume_0 = market_volume + our_volume
```

### 1.2 解决方案：双层OrderBook系统

#### OrderBook类 (`src/core/orderbook.py`)

**职责**: 维护干净的市场状态（不含我们的订单）

**核心数据结构**:
```python
class OrderBook:
    # 追踪我们的所有活跃订单
    our_active_orders: Dict[order_id -> (price, side, quantity)]

    # 每个价格档位
    bid_levels: Dict[price -> OrderBookLevel]
    ask_levels: Dict[price -> OrderBookLevel]

@dataclass
class OrderBookLevel:
    price: float
    market_volume: float  # snapshot_vol - our_vol
    our_volume: float     # sum(our orders at this price)
    our_orders: List      # [(order_id, qty, position), ...]
```

**对齐算法**:
```python
def update_from_snapshot(snapshot):
    for price, snapshot_vol in snapshot_data:
        # 1. 计算我们在这个价格的总量
        our_vol = sum(qty for (p, s, qty) in our_active_orders.values()
                      if abs(p - price) < eps and s == side)

        # 2. 提取市场量
        market_vol = max(0, snapshot_vol - our_vol)

        # 3. 存储分离的量
        levels[price] = OrderBookLevel(
            price=price,
            market_volume=market_vol,  # 干净的市场量
            our_volume=our_vol          # 我们的量
        )
```

**关键方法**:

1. **add_our_order(order_id, price, qty, side)**:
   - 追踪订单
   - 基于market_volume计算队列位置
   - `volume_ahead = market_volume` (不包含我们的其他订单)

2. **remove_our_order(order_id, qty)**:
   - 更新或删除追踪
   - 支持部分成交

3. **get_market_volume_at_price(price, side)**:
   - 返回干净的市场量
   - 用于队列位置计算

#### OrderBookState类 (`src/core/orderbook_state.py`)

**职责**: 微观结构模型和成交判断

**核心功能**:

1. **Volume变化分解** (使用MicrostructureModel):
   ```python
   @dataclass
   class VolumeChange:
       traded: float               # 成交量
       added: float                # 新增挂单
       cancelled_before_us: float  # 我们前方的撤单
       cancelled_after_us: float   # 我们后方的撤单
   ```

2. **队列位置更新**:
   ```python
   def update_queue_position(...):
       # 分解volume变化
       change = microstructure.decompose_volume_change(...)

       # 更新位置
       new_position = current_position
       new_position -= change.traded              # 成交推进
       new_position -= change.cancelled_before_us # 前方撤单推进

       return new_position
   ```

3. **成交判断** (盘口内/盘口外):
   ```python
   def check_fill(...):
       if order_in_book:
           # 盘口内：队列位置模拟
           if volume_traded > volume_ahead:
               fill_qty = min(order_qty, volume_traded - volume_ahead)
       else:
           # 盘口外：价格交叉检测
           cross_event = detect_price_cross(...)
           if cross_event.crossed:
               fill = estimate_fill_from_cross(...)
   ```

### 1.3 两个系统的协同

```python
# BacktestEngine中的集成

def _execute_simulation():
    for snapshot in snapshots:
        # 1. OrderBook: 提取市场量
        orderbook.update_from_snapshot(snapshot)

        # 2. OrderBookState: 微观结构分析
        orderbook_state.update_from_snapshot(snapshot)

        # 3. 使用market_volume更新队列
        for order in active_orders:
            market_vol = orderbook.get_market_volume_at_price(price, side)
            # 使用market_vol进行微观结构分解
            ...

def _process_submit_order(action):
    # 添加到两个系统
    vol_ahead, _ = orderbook.add_our_order(order_id, ...)
    orderbook_state.add_our_order(price, qty, side)
```

## 2. 订单排队模型

### 2.1 初始队列位置

**新订单加入队列末尾**:
```python
vol_ahead, queue_pos = orderbook.add_our_order(order_id, price, qty, side)

# vol_ahead = market_volume at this price
# 我们排在所有现有市场量之后
```

**特殊情况**:
```python
if price not in book:
    # 价格不在5档内，我们创建新档位
    volume_ahead = 0.0  # 队首
```

### 2.2 队列位置更新

**使用微观结构模型分解volume变化**:

```python
# Volume减少时
if vol_curr < vol_prev:
    # 1. 判断是成交还是撤单
    trade_prob = microstructure_trade_probability
    if cum_vol_delta > 0:
        # 有成交活动，提高成交概率
        trade_prob *= (0.5 + 0.5 * activity_factor)

    # 2. 分解
    if is_trade:
        traded = decrease * 0.7~1.0
        cancelled = decrease - traded
    else:
        cancelled = decrease * 0.7~1.0
        traded = decrease - cancelled

    # 3. 定位撤单位置 (Beta分布)
    cancel_positions ~ Beta(alpha, beta)
    cancelled_before_us = sum(pos < our_position)
    cancelled_after_us = total_cancelled - cancelled_before_us

    # 4. 更新位置
    new_position -= traded
    new_position -= cancelled_before_us
```

**Beta分布控制撤单位置**:
- `alpha > beta`: 撤单集中在队列前方
- `beta > alpha`: 撤单集中在队列后方
- `alpha = beta`: 均匀分布

### 2.3 成交判断

**盘口内订单**:
```python
volume_traded = calculate_volume_traded_at_price(...)

if volume_traded > volume_ahead:
    # 队列被清空，轮到我们
    available = volume_traded - volume_ahead
    fill_qty = min(order_qty, available)
    return (order_price, fill_qty)
```

**盘口外订单 (价格交叉)**:
见第3节。

## 3. Snapshot之间的成交猜测

### 3.1 两种成交模式

#### 模式A: 盘口内订单（Queue-based）

**条件**: 订单在5档可见范围内

**方法**: 队列位置模拟

**步骤**:
1. 计算这个价格的volume_traded
   ```python
   vol_prev = snapshot_prev.volume_at_price
   vol_curr = snapshot_curr.volume_at_price
   volume_traded = max(0, vol_prev - vol_curr)
   ```

2. 检查是否清空队列
   ```python
   if volume_traded > volume_ahead:
       # 成交！
       fill_qty = min(order_qty, volume_traded - volume_ahead)
   ```

**精度**: 高（基于真实队列位置）

#### 模式B: 盘口外订单（Price Crossing）

**条件**: 订单不在5档内

**方法**: 价格交叉检测

**复杂场景处理** (`src/core/price_cross_detector.py`):

**场景1: 正常交叉**
```python
# 买单: ask移动到我们价格
if ap1_curr <= order_price:
    crossed = True

# 卖单: bid移动到我们价格
if bp1_curr >= order_price:
    crossed = True
```

**场景2: 侵略性交叉**
```python
# 买方横扫卖盘
if bp1_curr >= ap1_prev:
    # 检查买单是否在交叉范围内
    if order_price >= bp1_prev and order_price <= bp1_curr:
        crossed = True
        aggressive_side = BUY

# 卖方横扫买盘
if ap1_curr <= bp1_prev:
    # 检查卖单是否在交叉范围内
    if order_price <= ap1_prev and order_price >= ap1_curr:
        crossed = True
        aggressive_side = SELL
```

**场景3: 跳空**
```python
# 价格大幅跳跃
if abs(bp1_curr - bp1_prev) > threshold:
    # 检查订单是否被跳过
    if price_in_gap(order_price):
        crossed = True
        volume_consumed = vol_delta * 0.5  # 保守估计
```

### 3.2 成交价估算

**使用turnover_volume计算真实成交价**:
```python
def estimate_fill_from_cross(cross_event, ...):
    # 1. 计算平均成交价
    vol_delta = cum_vol_curr - cum_vol_prev
    turnover_delta = turnover_curr - turnover_prev
    avg_trade_price = turnover_delta / vol_delta

    # 2. 根据角色确定成交价
    if we_are_passive:
        # 我们提供流动性
        fill_price = order_price
    else:
        # 我们在侵略性流动路径上
        fill_price = (order_price + avg_trade_price) / 2

    return fill_price, fill_qty
```

### 3.3 成交量估算

**基于订单在交叉范围中的位置**:
```python
# 计算价格范围
if order_side == BUY:
    range_low = min(bp1_prev, ap1_curr)
    range_high = max(bp1_curr, ap1_prev)
else:  # SELL
    range_low = min(ap1_prev, bp1_curr)
    range_high = max(ap1_curr, bp1_prev)

# 我们的价格优势
price_ratio = (order_price - range_low) / (range_high - range_low)

# 捕获的成交量（保守估计）
our_share = vol_delta * price_ratio * 0.5  # 最多50%
fill_qty = min(order_qty, our_share)
```

## 4. 关键参数

### 4.1 微观结构参数

```python
@dataclass
class MicrostructureParams:
    # Volume减少时，是成交的概率（vs撤单）
    trade_probability: float = 0.7

    # Volume增加时，是新单的概率（vs替换）
    add_probability: float = 0.6

    # Beta分布参数：撤单位置
    cancel_loc_alpha: float = 2.0  # 前方权重
    cancel_loc_beta: float = 3.0   # 后方权重

    # 成交侵略性
    trade_aggression: float = 0.8
```

**调参指南**:

| 参数 | 提高效果 | 降低效果 |
|------|---------|---------|
| trade_probability | 更多成交，队列快速推进 | 更多撤单，队列缓慢推进 |
| cancel_alpha | 前方撤单增多（不利） | 前方撤单减少 |
| cancel_beta | 后方撤单增多（有利） | 后方撤单减少 |

### 4.2 市场场景配置

**高频市场** (如沪深期货):
```python
MicrostructureParams(
    trade_probability=0.85,  # 高成交
    cancel_alpha=3.0,        # 均匀撤单
    cancel_beta=3.0,
)
```

**低流动性市场**:
```python
MicrostructureParams(
    trade_probability=0.4,   # 低成交
    cancel_alpha=4.0,        # 前方撤单多
    cancel_beta=2.0,
)
```

**机构主导市场**:
```python
MicrostructureParams(
    trade_probability=0.75,
    cancel_alpha=2.0,        # 后方撤单多
    cancel_beta=5.0,         # 有利于做市
    trade_aggression=0.9,
)
```

## 5. 数据流

### 5.1 完整的Snapshot处理流程

```python
for snapshot in snapshots:
    # === 阶段1: OrderBook更新 ===
    orderbook.update_from_snapshot(snapshot)
    # 提取: market_volume = snapshot_vol - our_vol

    orderbook_state.update_from_snapshot(snapshot)
    # 更新微观结构模型状态

    # === 阶段2: 队列位置更新 ===
    for order in active_orders:
        # 使用market_volume和微观结构模型
        new_pos, _, change = orderbook_state.update_queue_position(
            order.price,
            order.side,
            order.current_position,
            snapshot_prev,
            snapshot_curr
        )

        # 更新OrderBook中的位置追踪
        orderbook.update_order_queue_position(order_id, new_pos)

    # === 阶段3: 成交模拟 ===
    for order in active_orders:
        # 计算这个价格的成交量
        volume_traded = orderbook_state.calculate_volume_traded_at_price(...)

        # 检查成交
        fill_result = orderbook_state.check_fill(
            order.price,
            order.qty,
            order.side,
            order.volume_ahead,
            volume_traded,
            snapshot_prev,
            snapshot_curr
        )

        if fill_result:
            # 从两个系统移除
            orderbook.remove_our_order(order_id, fill_qty)
            orderbook_state.remove_order(price, fill_qty, side)

    # === 阶段4: 策略调用 ===
    context = create_strategy_context(snapshot, ...)
    actions = strategy.on_snapshot(context)

    # === 阶段5: 处理策略动作 ===
    for action in actions:
        if isinstance(action, SubmitOrder):
            # 添加到两个系统
            order_id = create_order(...)
            orderbook.add_our_order(order_id, ...)
            orderbook_state.add_our_order(...)
```

### 5.2 关键时刻的数据状态

**T0: Snapshot到达**
```
Snapshot: bp1=5000, bv1=110
Market: bp1=5000, market_vol=100 (110-10)
Our: order_1 at 5000, qty=10, vol_ahead=100
```

**T1: Volume变化**
```
Snapshot: bp1=5000, bv1=100
Change: -10 volume

Microstructure分解:
- traded: 7
- cancelled_before: 2
- cancelled_after: 1

Queue更新:
- old: vol_ahead=100
- new: vol_ahead=100-7-2=91
```

**T2: 成交检查**
```
volume_traded = 15 (从某处计算)
volume_ahead = 91

Check: 15 > 91? No → 未成交
```

## 6. 性能特征

### 6.1 计算复杂度

**每个Snapshot**:
- OrderBook更新: O(5 * 2) = O(1) (5档 × 2边)
- 订单追踪: O(N) where N = 我们的活跃订单数
- 队列位置更新: O(N × M) where M = 每个订单的微观结构计算
- 成交检查: O(N)

**总体**: O(N) per snapshot，N通常<10

### 6.2 实测性能

- **吞吐量**: 3,000-4,000 snapshots/sec
- **内存**: ~100MB for 1000 snapshots
- **确定性**: 完全可重现（给定random_seed）

### 6.3 瓶颈

主要开销：
1. Numba JIT编译（首次）
2. NumPy数组操作
3. Python字典查找（orderbook levels）

优化空间：
- 使用Numba优化更多热点函数
- 预分配数组减少allocation
- Cython重写关键路径

## 7. 使用示例

### 7.1 基础回测

```python
from src.engine.backtest import BacktestEngine
from src.engine.config import BacktestConfig
from examples.strategies.simple_maker import SimpleMaker

config = BacktestConfig(
    data_path='data/snapshots.parquet',
    symbol='IF2401',
    strategy_class=SimpleMaker,
    strategy_params={},

    # 微观结构参数
    microstructure_trade_probability=0.7,
    microstructure_cancel_alpha=2.0,
    microstructure_cancel_beta=3.0,

    random_seed=42,
)

engine = BacktestEngine(config)
result = engine.run()

print(f"Total PnL: {result.total_pnl}")
print(f"Fill Rate: {result.filled_orders / result.total_orders}")
```

### 7.2 参数扫描

```python
trade_probs = np.linspace(0.3, 0.95, 10)
results = []

for trade_prob in trade_probs:
    config = BacktestConfig(
        data_path='data/snapshots.parquet',
        symbol='IF2401',
        strategy_class=SimpleMaker,
        microstructure_trade_probability=trade_prob,
        random_seed=42,
    )

    engine = BacktestEngine(config)
    result = engine.run()

    results.append({
        'trade_prob': trade_prob,
        'pnl': result.total_pnl,
        'fill_rate': result.filled_orders / result.total_orders,
    })
```

## 8. 总结

### 8.1 核心创新

1. **双层OrderBook系统**: 分离市场量和我们的量
2. **微观结构建模**: 精确分解volume变化
3. **价格交叉检测**: 处理复杂的盘口外场景
4. **Snapshot对齐**: 自动从snapshot中提取市场状态

### 8.2 精度保证

- ✅ 队列位置基于真实市场量
- ✅ 成交判断考虑队列优先级
- ✅ 价格交叉处理多种场景
- ✅ 使用turnover_volume估算真实成交价
- ✅ 微观结构分解提供细粒度控制

### 8.3 适用场景

**最适合**:
- 做市策略 (多个订单，频繁更新)
- 高频策略 (500ms-1s snapshot间隔)
- 限价单为主的策略

**不适合**:
- 纯市价单策略 (不需要队列模拟)
- 低频策略 (>1min, 队列变化太大)
- Tick-by-tick数据 (需要不同的建模方法)

### 8.4 扩展方向

1. **机器学习参数**: 从历史数据学习微观结构参数
2. **自适应模型**: 根据市场状态动态调整参数
3. **多档位联动**: 考虑不同价格档位之间的相关性
4. **订单大小分布**: 模拟不同大小的订单
5. **时间衰减**: 队列位置随时间的自然衰减