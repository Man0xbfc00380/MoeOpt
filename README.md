# 芯片执行任务的动态调度动画

该项目实现了芯片执行任务的动态调度动画，包括调度算法和动画展示两部分。动画展示了4个芯片和4个DDR之间数据块的移动和计算过程。

## 项目结构

- `main.py`: 主程序入口
- `scheduler.py`: 调度算法接口和实现
- `animation.py`: 动画展示模块
- `data_block.py`: 数据块类定义
- `README.md`: 项目说明文档

## 功能特点

1. **调度算法模块**：
   - 提供了调度算法的接口，可以自定义不同的调度策略
   - 内置了四种调度算法：
     - 简单顺序调度（SimpleScheduler）：按顺序将数据块从DDR移动到芯片，计算后释放
     - 随机调度（RandomScheduler）：随机选择数据块和动作
     - 优先级调度（PriorityScheduler）：根据数据块的优先级进行调度
     - 轮询调度（RoundRobinScheduler）：轮流使用每个芯片
   - 调度算法控制数据块在DDR与芯片之间的移动和计算

2. **动画展示模块**：
   - 使用matplotlib实现动画效果
   - 可视化4个芯片和4个DDR的布局
   - 展示数据块在DDR与芯片之间的移动
   - 展示芯片对数据块的计算过程
   - 支持导出动画为视频文件

## 使用方法

### 安装依赖

```bash
pip install numpy matplotlib
```

### 运行程序

```bash
python main.py
```

### 命令行参数

- `--scheduler`: 选择调度算法，可选值为：
  - `simple`：简单顺序调度
  - `random`：随机调度
  - `priority`：优先级调度
  - `round_robin`：轮询调度
  默认为`simple`
- `--duration`: 动画持续时间（秒），默认为20秒
- `--fps`: 动画帧率，默认为30帧/秒
- `--save`: 保存动画为视频文件，如果不指定则显示动画

### 示例

```bash
# 使用简单调度算法，显示动画
python main.py --scheduler simple

# 使用随机调度算法，显示动画
python main.py --scheduler random

# 使用简单调度算法，保存为视频文件
python main.py --scheduler simple --save animation.mp4

# 使用随机调度算法，设置动画时长和帧率，保存为视频文件
python main.py --scheduler random --duration 30 --fps 60 --save animation.mp4

# 使用优先级调度算法，保存为GIF文件
python main.py --scheduler priority --duration 15 --save animation.gif

# 使用轮询调度算法
python main.py --scheduler round_robin --duration 25
```

## 自定义调度算法

项目提供了四种内置调度算法：

1. **SimpleScheduler**：简单顺序调度器，按顺序将数据块从DDR移动到芯片，计算后释放
2. **RandomScheduler**：随机调度器，随机选择数据块和动作
3. **PriorityScheduler**：优先级调度器，根据数据块的优先级进行调度
4. **RoundRobinScheduler**：轮询调度器，轮流使用每个芯片

要自定义调度算法，需要继承`scheduler.py`中的`Scheduler`基类，并实现`get_next_action()`方法。然后在`main.py`中添加对应的调度算法选项。

```python
from scheduler import Scheduler, Action, ActionType

class MyCustomScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        # 初始化自定义调度器
    
    def get_next_action(self):
        # 实现自定义调度逻辑
        # 返回一个Action对象
        return Action(...)
```

可以参考`custom_scheduler.py`中的示例实现，了解如何创建自定义调度算法。

## 注意事项

- 导出视频需要安装FFmpeg，请确保系统中已安装FFmpeg并添加到环境变量中
- 动画使用Arial字体，请确保系统中已安装该字体



# 整理思路

下面有如下任务可以做：

1. task emulator可以预先做好DDR的分配，而把指针管理交给scheduler
2. 开始开发基于几条基本规则的调度
3. 增加最初始的token dispatcher
3. 思考在交错流水的过程中，redispatch的实现
