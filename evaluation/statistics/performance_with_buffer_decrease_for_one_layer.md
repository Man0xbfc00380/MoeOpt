代码需求：

在 statistics 文件夹中创建 Python 文件，使用 matplotlib 绘制一张大图，内含 1 个子图：  
- 子图：4 种并行策略（EP、Hydra、FSE-DP (no paired-load)、FSE-DP (paired-load)）的“完成帧数 vs. chip buffer size”折线  

横坐标为 chip buffer size（从 64 逐级降到 8，左侧起点 64）。  
纵坐标为对应 buffer size 下跑完一轮所需的 frame 数。  

数据生成逻辑：  
1. 固定模型与数据集：Phi3.5-MoE × C4（其余组合可后续扩展）。  
2. 四种策略各取 1 份代表配置（输入长度 16）：  
   - EP: configs/EP_16_2_mul_2.json  
   - Hydra: configs/Hydra_16_2_mul_2.json  
   - FSE-DP (no paired-load): configs/FSE-DP_16_2_mul_2.json  
   - FSE-DP (paired-load): configs/FSE-DP-paired_16_2_mul_2.json  
3. 对每种策略，依次将芯片上的buffer size [64到8] 57个值（注意chip buffer size由HardwareConfig.token_buffer_size + HardwareConfig.expert_buffer_size, 我们固定HardwareConfig.token_buffer_size为5，而修改HardwareConfig.expert_buffer_size，即从59到3），每改一次执行：  
   python runner_one_layer.py --model Phi3.5-MoE --dataset C4 --config <对应config>  
   运行 1 轮即可。  
4. 运行结束后，从日志或生成的 .csv 中提取该次实验的总帧数 T，形成 4×57 的张量（strategy, buffer_size）→ frame。   

绘图：  
- 横轴：chip buffer size，刻度 64→8，从左到右降序。  
- 纵轴：完成帧数 frame，线性刻度。  
- 四条折线分别对应四种策略，用不同颜色/标记区分；图例给出策略标签。  
- 图片保存为：statistics/performance_vs_buffer.png，dpi 400。  
- 其余绘图风格参考论文半栏宽度，字体不小于 8 pt。  

代码修改：  
- 检查 runner_one_layer.py 是否已输出“总帧数”或“最后一帧编号”；若无，需补充记录逻辑。
