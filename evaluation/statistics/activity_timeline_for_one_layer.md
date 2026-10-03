代码需求：

在 statistics 文件夹中创建 Python 文件，使用 matplotlib 绘制一张大图，内含 1 个子图：  
- 子图：4 芯片 activity timeline  

横坐标为时间（Frame），纵坐标为 4 个芯片（Chip-0 ~ Chip-3）。  
每个芯片在同一子图内绘制 4 条断续线，分别表示：  
  1. 芯片间 receive  
  2. ddr_load  
  3. 计算（compute）  
  4. 芯片间 send  

线段连续时表示该操作正在进行，断开时表示空闲。  

数据生成逻辑：  
1. 只跑 1 组配置：Phi3.5-MoE × C4（其他配置可从(Phi3.5-MoE, Yuan2.0, DeepSeek-MoE, Qwen3-MoE) × (C4, wikitext-2)中选择）。  
2. 这 1 组固定取输入长度 16、并行策略 EP 的配置文件：  
   configs/FSE-DP_16_2_mul_2.json  
3. 对每组配置调用  
   python runner_one_layer.py --model Phi3.5-MoE --dataset C4 --config configs/FSE-DP_16_2_mul_2.json  
   运行 1 轮即可。  
4. 运行结束后，从输出日志或生成的 .csv 中提取：  
   - 每帧每芯片的 4 类操作活跃标志（0/1 或起止时间片段），形成 4×T×4 的张量（chip, frame, activity_type）。  
5. 将结果缓存到字典，键为 (model,dataset)，值为 dict{'timeline':np.array(4,T,4)}。  

绘图：  
- 纵轴 4 大刻度对应 Chip-0~Chip-3；每大刻度内再细分 4 条水平线，分别代表 receive、ddr_load、compute、send。  
- 横轴为 Frame，范围 0~T-1。  
- 每条线用不同颜色区分操作类型；线段出现即表示该帧活跃。  
- 图例给出 4 类操作标签。  
- 图片保存为：statistics/activity_timeline.png，dpi 尽量高。  
- 其余绘图风格可参考论文半栏宽度。  

代码修改：  
- 检查现有代码是否已逐帧记录每芯片的 receive、ddr_load、compute、send 活跃状态；若无，需补充记录逻辑。  
