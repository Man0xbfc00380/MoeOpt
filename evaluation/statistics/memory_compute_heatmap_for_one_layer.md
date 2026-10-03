代码需求：

在 statistics 文件夹中创建 Python 文件，使用 matplotlib 绘制一张大图，内含两个子图：  
- 子图1：Buffer Utilization 热力图  
- 子图2：Computation Utilization 热力图  

每个热力图 4 行，每行对应一块芯片（Chip-0 ~ Chip-3）的利用率随时间（Frame）变化。  
颜色越“热”表示利用率越高。横坐标为时间帧（Frame），纵坐标为芯片编号。  

数据生成逻辑：  
1. 只跑 1 组配置：DeepSeek-MoE × C4 (其他配置可从(Phi3.5-MoE, Yuan2.0, DeepSeek-MoE, Qwen3-MoE) × (C4, wikitext-2)中选择)。
2. 这1组固定取输入长度 64、并行策略 EP 的配置文件：  
   configs/FSE-DP_64_2_mul_2.json  
3. 对每组配置调用  
   python runner_one_layer.py --model DeepSeek-MoE --dataset C4 --config configs/FSE-DP_64_2_mul_2.json  
   运行 1 轮即可。  
4. 运行结束后，从输出日志或生成的.csv 中提取：  
   - 每帧每芯片的 buffer 利用率  
   - 每帧每芯片的 computation 利用率  
   形成两个 4×T 的矩阵（T 为总帧数）。  
5. 将结果缓存到字典，键为 (model,dataset)，值为 dict{'buffer':np.array(4,T), 'compute':np.array(4,T)}。  

绘图：  
- 大图 2×1 布局，上侧子图 title="Computation Utilization Heatmap"，下侧子图 title="Buffer Utilization Heatmap"。 图片大小为论文的半栏。
- 每个子图内4行，表示4个芯片分别的利用率情况，行标签为 “Chip 0” 等形式。  
- 横坐标为Frame，颜色映射用 'hot'，右侧加 colorbar。  
- 图片保存为：statistics/buffer_compute_heatmap.png），dpi尽量高。  
- 其余绘图风格可参考

代码修改：
- 检查代码是否能满足上述需求，即现有代码是否记录了每个芯片的buffer utilization和computation utilization。如果没有，需要修改代码，添加记录每个芯片利用率的代码。

调试输出：  
每跑完一组配置，立即打印：  
<MODEL>-<DATASET>  STEADY 平均芯片利用率: xx.xx % ,  FINISH 帧: xxx
