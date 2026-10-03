代码需求：

在 statistics 文件夹中创建一个 Python 文件，使用 matplotlib 绘制一张宽扁的柱状图，展示 8 组配置的**最大 buffer size** 对比。具体要求如下：


数据生成逻辑:
- 每组按顺序读取以下 12 个配置文件并运行 runner_one_layer.py（每个配置运行 10 轮，取**每轮最大 buffer size** 的**10 轮平均**作为该配置的值）：
    - EP_16_2_mul_2.json
    - Hydra_16_2_mul_2.json
    - FSE-DP_16_2_mul_2.json
    - FSE-DP-paired_16_2_mul_2.json
    - EP_64_2_mul_2.json
    - Hydra_64_2_mul_2.json
    - FSE-DP_64_2_mul_2.json
    - FSE-DP-paired_64_2_mul_2.json
    - EP_256_2_mul_2.json
    - Hydra_256_2_mul_2.json
    - FSE-DP_256_2_mul_2.json
    - FSE-DP-paired_256_2_mul_2.json
- 不同组通过 runner_one_layer.py 的参数指定 model 和 dataset，例如：
    ```
    python runner_one_layer.py --model DeepSeek-MoE --dataset C4 --config configs/EP_16_2_mul_2.json
    ```
- 每次运行一个配置：
    - 运行 10 轮，计算每轮在所有芯片上出现的**最大 buffer size**（单位：MB），取 10 轮平均作为该配置的值。
    - 运行结束后即时打印调试信息：
        <MODEL>-<DATASET>  STEADY 平均最大 buffer size: xx.xx MB ,  FINISH 帧: xxx
    - 将该配置结果保存到内存数据结构（例如字典或 DataFrame），便于后续绘图与持久化（可保存到 csv/json）。


绘图： 
- 图为单个子图（横向分组柱状图），包含 8 组（(Phi3.5-MoE, Yuan2.0, DeepSeek-MoE, Qwen3-MoE) × (C4, wikitext-2)）。
- 每组表示一个 (model, dataset) 对，每组内有 4 个策略（EP, Hydra, FSE-DP without paired-load, FSE-DP with paired-load），每个策略包含 3 个输入长度（16, 64, 256），即每组共 12 根柱子，8 组共 96 根柱子。
- 组间有明显间距，组内柱子聚簇排列，整体图片较宽且扁平（可通过 figsize 调整，如 figsize=(20,4) 或更宽）。
- 最终每组应得到形如 shape=(12,) 的**平均最大 buffer size** 序列（顺序与上面的 12 个配置文件一致）。
- 将 8 组数据合并为 shape=(8,12) 的矩阵用于绘图。
- 可选保存格式：
    - statistics/max_buffer_MB.npy 或 statistics/max_buffer_MB.csv
- 使用 matplotlib 绘制分组柱状图：
    - x 轴按 8 组分段，每段内 12 根柱子（或按策略分子分组再按 input-length 细分，保证可读性）。
    - y 轴为“**平均最大 buffer size（MB）**”。
    - 设置合理的颜色方案以区分策略和输入长度（可使用 colormap 或手动配色）。
    - 添加图例，标明策略与/或输入长度（若图例过多，可只标策略并用不同纹理区分长度，或在图注说明顺序）。
    - 在每组之间增加横向空白以增强可读性。
    - 设置大尺寸、较高 dpi（例如 dpi=300），保存路径： statistics/max_buffer_barplot.png。
    - 添加标题、x 轴组标签（如 "Phi3.5-MoE / C4", "Phi3.5-MoE / wikitext-2", ...），y 轴标签及网格。

代码修改：
- 检查代码是否能满足上述需求，即现有代码是否记录了**每个芯片的 buffer size**。如果没有，需要修改代码，添加记录**每轮最大 buffer size** 的代码。
