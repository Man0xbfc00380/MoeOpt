代码需求：

再文件夹statistics中创建python文件，通过matplotlib绘制柱状图, 包含8组。每组为(Phi3.5-MoE, Yuan2.0, DeepSeek-MoE, Qwen3-MoE) × (C4, wikitext-2) 中的一个配置。8组都画在一个柱状图上，只不过组织间有一定的间距，如给定的图的实例所示（该图分为了两组），所以整体图片比较宽、比较扁。纵坐标为整体平均利用率。

每一组又包含了4个序列，分别表示采用EP, Hydra, FSE-DP without paired-load, FSE-DP with paired load这四种策略的整体平均利用率。每个序列evaluate 3种不同输入长度（16, 64, 256）的情况。所以一组总共有12根柱子。总共8组有8*12=96根柱子。

每一组依次读取12个configs/config文件，表示运行的配置：

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

即每一组要调用runner_one_layer.py 12次，运行12个配置，每个配置运行10轮，取平均利用率。

而不同组(Phi3.5-MoE, Yuan2.0, DeepSeek-MoE, Qwen3-MoE) × (C4, wikitext-2) 则采用配置runner_one_layer.py的参数的方式指定。

例如，DeepSeek-MoE在C4数据集上的运行配置为：

```
python runner_one_layer.py --model DeepSeek-MoE --dataset C4 --config configs/EP_16_2_mul_2.json
```
除了生成最终集合了所有数据的柱状图外，没跑完一个配置后，输出该配置的STEADY阶段整体平均芯片利用率，和FINISH发生帧，以便及时调试。

代码需求：

在文件夹statistics中创建python文件，通过matplotlib绘制折线图, 包含4组。每组为(DeepSeek-MoE, Qwen3-MoE) × (C4, wikitext-2) 中的一个配置。4组都画在一个折线图上，每组包含4种不同在输入长度为64的情况下采用EP, Hydra, FSE-DP without paired-load, FSE-DP with paired load这四种策略的利用率曲线。参考plot_utilizations函数的实现，曲线要经过512窗口平滑处理。同一策略的曲线要采用同一色系，只是在不同组中的曲线的深浅不一致。使用runner_one_layer.py 实现上述功能，必要时可以runner_one_layer.py。

每一组依次读取4个configs/下的文件，表示运行的配置：
- EP_64_2_mul_2.json
- Hydra_64_2_mul_2.json
- FSE-DP_64_2_mul_2.json
- FSE-DP-paired_64_2_mul_2.json
其他绘图风格可参考
与其不同的是，该图为论文半栏的图，注意绘图大小