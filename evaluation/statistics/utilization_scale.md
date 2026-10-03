代码需求：

在文件夹statistics中创建python文件，通过matplotlib绘制柱状图。只运行Qwen3-MoE网络在C4数据集上。 包含4组, 每组分别为运行在2×2，2×3，3×3，4×4的芯片阵列上的结果。4组都画在一个柱状图上，只不过组之间有一定的间距,所以整体图片比较宽、比较扁。纵坐标为整体平均利用率。

每一组又包含4个序列，分别表示采用EP, Hydra, FSE-DP without paired-load, FSE-DP with paired load这四种策略的利用率。每个序列evaluate 4种不同输入长度（16, 64, 256, 1024）的情况。所以一组总共有16根柱子。总共8组有4*16=64根柱子。

每一组依次读取12个configs/下的配置文件，表示运行的配置：

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
- EP_1024_2_mul_2.json
- Hydra_1024_2_mul_2.json
- FSE-DP_1024_2_mul_2.json
- FSE-DP-paired_1024_2_mul_2.json

注意2×3，3×3，4×4，要在运行时overide HardwareConfig.row_number_of_chip和HardwareConfig.column_number_of_chip。

即每一组要调用runner_one_layer.py 16次，运行16个配置，每个数据点运行[16, 20, 24, 28]这四层，取平均利用率。

因为需要跑很长时间才能出结果，所以每次跑出一层的结果之后，就存下来一个结果到一个json文件中，防止程序中断需要从头跑。下次运行时跳过已有的结果的仿真。

整体绘图风格与实现方式请参考