在中创建一个 Python 文件，使用 matplotlib 绘制一张宽扁的柱状图，，展示5种不同的策略在四种网络模型（Phi3.5-MoE, Yuan2.0, DeepSeek-MoE, Qwen3-MoE）和两种输入长度(64，256)上的的utilization及throughtput。所以每个柱状图分为8组，每组5个柱子。数据集为C4。

我们将5种策略成为A1-A5。针对不同长度的输入与不同的策略，运行configs/ablation_config中的如下的不同的配置文件:
- A1_64_2_mul_2.json
- A2_64_2_mul_2.json
- A3_64_2_mul_2.json
- A4_64_2_mul_2.json
- A5_64_2_mul_2.json
- A1_256_2_mul_2.json
- A2_256_2_mul_2.json
- A3_256_2_mul_2.json
- A4_256_2_mul_2.json
- A5_256_2_mul_2.json

其余的实现方式均与完全相同（包括断点续跑cache等，但要采用一个不同cache文件夹）。A5策略的buffering slackness为10%，其余策略的buffering slackness为0%（即没有不采取token buffering）。
