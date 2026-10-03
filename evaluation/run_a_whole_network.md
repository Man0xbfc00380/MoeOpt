参考 runner_one_layer.py，创建 runner_one_network.py，完整运行神经网络多次前向过程，以测试包含 token buffering 在内的整体性能。

我们是否可以仍用静态的方法实现这个测试，即先跑网络得出结果，再运行统计程序？

是可以的。我们可以连续跑多次固定 token 数量的前向过程，然后在每次前向过程中记录被 buffer 掉的 token，并将这些被 buffer 掉的 token 累加到下一次前向过程中。

我们把上述流程分为两个阶段：

1. 网络运行阶段：针对每个网络在每个数据集上，多次随机选取数据集上一定长度的输入，进行前向过程，并保留每次前向过程的 expert trace。
2. 统计阶段：运行统计程序，输入 expert trace，衡量每次前向过程中被 buffer 掉的 token 数量，并输出统计结果。

网络运行阶段较为简单，只需要在现有程序上修改一个在数据集上随机选取一定长度输入的功能即可。

统计阶段则较为复杂，可能无法直接在当前的 runner_one_layer.py 程序上修改。

- 首先要增加多次forward, 以及每次forward的每一层的循环，不同的forward-layer取查找对应的trace json文件。
- 针对每个trace json文件，统计每个expert被激活的token个数，筛选出需要被buffer掉的token。暂存这些token的expert激活情况。（这里有个潜在问题，我们如果仿真一个request对于token buffering的slackness?）

