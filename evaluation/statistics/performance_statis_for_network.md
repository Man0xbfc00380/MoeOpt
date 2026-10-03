在文件夹statistics中创建Python文件，通过matplotlib绘制柱状图，共8组。每组为(Phi3.5-MoE, Yuan2.0, DeepSeek-MoE, Qwen3-MoE) × (C4, wikitext-2) 中的一个配置。8组画在同一张柱状图上，组间留适当间距，整体图较宽扁，纵坐标为整体平均芯片利用率。

每组包含5个序列：EP、Hydra、FSE-DP with paired、FSE-DP with paired and buffer 10%、FSE-DP with paired and buffer 20%、FSE-DP with paired and buffer 30%。  
仅评估输入长度256的情况，每组共5根柱子，8组总计40根柱子。

每组依次读取如下5个config文件（仅256长度）：

- EP_256_2_mul_2.json  
- Hydra_256_2_mul_2.json  
- FSE-DP-paired_256_2_mul_2.json  
- FSE-DP-paired-buffer10_256_2_mul_2.json  
- FSE-DP-paired-buffer20_256_2_mul_2.json  
- FSE-DP-paired-buffer30_256_2_mul_2.json  

即每组调用runner_one_layer.py 5次，每个配置运行100轮，取100轮平均利用率。  
运行单层：固定仿真第6层（layer=6）。

若运行wikitext数据集第4次实验，则到wikitext_256_100/run_4/layer_6_token_experts.json读取对应的expert-token绑定关系。

不同组通过命令行参数指定，例如DeepSeek-MoE在C4数据集上运行FSE-DP-paired-buffer20：

python runner_one_layer.py --model DeepSeek-MoE --dataset C4 --config configs/FSE-DP-paired-buffer20_256_2_mul_2.json

buffer 20%指的是token buffering策略已20%的slackness执行。什么是token buffering策略：

For experts that are activated by only a few tokens, loading them onto the chip causes significantly inefficient bandwidth utilization due to low data-reuse rates. Therefore, instead of processing these tokens in the current iteration, we buffer the tokens that activate them and combine these buffered tokens with new incoming tokens in the next forward pass to re-evaluate expert-activation patterns.

20%的slackness，就是指在这轮（run i）执行中，需要计算的token数量的排名中末尾20%的expert对应的token，会被buffering，相应的expert不会被立即执行（从当前的expert-token绑定关系中去除）。而本Python文件，会记录哪些token被buffering掉了，然后再run i+1时，把buffering掉的token和相应的expert再加入到新的expert-token绑定关系中。以此类推，直到run 100次。最后一次，所有的token都需要被计算。每一个轮,统计执行阶段的时间，然后一个柱子的数据，是100轮的时间综合。

值得注意的是，因为程序运行时间会很长，所以每一轮输出的结果，都应该暂存在一个json文件中。即便画图程序中断，也可以在画图程序重新运行时，跳过json文件中已存在的数据的仿真。


