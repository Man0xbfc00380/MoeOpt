import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import importlib.util
import sys
import logging
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
from collections import defaultdict
import datetime
import shutil
import random
import glob
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datasets import Dataset

# 配置日志记录
log_file_path = "qwen3_moe_expert_analysis_sweep.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Qwen3-MoE Expert Activation Analysis Sweep')
    parser.add_argument('--dataset_name', type=str, default="dataset/wikitext/wikitext-2-raw-v1",
                      help='Dataset name or path')
    parser.add_argument('--dataset_type', type=str, default="wikitext", choices=["wikitext", "mmlu", "winogrande"],
                      help='Type of dataset to use (wikitext, mmlu, or winogrande)')
    parser.add_argument('--mmlu_subset', type=str, default="all",
                      help='MMLU subset to use (e.g., "all", "stem", "humanities", "social_sciences", "other")')
    parser.add_argument('--max_lengths', type=str, default="16,32,64,128,256",
                      help='Comma-separated list of max_lengths to test')
    parser.add_argument('--num_samples', type=int, default=1,
                      help='Number of samples per experiment')
    parser.add_argument('--num_repeats', type=int, default=3,
                      help='Number of times to repeat each experiment with different random samples')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for processing')
    parser.add_argument('--target_layers', type=str, default=None,
                      help='Comma-separated layer indices to analyze (default: all layers)')
    parser.add_argument('--min_text_length', type=int, default=100,
                      help='Minimum text length for filtering samples')
    parser.add_argument('--output_dir', type=str, default="expert_stats_sweep",
                      help='Output directory for statistics')
    parser.add_argument('--show_mean', action='store_true',
                      help='Show mean values in aggregated plots')
    return parser.parse_args()

# 动态加载本地 modular_qwen3_moe.py
model_code_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(model_code_path)
spec = importlib.util.spec_from_file_location("modular_qwen3_moe", os.path.join(model_code_path, "modular_qwen3_moe.py"))
modular_qwen3_moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modular_qwen3_moe)
Qwen3MoeForCausalLM = modular_qwen3_moe.Qwen3MoeForCausalLM

class ExpertActivationStats:
    def __init__(self, num_layers, num_experts, target_layers=None, pad_token_id=100001):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.pad_token_id = pad_token_id
        self.target_layers = target_layers  # 只用于绘图，不影响统计收集
        self.expert_usage = np.zeros((num_layers, num_experts))
        self.token_count = 0
        self.input_token_count = 0
        self.non_pad_token_count = 0
        self.expert_token_count = np.zeros((num_layers, num_experts))
        self.sequence_expert_usage = defaultdict(lambda: defaultdict(set))
        self.expert_weights = defaultdict(list)
        
    def update(self, layer_activations, sequence_idx, input_ids=None):
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.cpu()
            non_pad_mask = (input_ids != self.pad_token_id)
            self.non_pad_token_count += non_pad_mask.sum().item()
        
        for layer_idx, layer_data in enumerate(layer_activations):
            if not isinstance(layer_data, dict) or 'expert_activations' not in layer_data:
                logger.warning(f"Layer {layer_idx} has no expert activation data")
                continue
                
            expert_data = layer_data['expert_activations']
            if not isinstance(expert_data, dict) or 'selected_experts' not in expert_data or 'routing_weights' not in expert_data:
                logger.warning(f"Layer {layer_idx} has no expert routing info (keys: {list(expert_data.keys()) if isinstance(expert_data, dict) else expert_data})")
                continue

            selected_experts = expert_data['selected_experts']
            routing_weights = expert_data['routing_weights']
            
            if isinstance(selected_experts, torch.Tensor):
                selected_experts = selected_experts.cpu().numpy()
            if isinstance(routing_weights, torch.Tensor):
                routing_weights = routing_weights.to(torch.float32).cpu().numpy()
            
            for batch_idx in range(selected_experts.shape[0]):
                for token_idx in range(selected_experts.shape[1]):
                    expert_indices = selected_experts[batch_idx, token_idx]
                    weights = routing_weights[batch_idx, token_idx]
                    
                    for expert_idx, weight in zip(expert_indices, weights):
                        expert_idx = int(expert_idx)
                        self.expert_usage[layer_idx][expert_idx] += weight
                        self.token_count += 1
                        self.expert_token_count[layer_idx][expert_idx] += 1
                        self.sequence_expert_usage[sequence_idx][layer_idx].add(expert_idx)
                        self.expert_weights[(layer_idx, expert_idx)].append(float(weight))
                        
            if layer_idx == 0:
                self.input_token_count += selected_experts.shape[1]

    def get_statistics(self):
        expert_usage_rate = self.expert_usage / self.non_pad_token_count if self.non_pad_token_count > 0 else None
        expert_token_ratio = self.expert_token_count / self.non_pad_token_count if self.non_pad_token_count > 0 else None
        
        sequence_expert_counts = {
            seq_idx: {layer: len(experts) for layer, experts in layer_experts.items()}
            for seq_idx, layer_experts in self.sequence_expert_usage.items()
        }
        
        expert_weight_stats = {}
        for (layer, expert), weights in self.expert_weights.items():
            expert_weight_stats[f"layer_{layer}_expert_{expert}"] = {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights))
            }
        
        stats = {
            "expert_usage": self.expert_usage.tolist(),
            "token_count": self.token_count,
            "input_token_count": self.input_token_count,
            "non_pad_token_count": self.non_pad_token_count,
            "usage_percentage": (self.expert_usage / self.non_pad_token_count * 100).tolist() if self.non_pad_token_count > 0 else None,
            "expert_token_ratio": expert_token_ratio.tolist() if expert_token_ratio is not None else None,
            "sequence_expert_counts": sequence_expert_counts,
            "expert_weight_stats": expert_weight_stats,
            "expert_token_count": self.expert_token_count.tolist()  # 添加expert_token_count到统计数据中
        }
        return stats

    def save(self, filename):
        """保存统计结果到JSON文件"""
        stats = self.get_statistics()
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

def get_text_field_name(dataset):
    for candidate in ['ctx', 'text', 'input', 'content', 'sentence']:
        if candidate in dataset.column_names:
            return candidate
    return dataset.column_names[0]

def load_and_prepare_dataset(dataset_name, dataset_type, mmlu_subset="all", split="train", num_samples=1, min_text_length=100):
    if dataset_type == "wikitext":
        if dataset_name.endswith(".parquet") or os.path.isdir(dataset_name):
            if os.path.isdir(dataset_name):
                data_files = {split: os.path.join(dataset_name, f"{split}-00000-of-00001.parquet")}
            else:
                data_files = {split: dataset_name}
            dataset = load_dataset("parquet", data_files=data_files, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        text_field = get_text_field_name(dataset)
        dataset = dataset.filter(lambda x: len(x[text_field].strip()) >= min_text_length)
        logger.info(f"After filtering empty/short texts: {len(dataset)} samples remaining")
        
    elif dataset_type == "mmlu":
        # 从本地加载MMLU数据集
        try:
            local_path = os.path.join("dataset/mmlu", mmlu_subset, "data.json")
            if os.path.exists(local_path):
                logger.info(f"从本地加载MMLU数据集: {local_path}")
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dataset = Dataset.from_dict(data)
            else:
                logger.info(f"本地数据集不存在，从Hugging Face下载: {mmlu_subset}")
                dataset = load_dataset("cais/mmlu", mmlu_subset, split="test")
            
            # 将问题和选项组合成完整文本
            def format_mmlu_example(example):
                # 确保所有字段都存在
                question = example.get('question', '')
                options = {
                    'A': example.get('A', ''),
                    'B': example.get('B', ''),
                    'C': example.get('C', ''),
                    'D': example.get('D', '')
                }
                answer = example.get('answer', '')
                
                # 构建完整文本
                text = f"Question: {question}\n"
                for opt, content in options.items():
                    text += f"{opt}) {content}\n"
                text += f"Answer: {answer}"
                
                return {'text': text}
            
            dataset = dataset.map(format_mmlu_example)
            logger.info(f"Loaded MMLU dataset with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Error loading MMLU dataset: {e}")
            raise
    elif dataset_type == "winogrande":
        # 加载Winogrande数据集
        try:
            dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
            def format_winogrande(example):
                sentence = example.get('sentence', '')
                option1 = example.get('option1', '')
                option2 = example.get('option2', '')
                answer = example.get('answer', '')
                try:
                    ans = int(answer)
                except Exception:
                    ans = 1
                option = option1 if ans == 1 else option2
                # 替换 _ 或直接拼接
                if '_' in sentence:
                    text = sentence.replace('_', option)
                else:
                    text = sentence + ' ' + option
                return {'text': text}
            dataset = dataset.map(format_winogrande)
            logger.info(f"Loaded Winogrande dataset with {len(dataset)} samples (自然语言化处理)")
        except Exception as e:
            logger.error(f"Error loading Winogrande dataset: {e}")
            raise
    
    if num_samples:
        total = len(dataset)
        indices = random.sample(range(total), min(num_samples, total))
        dataset = dataset.select(indices)
        logger.info(f"After random sampling: {len(dataset)} samples selected")
    
    return dataset

def process_dataset(dataset, model, tokenizer, max_length, batch_size=1, target_layers=None):
    num_experts = model.config.num_experts
    stats = ExpertActivationStats(num_layers=model.config.num_hidden_layers, 
                                 num_experts=num_experts,
                                 target_layers=target_layers,
                                 pad_token_id=tokenizer.pad_token_id)
    sequence_idx = 0

    # 检查是否是Winogrande结构
    is_winogrande = all(x in dataset.column_names for x in ['sentence', 'option1', 'option2', 'answer'])

    if is_winogrande:
        # 只拼接一条长文本，token数大于max_length
        text_field = get_text_field_name(dataset)
        # 随机打乱样本顺序
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        joined_text = ""
        token_count = 0
        for idx in indices:
            candidate = dataset[idx][text_field]
            temp_text = (joined_text + " " + candidate).strip() if joined_text else candidate
            temp_tokens = tokenizer(temp_text, truncation=False, add_special_tokens=False)["input_ids"]
            if len(temp_tokens) > max_length:
                joined_text = temp_text  # 保证拼接后token数大于max_length
                token_count = len(temp_tokens)
                break
            joined_text = temp_text
            token_count = len(temp_tokens)
        if not joined_text:
            # 单条样本都超长，直接截断
            joined_text = dataset[indices[0]][text_field]
            token_count = len(tokenizer(joined_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"])
        # 输入验证日志
        logger.info("==== 输入验证 ====")
        logger.info(f"Input text: {joined_text}")
        logger.info(f"Token count (拼接后): {token_count}")
        try:
            decoded = tokenizer.decode(tokenizer(joined_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"])
        except Exception as e:
            decoded = f"[decode error: {e}]"
        logger.info(f"Decoded (截断后): {decoded}")
        
        inputs = tokenizer([joined_text], return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)
        # 打印模型输出的keys
        if hasattr(outputs, '__dict__'):
            logger.info(f"Model output keys: {list(outputs.__dict__.keys())}")
        elif hasattr(outputs, 'keys'):
            logger.info(f"Model output keys: {outputs.keys()}")
        else:
            logger.info(f"Model output type: {type(outputs)}")
        # 打印expert_activations内容
        if hasattr(outputs, 'expert_activations'):
            logger.info(f"expert_activations: {outputs.expert_activations}")
        if hasattr(outputs, "expert_activations"):
            stats.update(outputs.expert_activations, sequence_idx, inputs['input_ids'])
        sequence_idx += 1
    else:
        # 非Winogrande数据集，原逻辑
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            text_field = get_text_field_name(dataset)
            texts = batch[text_field]
            try:
                inputs = tokenizer(list(texts), return_tensors="pt", truncation=True, max_length=max_length, padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                for b in range(len(texts)):
                    single_inputs = {k: v[b:b+1] for k, v in inputs.items()}
                    sample_text = texts[b]
                    input_ids = single_inputs['input_ids'][0].tolist()
                    token_count = len(input_ids)
                    non_pad_count = sum(1 for x in input_ids if x != tokenizer.pad_token_id)
                    # 输入验证日志
                    logger.info("==== 输入验证 ====")
                    logger.info(f"Input text: {sample_text}")
                    logger.info(f"Input ids: {input_ids}")
                    try:
                        decoded = tokenizer.decode(single_inputs['input_ids'][0])
                    except Exception as e:
                        decoded = f"[decode error: {e}]"
                    logger.info(f"Decoded: {decoded}")
                    logger.info(f"Sample {sequence_idx}: text={repr(sample_text)[:200]} ... (len={len(sample_text)}) | token_count={token_count} | non_pad_count={non_pad_count} | input_ids={input_ids[:20]}{'...' if token_count>20 else ''}")
                    with torch.no_grad():
                        outputs = model(**single_inputs, output_router_logits=True)
                    # 打印模型输出的keys
                    if hasattr(outputs, '__dict__'):
                        logger.info(f"Model output keys: {list(outputs.__dict__.keys())}")
                    elif hasattr(outputs, 'keys'):
                        logger.info(f"Model output keys: {outputs.keys()}")
                    else:
                        logger.info(f"Model output type: {type(outputs)}")
                    # 打印expert_activations内容
                    if hasattr(outputs, 'expert_activations'):
                        logger.info(f"expert_activations: {outputs.expert_activations}")
                    if hasattr(outputs, "expert_activations"):
                        stats.update(outputs.expert_activations, sequence_idx, single_inputs['input_ids'])
                    sequence_idx += 1
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
    return stats

def plot_aggregated_results(results_dir, output_dir, show_mean=False, target_layers=None):
    """汇总所有实验结果并生成对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 24,           # 全局字体大小
        'font.weight': 'bold',     # 全局字体粗细
        'axes.titlesize': 24,      # 标题字体
        'axes.labelsize': 24,      # 坐标轴标签字体
        'xtick.labelsize': 24,     # x轴刻度字体
        'ytick.labelsize': 24,     # y轴刻度字体
        'legend.fontsize': 23      # 图例字体
    })
    
    # 收集所有实验结果
    all_results = []
    for result_file in glob.glob(os.path.join(results_dir, "**/expert_activation_stats.json"), recursive=True):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                # 从文件路径中提取max_length和repeat信息
                dir_name = os.path.basename(os.path.dirname(result_file))
                # 解析目录名，格式为：bs1_20250523_040249_wikitext_maxlen16_repeat1
                parts = dir_name.split('_')
                # 找到包含maxlen的部分
                maxlen_part = next(part for part in parts if part.startswith('maxlen'))
                max_length = int(maxlen_part.replace('maxlen', ''))
                # 找到包含repeat的部分
                repeat_part = next(part for part in parts if part.startswith('repeat'))
                repeat_idx = int(repeat_part.replace('repeat', '')) - 1  # 减1是因为目录名中的repeat是从1开始的
                all_results.append((max_length, repeat_idx, data))
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")
    
    if not all_results:
        logger.warning("No results found to aggregate")
        return
    
    # 按max_length和repeat_idx排序
    all_results.sort(key=lambda x: (x[0], x[1]))
    
    # 确定要处理的层
    num_layers = len(all_results[0][2]['expert_token_count'])
    layers_to_process = target_layers if target_layers is not None else range(num_layers)
    
    # 为每个max_length分配一个基础颜色
    max_lengths = sorted(set(x[0] for x in all_results))
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(max_lengths)))  # 使用tab10色板
    
    # 为每个指定的层创建汇总图
    for layer in layers_to_process:
        plt.figure(figsize=(15, 8))
        
        # 为每个max_length的所有重复实验绘制曲线
        max_length_data = defaultdict(list)
        for max_length, repeat_idx, data in all_results:
            expert_counts = np.array(data['expert_token_count'][layer])
            sorted_indices = np.argsort(-expert_counts)
            sorted_counts = expert_counts[sorted_indices]
            max_length_data[max_length].append((repeat_idx, sorted_counts))
        
        # 为每个max_length绘制所有重复实验的曲线
        for i, max_length in enumerate(sorted(max_length_data.keys())):
            base_color = base_colors[i]
            for repeat_idx, counts in max_length_data[max_length]:
                # 使用相同的基础颜色，但调整透明度
                alpha = 0.3 + (repeat_idx * 0.2)  # 透明度从0.3到0.7
                plt.plot(range(len(counts)), counts, 
                        label=f'token_num={max_length} R{repeat_idx + 1}', 
                        color=base_color,
                        marker='o', 
                        markersize=4,
                        alpha=alpha)  # 使用调整后的透明度
            
            # 如果需要显示平均值
            if show_mean:
                counts_array = np.array([c for _, c in max_length_data[max_length]])
                mean_counts = np.mean(counts_array, axis=0)
                std_counts = np.std(counts_array, axis=0)
                
                # 绘制平均值曲线（使用相同的基础颜色，但更粗的线条和更高的透明度）
                plt.plot(range(len(mean_counts)), mean_counts, 
                        label=f'token_num={max_length} mean', 
                        color=base_color,
                        linewidth=2,
                        alpha=0.8)
                # 添加标准差区间
                plt.fill_between(range(len(mean_counts)), 
                               mean_counts - std_counts,
                               mean_counts + std_counts,
                               color=base_color,
                               alpha=0.1)
        
        # plt.title(f'Expert Activation Distribution Comparison - Layer {layer}')
        # plt.xlabel('Expert Index (sorted by token count)')
        # plt.ylabel('Number of Tokens')
        plt.grid(True, linestyle='--', alpha=0.7)
        # 去掉上边框和右边框
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 去掉x轴刻度
        ax.set_xticks([])
        # 调整图例位置和布局
        legend = plt.legend(
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            ncol=2,                # 两列
            framealpha=0.8,
            columnspacing=1.0,     # 列间距
            handletextpad=0.5,     # 图例标记和文本之间的间距
            borderpad=0.5,         # 图例边框的内边距
            labelspacing=0.5,      # 图例项之间的垂直间距
            prop={'size': 30, 'weight': 'normal'}  # legend字体大一号且不加粗
        )
        ax.tick_params(axis='y', labelsize=29, width=0)
        # y轴刻度字体不加粗
        for label in ax.get_yticklabels():
            label.set_weight('normal')
        # 调整图形布局以适应图例
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, f'layer_{layer}_comparison_zwh.png'), bbox_inches='tight')
        plt.close()

def main():
    args = parse_args()
    stats_dir = args.output_dir

    # 每次运行先清空expert_stats_sweep文件夹
    if os.path.exists(stats_dir):
        shutil.rmtree(stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    
    # 解析max_lengths
    max_lengths = [int(x.strip()) for x in args.max_lengths.split(',')]
    
    # 解析target_layers
    if args.target_layers is not None:
        target_layers = [int(x.strip()) for x in args.target_layers.split(',')]
    else:
        target_layers = None
    
    logger.info(f"Experiment configuration:")
    logger.info(f"Dataset type: {args.dataset_type}")
    if args.dataset_type == "mmlu":
        logger.info(f"MMLU subset: {args.mmlu_subset}")
    logger.info(f"Max lengths to test: {max_lengths}")
    logger.info(f"Number of samples per experiment: {args.num_samples}")
    logger.info(f"Number of experiment repeats: {args.num_repeats}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Target layers: {target_layers}")
    logger.info(f"Min text length: {args.min_text_length}")

    # 加载模型和分词器（只加载一次）
    logger.info("Loading model and tokenizer...")
    model_name = "model/qwen/Qwen3-30B-A3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3MoeForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # 为每个max_length运行实验
    for max_length in max_lengths:
        logger.info(f"\nProcessing max_length={max_length}")
        
        # 为每个重复实验创建子目录
        for repeat_idx in range(args.num_repeats):
            logger.info(f"Running repeat {repeat_idx + 1}/{args.num_repeats}")
            
            # 为每个实验创建时间戳子目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_tag = f"{args.dataset_type}"
            if args.dataset_type == "mmlu":
                dataset_tag += f"_{args.mmlu_subset}"
            run_tag = f"bs{args.batch_size}_{timestamp}_{dataset_tag}_maxlen{max_length}_repeat{repeat_idx + 1}"
            run_dir = os.path.join(stats_dir, run_tag)
            os.makedirs(run_dir, exist_ok=True)
            
            try:
                # 加载数据集
                dataset = load_and_prepare_dataset(
                    args.dataset_name,
                    args.dataset_type,
                    args.mmlu_subset,
                    "train",
                    args.num_samples,
                    args.min_text_length
                )
                
                # 处理数据集并收集统计信息
                stats = process_dataset(
                    dataset,
                    model,
                    tokenizer,
                    max_length,
                    args.batch_size,
                    target_layers
                )
                
                # 保存统计结果
                output_file = os.path.join(run_dir, "expert_activation_stats.json")
                stats.save(output_file)
                
                logger.info(f"Results saved to {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing max_length={max_length}, repeat={repeat_idx + 1}: {e}")
                continue
    
    # 生成汇总图表
    logger.info("\nGenerating aggregated plots...")
    plots_dir = os.path.join(stats_dir, "aggregated_plots")
    plot_aggregated_results(stats_dir, plots_dir, show_mean=args.show_mean, target_layers=target_layers)
    logger.info(f"Aggregated plots saved to {plots_dir}")

if __name__ == "__main__":
    main()