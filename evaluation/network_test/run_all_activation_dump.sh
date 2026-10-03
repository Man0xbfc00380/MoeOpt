#!/usr/bin/env bash
set -euo pipefail

# 切换到脚本所在目录，确保相对路径正确
cd "$(dirname "$0")"

# 确保输出目录存在
mkdir -p deepseek_outputs qwen_outputs phi_outputs

echo "[DeepSeek] 开始执行..."
python run_deepseek_activation_dump.py --dataset_type c4 --max_length 16 --output_dir deepseek_outputs/ --times 100
python run_deepseek_activation_dump.py --dataset_type c4 --max_length 64 --output_dir deepseek_outputs/ --times 100
python run_deepseek_activation_dump.py --dataset_type c4 --max_length 256 --output_dir deepseek_outputs/ --times 100
python run_deepseek_activation_dump.py --dataset_type wikitext --max_length 16 --output_dir deepseek_outputs/ --times 100
python run_deepseek_activation_dump.py --dataset_type wikitext --max_length 64 --output_dir deepseek_outputs/ --times 100
python run_deepseek_activation_dump.py --dataset_type wikitext --max_length 256 --output_dir deepseek_outputs/ --times 100

echo "[Qwen] 开始执行..."
python run_qwen_activation_dump.py --dataset_type c4 --max_length 16 --output_dir qwen_outputs/ --times 100
python run_qwen_activation_dump.py --dataset_type c4 --max_length 64 --output_dir qwen_outputs/ --times 100
python run_qwen_activation_dump.py --dataset_type c4 --max_length 256 --output_dir qwen_outputs/ --times 100
python run_qwen_activation_dump.py --dataset_type wikitext --max_length 16 --output_dir qwen_outputs/ --times 100
python run_qwen_activation_dump.py --dataset_type wikitext --max_length 64 --output_dir qwen_outputs/ --times 100
python run_qwen_activation_dump.py --dataset_type wikitext --max_length 256 --output_dir qwen_outputs/ --times 100

echo "[Phi-MoE] 开始执行..."
python run_phimoe_activation_dump.py --dataset_type c4 --max_length 16 --output_dir phi_outputs/ --times 100
python run_phimoe_activation_dump.py --dataset_type c4 --max_length 64 --output_dir phi_outputs/ --times 100
python run_phimoe_activation_dump.py --dataset_type c4 --max_length 256 --output_dir phi_outputs/ --times 100
python run_phimoe_activation_dump.py --dataset_type wikitext --max_length 16 --output_dir phi_outputs/ --times 100
python run_phimoe_activation_dump.py --dataset_type wikitext --max_length 64 --output_dir phi_outputs/ --times 100
python run_phimoe_activation_dump.py --dataset_type wikitext --max_length 256 --output_dir phi_outputs/ --times 100

echo "[Yuan-MoE] 开始执行..."
python run_yuan_activation_dump.py --dataset_type c4 --max_length 16 --output_dir yuan_outputs/ --times 100
python run_yuan_activation_dump.py --dataset_type c4 --max_length 64 --output_dir yuan_outputs/ --times 100
python run_yuan_activation_dump.py --dataset_type c4 --max_length 256 --output_dir yuan_outputs/ --times 100
python run_yuan_activation_dump.py --dataset_type wikitext --max_length 16 --output_dir yuan_outputs/ --times 100
python run_yuan_activation_dump.py --dataset_type wikitext --max_length 64 --output_dir yuan_outputs/ --times 100
python run_yuan_activation_dump.py --dataset_type wikitext --max_length 256 --output_dir yuan_outputs/ --times 100

echo "全部任务执行完成。"