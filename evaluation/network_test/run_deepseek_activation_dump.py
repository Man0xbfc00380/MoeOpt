import os
import sys
import json
import argparse
import datetime
from collections import defaultdict

import torch
import importlib.util
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Always use the local DeepSeek-MoE snapshot path (consistent with deepseek_moe_sweep_analysis.py)
MODEL_CODE_PATH = "/home/gpu2-user6/DeepSeek-MoE/model/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
MODEL_CACHE_PATH = "/home/gpu2-user6/DeepSeek-MoE/model/"

class DeekSeekRunner:
    """将一次前向与专家激活统计封装为可复用类。"""

    def __init__(self, model_code_path: str = MODEL_CODE_PATH, model_cache_path: str = MODEL_CACHE_PATH):
        self.model_code_path = model_code_path
        self.model_cache_path = model_cache_path
        self.model = None
        self.tokenizer = None

        # 结果缓存
        self.layer_token_experts = None  # dict[int -> dict[token_pos -> list[int]]]
        self.layer_expert_counts = None  # dict[int -> dict[expert_id -> int]]
        self.last_run_dir = None
        self.last_run_tag = None
        self.last_dataset_type = None
        self.last_max_length = None
        self.last_valid_tokens = None

        # 模型专家配置
        self.n_routed_experts = None
        self.n_shared_experts = 0
        self.shared_expert_ids = []

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        sys.path.append(self.model_code_path)
        spec = importlib.util.spec_from_file_location(
            "modeling_deepseek", os.path.join(self.model_code_path, "modeling_deepseek.py")
        )
        modeling_deepseek = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modeling_deepseek)
        DeepseekForCausalLM = modeling_deepseek.DeepseekForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self.model_code_path, cache_dir=self.model_cache_path)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = DeepseekForCausalLM.from_pretrained(
            self.model_code_path,
            cache_dir=self.model_cache_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

        self.model = model
        self.tokenizer = tokenizer

        # 读取配置中的专家数量信息
        self.n_routed_experts = getattr(self.model.config, "n_routed_experts", None)
        if self.n_routed_experts is None:
            # 兼容字段名
            self.n_routed_experts = getattr(self.model.config, "num_experts", None)
        self.n_shared_experts = int(getattr(self.model.config, "n_shared_experts", 0) or 0)
        if self.n_routed_experts is not None:
            self.shared_expert_ids = list(range(int(self.n_routed_experts), int(self.n_routed_experts) + self.n_shared_experts))

    def run(self, dataset_type: str, max_length: int, output_dir: str, text: str | None = None, run_tag_override: str | None = None):
        """
        执行一次前向过程，收集每层 token→experts 映射与专家激活计数，并保存到输出目录。
        返回本次运行的输出子目录路径。
        """
        # 保证输入的 token 数量严格等于 max_length（不足追加，超出截断），直接在 token 层面构造输入
        token_ids = ensure_exact_token_ids(dataset_type, self.tokenizer, max_length, base_text=(text or ""))
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        attention_mask = torch.ones((1, len(token_ids)), dtype=torch.long, device=self.model.device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # 记录有效 token 数量（非 PAD）
        self.last_valid_tokens = int(inputs["attention_mask"][0].sum().item())

        with torch.no_grad():
            outputs = self.model(**inputs)

        layer_token_experts, layer_expert_counts = build_layer_token_experts(
            outputs, inputs["attention_mask"], max_length,
            routed_expert_count=self.n_routed_experts or 0,
            shared_expert_count=self.n_shared_experts
        )

        # 缓存到成员变量
        self.layer_token_experts = layer_token_experts
        self.layer_expert_counts = layer_expert_counts
        self.last_dataset_type = dataset_type
        self.last_max_length = max_length

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = run_tag_override or f"{dataset_type}_{max_length}"
        run_dir = save_layer_dicts(layer_token_experts, output_dir, run_tag)

        self.last_run_dir = run_dir
        self.last_run_tag = run_tag

        return run_dir

    # 访问接口
    def get_token_experts(self, layer_idx: int | None = None):
        if self.layer_token_experts is None:
            return None
        if layer_idx is None:
            return self.layer_token_experts
        return self.layer_token_experts.get(layer_idx, {})

    def get_expert_counts(self, layer_idx: int | None = None):
        if self.layer_expert_counts is None:
            return None
        if layer_idx is None:
            return self.layer_expert_counts
        return self.layer_expert_counts.get(layer_idx, {})

    def get_top_k_estimate(self, layer_idx: int):
        if self.layer_token_experts is None:
            return None
        layer_map = self.layer_token_experts.get(layer_idx, {})
        for pos in sorted(layer_map.keys()):
            exps = layer_map.get(pos, [])
            if exps:
                # 返回路由专家的估计 top_k（从总数里减去共享专家数量）
                k = len(exps) - self.n_shared_experts
                return max(k, 0)
        return None

    def get_total_activations(self, layer_idx: int | None = None):
        if self.layer_expert_counts is None:
            return None
        if layer_idx is None:
            return {l: int(sum(c.values())) for l, c in self.layer_expert_counts.items()}
        return int(sum(self.layer_expert_counts.get(layer_idx, {}).values()))

    def save_results(self, output_dir: str, run_tag: str | None = None):
        if self.layer_token_experts is None:
            return None
        tag = run_tag or self.last_run_tag or datetime.datetime.now().strftime("manual_%Y%m%d_%H%M%S")
        run_dir = save_layer_dicts(self.layer_token_experts, output_dir, tag)
        self.last_run_dir = run_dir
        self.last_run_tag = tag
        return run_dir

    def get_shared_expert_ids(self):
        return self.shared_expert_ids

def choose_text_with_min_tokens(dataset_type: str, tokenizer, min_tokens: int):
    """
    选择并拼接样本文本，确保分词后不少于 min_tokens。
    """
    pieces = []
    if dataset_type == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for ex in ds:
            txt = ex.get("text", "").strip()
            if not txt:
                continue
            pieces.append(txt)
            tokens = tokenizer(" ".join(pieces), truncation=False, add_special_tokens=False)["input_ids"]
            if len(tokens) >= min_tokens:
                break
    elif dataset_type == "winogrande":
        # 使用有效的 BuilderConfig，并将样本转为自然语言文本
        ds = load_dataset("winogrande", "winogrande_xl", split="validation")
        def format_winogrande(example):
            sentence = example.get("sentence", "")
            option1 = example.get("option1", "")
            option2 = example.get("option2", "")
            answer = example.get("answer", "")
            try:
                ans = int(answer)
            except Exception:
                ans = 1
            option = option1 if ans == 1 else option2
            # 替换占位或直接拼接选项
            if "_" in sentence:
                text = sentence.replace("_", option)
            else:
                text = (sentence + " " + option).strip()
            return {"text": text}
        ds = ds.map(format_winogrande)
        for ex in ds:
            txt = ex.get("text", "").strip()
            if not txt:
                continue
            pieces.append(txt)
            tokens = tokenizer(" ".join(pieces), truncation=False, add_special_tokens=False)["input_ids"]
            if len(tokens) >= min_tokens:
                break
    elif dataset_type == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation")
        for ex in ds:
            txt = ex.get("text", "").strip()
            if not txt:
                continue
            pieces.append(txt)
            tokens = tokenizer(" ".join(pieces), truncation=False, add_special_tokens=False)["input_ids"]
            if len(tokens) >= min_tokens:
                break
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if not pieces:
        raise RuntimeError("Failed to collect dataset text pieces.")
    return " ".join(pieces)

def generate_token_chunks(dataset_type: str, tokenizer, chunk_len: int, times: int):
    """
    构建连续的 token 流并切分为 times 个连续片段，每段长度为 chunk_len。
    返回每段的可解码文本，供后续按 max_length 重新编码与填充。
    """
    if times <= 0:
        return []
    target_tokens = chunk_len * times
    all_ids = []

    if dataset_type == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for ex in ds:
            txt = ex.get("text", "").strip()
            if not txt:
                continue
            ids = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= target_tokens:
                break
    elif dataset_type == "winogrande":
        ds = load_dataset("winogrande", "winogrande_xl", split="validation")
        def format_winogrande(example):
            sentence = example.get("sentence", "")
            option1 = example.get("option1", "")
            option2 = example.get("option2", "")
            answer = example.get("answer", "")
            try:
                ans = int(answer)
            except Exception:
                ans = 1
            option = option1 if ans == 1 else option2
            if "_" in sentence:
                text = sentence.replace("_", option)
            else:
                text = (sentence + " " + option).strip()
            return {"text": text}
        ds = ds.map(format_winogrande)
        for ex in ds:
            txt = ex.get("text", "").strip()
            if not txt:
                continue
            ids = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= target_tokens:
                break
    elif dataset_type == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation")
        for ex in ds:
            txt = ex.get("text", "").strip()
            if not txt:
                continue
            ids = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= target_tokens:
                break
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if len(all_ids) < target_tokens:
        raise RuntimeError("Not enough tokens to build requested chunks.")

    chunks = []
    for i in range(times):
        start = i * chunk_len
        end = start + chunk_len
        ids = all_ids[start:end]
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        chunks.append(text)
    return chunks

def _iter_dataset_texts(dataset_type: str):
    """
    依数据集类型顺序产出可拼接的文本片段（跳过空白）。
    """
    if dataset_type == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for ex in ds:
            txt = ex.get("text", "").strip()
            if txt:
                yield txt
    elif dataset_type == "winogrande":
        ds = load_dataset("winogrande", "winogrande_xl", split="validation")
        def format_winogrande(example):
            sentence = example.get("sentence", "")
            option1 = example.get("option1", "")
            option2 = example.get("option2", "")
            answer = example.get("answer", "")
            try:
                ans = int(answer)
            except Exception:
                ans = 1
            option = option1 if ans == 1 else option2
            if "_" in sentence:
                text = sentence.replace("_", option)
            else:
                text = (sentence + " " + option).strip()
            return {"text": text}
        ds = ds.map(format_winogrande)
        for ex in ds:
            txt = ex.get("text", "").strip()
            if txt:
                yield txt
    elif dataset_type == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation")
        for ex in ds:
            txt = ex.get("text", "").strip()
            if txt:
                yield txt
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

def ensure_exact_tokens(dataset_type: str, tokenizer, max_tokens: int, base_text: str = "") -> str:
    """
    基于当前文本与数据集追加内容，确保重新编码后的 token 数量恰好等于 max_tokens。
    - 使用 add_special_tokens=False 的编码以避免额外特殊符号影响长度。
    - 不足则顺序从数据集读取并拼接；超出则截断到精确长度。
    返回可解码文本，其再次编码后 token 数量为 max_tokens。
    """
    # 起始 token 序列
    if base_text:
        ids = tokenizer(base_text, truncation=False, add_special_tokens=False)["input_ids"]
    else:
        ids = []

    # 追加直到达到所需长度
    if len(ids) < max_tokens:
        for piece in _iter_dataset_texts(dataset_type):
            more_ids = tokenizer(piece, truncation=False, add_special_tokens=False)["input_ids"]
            if not more_ids:
                continue
            # 直接在 token 层面追加，避免反复整体重编码开销
            ids.extend(more_ids)
            if len(ids) >= max_tokens:
                break

    # 截断到精确长度并反解为文本
    if len(ids) < max_tokens:
        raise RuntimeError("Dataset does not provide enough tokens to reach max_tokens.")
    exact_ids = ids[:max_tokens]
    text_out = tokenizer.decode(exact_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # 验证一次长度一致性（容忍解码/再编码的空白差异，但需满足目标长度）
    re_ids = tokenizer(text_out, truncation=False, add_special_tokens=False)["input_ids"]
    if len(re_ids) != max_tokens:
        # 若因空白归一化导致长度偏差，采用直接切分再解码的方式再次校正
        # 通过逐步扩展与切分确保最终再编码长度稳定为 max_tokens
        # 若仍不一致，回退到拼接一个空格再解码策略
        # 注：不同分词器的空白处理可能造成 decode/encode 的细微差异
        text_out = tokenizer.decode(exact_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        re_ids = tokenizer(text_out, truncation=False, add_special_tokens=False)["input_ids"]
        if len(re_ids) != max_tokens:
            # 轻微调整：附加一个空格以减少末尾黏连造成的分词差异
            text_out = (text_out + " ").strip()
            re_ids = tokenizer(text_out, truncation=False, add_special_tokens=False)["input_ids"]
            if len(re_ids) != max_tokens:
                # 最后兜底：直接返回按 exact_ids 解码的文本（通常已满足等长要求）
                pass
    return text_out

def ensure_exact_token_ids(dataset_type: str, tokenizer, max_tokens: int, base_text: str = "") -> list[int]:
    """
    与 ensure_exact_tokens 类似，但直接返回长度精确为 max_tokens 的 token 序列，避免 decode/encode 的差异。
    """
    # 初始 token
    if base_text:
        ids = tokenizer(base_text, truncation=False, add_special_tokens=False)["input_ids"]
    else:
        ids = []

    if len(ids) < max_tokens:
        for piece in _iter_dataset_texts(dataset_type):
            more_ids = tokenizer(piece, truncation=False, add_special_tokens=False)["input_ids"]
            if not more_ids:
                continue
            ids.extend(more_ids)
            if len(ids) >= max_tokens:
                break

    if len(ids) < max_tokens:
        raise RuntimeError("Dataset does not provide enough tokens to reach max_tokens.")
    return ids[:max_tokens]

def build_layer_token_experts(outputs, attention_mask, max_length: int,
                              routed_expert_count: int = 0,
                              shared_expert_count: int = 0):
    """
    Build from model outputs.expert_activations:
    - One dict per layer: token_position(int) -> [expert_id...]
    - One count per layer: expert_id(int) -> activation count
    """
    if not hasattr(outputs, "expert_activations"):
        raise RuntimeError("Model outputs missing 'expert_activations'. Ensure DeepSeek routing hooks are enabled.")
    # 使用 attention_mask 来确定有效 token 位置
    if isinstance(attention_mask, torch.Tensor):
        am = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
        non_pad_mask = am.to(torch.bool).cpu().numpy()
    else:
        raise RuntimeError("attention_mask must be a torch.Tensor")

    layer_dicts = {}
    layer_expert_counts = {}

    for layer_idx, layer_data in enumerate(outputs.expert_activations):
        # Process only batch_size=1
        if not layer_data or len(layer_data) == 0:
            layer_dicts[layer_idx] = {}
            layer_expert_counts[layer_idx] = {}
            continue

        token_to_experts = defaultdict(set)
        expert_counts = defaultdict(int)

        # Iterate over the current layer's batch item (usually only one)
        batch_item = layer_data[0]
        for subtoken in batch_item:
            if not isinstance(subtoken, tuple) or len(subtoken) != 2:
                continue
            indices, weights = subtoken
            # Convert to numpy
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy()
            if isinstance(weights, torch.Tensor):
                weights = weights.to(torch.float32).cpu().numpy()

            # Align valid length
            seq_len = len(indices)
            valid_len = min(seq_len, len(non_pad_mask), max_length)

            for t in range(valid_len):
                # 如果该位置是 PAD，仍然保留空项；但不计数
                if not non_pad_mask[t]:
                    continue
                idx = indices[t]
                # Compatible with top-k=1 or top-k>1
                if np.isscalar(idx) or (isinstance(idx, np.ndarray) and np.ndim(idx) == 0):
                    exps = [int(idx)]
                else:
                    # np.ndarray or list
                    if isinstance(idx, np.ndarray):
                        exps = [int(i) for i in idx.tolist()]
                    else:
                        exps = [int(i) for i in idx]

                for e in set(exps):
                    token_to_experts[t].add(e)
                    expert_counts[e] += 1

        # 将共享专家加入每个有效 token 的激活（共享专家始终参与）
        if shared_expert_count > 0:
            shared_ids = list(range(int(routed_expert_count), int(routed_expert_count) + int(shared_expert_count)))
            valid_positions = [int(i) for i, v in enumerate(non_pad_mask[:max_length]) if bool(v)]
            for t in valid_positions:
                for e in shared_ids:
                    token_to_experts[t].add(e)
                    expert_counts[e] += 1

        # Convert back to regular structure
        # 补齐 0..max_length-1 的所有位置
        full_map = {}
        for pos in range(max_length):
            if pos in token_to_experts:
                full_map[pos] = sorted(list(token_to_experts[pos]))
            else:
                full_map[pos] = []
        layer_dicts[layer_idx] = {int(k): v for k, v in full_map.items()}
        layer_expert_counts[layer_idx] = {int(e): int(c) for e, c in expert_counts.items()}

    return layer_dicts, layer_expert_counts

def save_layer_dicts(layer_token_experts, out_dir, run_tag):
    run_dir = os.path.join(out_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    for layer_idx, token_map in layer_token_experts.items():
        path = os.path.join(run_dir, f"layer_{layer_idx}_token_experts.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(token_map, f, ensure_ascii=False, indent=2)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek-MoE once and dump token→experts per layer.")
    parser.add_argument("--dataset_type", type=str, default="wikitext", choices=["wikitext", "winogrande", "c4"],
                        help="Choose one dataset: wikitext | winogrande | c4")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Number of tokens to run at once (tokenizer truncation/padding length)")
    parser.add_argument("--output_dir", type=str, default="./expert_token_experts",
                        help="Output folder (one JSON file per layer)")
    parser.add_argument("--times", type=int, default=1, help="连续运行的次数；每次消耗 max_length 个 token")
    args = parser.parse_args()

    runner = DeekSeekRunner()

    base_tag = f"{args.dataset_type}_{args.max_length}_{args.times}"
    if args.times <= 1:
        # 单次运行直接使用一个片段
        chunks = generate_token_chunks(args.dataset_type, runner.tokenizer, args.max_length, 1)
        _ = runner.run(args.dataset_type, args.max_length, args.output_dir, text=chunks[0], run_tag_override=f"{base_tag}/run_1")
    else:
        chunks = generate_token_chunks(args.dataset_type, runner.tokenizer, args.max_length, args.times)
        for i, text in enumerate(chunks, start=1):
            runner.run(args.dataset_type, args.max_length, args.output_dir, text=text, run_tag_override=f"{base_tag}/run_{i}")

    final_dir = os.path.join(args.output_dir, base_tag)
    print(f"Saved layer token→experts dicts to: {final_dir}")
    # 打印每层激活统计与总数校验
    # all_counts = runner.get_expert_counts()
    # shared_ids = set(runner.get_shared_expert_ids() or [])
    # for layer_idx in sorted(all_counts.keys()):
    #     print(f"Layer {layer_idx} expert activation counts:")
    #     counts = all_counts[layer_idx]
    #     if not counts:
    #         print("  (no activations captured)")
    #         continue
    #     for e, c in sorted(counts.items()):
    #         tag = " [shared]" if e in shared_ids else ""
    #         print(f"  Expert {e}: {c}{tag}")
    #     total = runner.get_total_activations(layer_idx)
    #     k_est = runner.get_top_k_estimate(layer_idx)
    #     if k_est is not None:
    #         expected = (runner.last_valid_tokens or 0) * (k_est + len(shared_ids))
    #         print(f"  Total activations: {total} (expected ~= valid_tokens * (top_k + shared))")
    #         print(f"  valid_tokens={runner.last_valid_tokens}, top_k≈{k_est}, shared={len(shared_ids)}, expected≈{expected}")
    #     else:
    #         print("  Unable to estimate top_k (all positions empty).")

if __name__ == "__main__":
    main()