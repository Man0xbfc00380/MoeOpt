import os
import sys
import json
import argparse
import datetime
from collections import defaultdict

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# 使用 Hugging Face 模型 ID，保证与本地缓存协同
MODEL_ID = "microsoft/Phi-3.5-MoE-instruct"
MODEL_CACHE_PATH = "/home/gpu2-user6/Phi-3.5-MoE/"  # 与现有目录保持一致


class PhiRunner:
    """将一次前向与专家激活统计封装为可复用类（Phi-3.5-MoE）。

    功能与 run_deepseek_activation_dump.py 的 DeekSeekRunner 完全一致：
    - 按数据集类型采样并拼接文本，保证不少于 max_length 个 token；
    - 运行模型一次前向；
    - 构建每层 token→experts 的映射与专家激活计数；
    - 保存到输出目录；
    - 提供若干便捷的查询/统计接口。
    """

    def __init__(self, model_id: str = MODEL_ID, model_cache_path: str = MODEL_CACHE_PATH):
        self.model_id = model_id
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

        # 模型专家配置（Phi-3.5-MoE 不区分共享专家）
        self.n_routed_experts = None
        self.n_shared_experts = 0
        self.shared_expert_ids = []

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        # 明确设置缓存目录，确保权重/源码落在本地固定路径
        os.environ.setdefault("HF_HOME", self.model_cache_path)
        os.environ.setdefault("TRANSFORMERS_CACHE", self.model_cache_path)
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(self.model_cache_path, "datasets"))

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=self.model_cache_path)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=self.model_cache_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        ).eval()

        self.model = model
        self.tokenizer = tokenizer

        # 读取配置中的专家数量信息
        cfg = self.model.config
        # Phi 模型的专家数量字段为 num_local_experts；每 token top-k 为 num_experts_per_tok
        self.n_routed_experts = int(getattr(cfg, "num_local_experts", 0) or 0)
        # Phi 无共享专家，保持 0
        self.n_shared_experts = 0
        self.shared_expert_ids = []

    def run(self, dataset_type: str, max_length: int, output_dir: str, text: str | None = None, run_tag_override: str | None = None):
        """
        执行一次前向过程，收集每层 token→experts 映射与专家激活计数，并保存到输出目录。
        返回本次运行的输出子目录路径。
        """
        # 构造严格等于 max_length 的输入 token（不足则顺序从数据集追加，超出则截断）
        token_ids = ensure_exact_token_ids(dataset_type, self.tokenizer, max_length, base_text=(text or ""))
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        attention_mask = torch.ones((1, len(token_ids)), dtype=torch.long, device=self.model.device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # 记录有效 token 数量（非 PAD）
        self.last_valid_tokens = int(inputs["attention_mask"][0].sum().item())

        with torch.no_grad():
            _ = self.model(**inputs)

        # 从每层的 block_sparse_moe 中读取专家激活（selected_experts, routing_weights）
        layer_token_experts, layer_expert_counts = build_layer_token_experts_from_phi(
            self.model, inputs["attention_mask"], max_length,
            routed_expert_count=self.n_routed_experts,
            shared_expert_count=self.n_shared_experts,
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
                # 返回路由专家的估计 top_k（Phi 无共享专家）
                return len(exps)
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
    构建连续 token 流，切分为 times 个长度为 chunk_len 的片段，并返回可解码文本。
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

def ensure_exact_token_ids(dataset_type: str, tokenizer, max_tokens: int, base_text: str = "") -> list[int]:
    """
    在 token 层面将 base_text 与数据集文本顺序拼接，保证总长度恰为 max_tokens。
    使用 add_special_tokens=False，避免特殊符号引入长度偏差。
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


def build_layer_token_experts_from_phi(model, attention_mask, max_length: int,
                                       routed_expert_count: int = 0,
                                       shared_expert_count: int = 0):
    """
    从 Phi 模型的每层 block_sparse_moe.expert_activations 构建：
    - 每层一个 dict: token_position(int) -> [expert_id...]
    - 每层一个计数: expert_id(int) -> activation count

    expert_activations 在 Phi 实现中是一个列表，元素为 (selected_experts, routing_weights)。
    selected_experts 形状约为 [batch_size * seq_len, top_k]；routing_weights 同形状。
    """
    if isinstance(attention_mask, torch.Tensor):
        am = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
        non_pad_mask = am.to(torch.bool).cpu().numpy()
    else:
        raise RuntimeError("attention_mask must be a torch.Tensor")

    layer_dicts = {}
    layer_expert_counts = {}

    # 仅支持 PhiMoEForCausalLM
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise RuntimeError("Phi model does not expose decoder layers via model.layers")

    batch_size = int(attention_mask.shape[0])
    seq_len = int(attention_mask.shape[1])

    for layer_idx, layer in enumerate(layers):
        block = getattr(layer, "block_sparse_moe", None)
        if block is None:
            layer_dicts[layer_idx] = {}
            layer_expert_counts[layer_idx] = {}
            continue

        acts = getattr(block, "expert_activations", None)
        if not acts or len(acts) == 0:
            # 没有捕获到激活信息
            layer_dicts[layer_idx] = {int(pos): [] for pos in range(max_length)}
            layer_expert_counts[layer_idx] = {}
            continue

        token_to_experts = defaultdict(set)
        expert_counts = defaultdict(int)

        # Phi 将当前 batch 的所有 token 激活存入一个元组
        selected_experts, routing_weights = acts[0]
        if isinstance(selected_experts, torch.Tensor):
            # 恢复为 [batch, seq_len, top_k]
            top_k = int(selected_experts.shape[-1])
            sel = selected_experts.view(batch_size, seq_len, top_k).cpu().numpy()
        else:
            # 兜底：转 numpy 并尝试 reshape
            arr = np.array(selected_experts)
            top_k = int(arr.shape[-1]) if arr.ndim >= 1 else 1
            sel = arr.reshape(batch_size, seq_len, top_k)

        valid_len = min(seq_len, len(non_pad_mask), max_length)
        for t in range(valid_len):
            if not non_pad_mask[t]:
                continue
            # 仅处理 batch_size=1
            exps = [int(x) for x in np.array(sel[0, t]).flatten().tolist()]
            for e in set(exps):
                token_to_experts[t].add(e)
                expert_counts[e] += 1

        # 共享专家（Phi 无共享专家，保持兼容接口）
        if shared_expert_count > 0:
            shared_ids = list(range(int(routed_expert_count), int(routed_expert_count) + int(shared_expert_count)))
            valid_positions = [int(i) for i, v in enumerate(non_pad_mask[:max_length]) if bool(v)]
            for t in valid_positions:
                for e in shared_ids:
                    token_to_experts[t].add(e)
                    expert_counts[e] += 1

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
    parser = argparse.ArgumentParser(description="Run Phi-3.5-MoE once and dump token→experts per layer.")
    parser.add_argument("--dataset_type", type=str, default="wikitext", choices=["wikitext", "winogrande", "c4"],
                        help="Choose one dataset: wikitext | winogrande | c4")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Number of tokens to run at once (tokenizer truncation/padding length)")
    parser.add_argument("--output_dir", type=str, default="./expert_token_experts",
                        help="Output folder (one JSON file per layer)")
    parser.add_argument("--times", type=int, default=1, help="连续运行的次数；每次消耗 max_length 个 token")
    args = parser.parse_args()

    runner = PhiRunner()

    base_tag = f"{args.dataset_type}_{args.max_length}_{args.times}"
    if args.times <= 1:
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