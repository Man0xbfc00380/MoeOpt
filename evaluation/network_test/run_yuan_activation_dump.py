import os
import json
import argparse
import datetime
from collections import defaultdict

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, LlamaTokenizer


# 默认使用 Hugging Face 上的 Yuan2 MoE 权重（与现有示例一致）
DEFAULT_MODEL_ID = "IEITYuan/Yuan2-M32-hf"


class YuanRunner:
    """封装一次前向与专家激活导出，格式与 QwenRunner 对齐。"""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
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

        # 模型专家配置（Yuan2-M32 为 MoE；若非 MoE 则捕获为空结构）
        self.top_k = None
        self.num_experts = None

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        # 加载 tokenizer（Yuan 使用 LlamaTokenizer 并添加特殊 token）
        tokenizer = LlamaTokenizer.from_pretrained(
            self.model_id,
            add_eos_token=False,
            add_bos_token=False,
            eos_token="<eod>",
            trust_remote_code=True,
        )
        tokenizer.add_tokens(
            [
                "<sep>", "<pad>", "<mask>", "<predict>",
                "<FIM_SUFFIX>", "<FIM_PREFIX>", "<FIM_MIDDLE>",
                "<commit_before>", "<commit_msg>", "<commit_after>",
                "<jupyter_start>", "<jupyter_text>", "<jupyter_code>",
                "<jupyter_output>", "<empty_output>",
            ],
            special_tokens=True,
        )
        # 明确设置 PAD token，避免在使用 padding 时抛错
        if tokenizer.pad_token is None:
            added_vocab = tokenizer.get_added_vocab() or {}
            if "<pad>" in added_vocab:
                tokenizer.pad_token = "<pad>"
            elif tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
        ).eval()

        # 词嵌入矩阵大小需要与 tokenizer 同步
        model.resize_token_embeddings(len(tokenizer))

        # 设备放置
        if torch.cuda.is_available():
            model = model.to("cuda:0")
        model_device = next(model.parameters()).device

        self.model = model
        self.tokenizer = tokenizer
        self.device = model_device

        # 缓存配置（若可用）
        try:
            cfg = getattr(self.model, "config", None)
            self.num_experts = int(getattr(cfg, "moe_config", {}).get("moe_num_experts", 0) or 0)
            self.top_k = int(getattr(cfg, "moe_config", {}).get("moe_top_k", 0) or 0)
        except Exception:
            self.num_experts = None
            self.top_k = None

    def run(self, dataset_type: str, max_length: int, output_dir: str, text: str | None = None, run_tag_override: str | None = None):
        """
        执行一次前向过程，收集每层 token→experts 映射与专家激活计数，并保存到输出目录。
        返回本次运行的输出子目录路径。
        """
        # 构造精确长度为 max_length 的 input_ids 与 attention_mask（均为 1）
        input_ids, attention_mask = ensure_exact_token_ids(
            dataset_type=dataset_type,
            tokenizer=self.tokenizer,
            max_tokens=max_length,
            base_text=text,
        )
        # pad_token 兜底（虽然本实现不依赖 padding，但保持安全设置）
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

        # 记录有效 token 数量（非 PAD）
        self.last_valid_tokens = int(inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))[0].sum().item())

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # 解析 expert_activations（兼容 Yuan 模型中的占位结构或真实 MoE 输出）
        layer_token_experts, layer_expert_counts = build_layer_token_experts_yuan(
            outputs, inputs.get("attention_mask", None), max_length
        )
        # 当模型未返回专家信息时，按层数生成空映射，保持输出结构稳定
        if not layer_token_experts:
            try:
                n_layers = int(getattr(self.model.config, "num_hidden_layers", 0) or 0)
            except Exception:
                n_layers = 0
            for li in range(n_layers):
                layer_token_experts[li] = {int(pos): [] for pos in range(max_length)}
                layer_expert_counts[li] = {}

        # 缓存到成员变量
        self.layer_token_experts = layer_token_experts
        self.layer_expert_counts = layer_expert_counts
        self.last_dataset_type = dataset_type
        self.last_max_length = max_length

        # 与 QwenRunner 保持一致的命名，可覆盖
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

    def get_top_k_estimate(self):
        return self.top_k

    def get_total_activations(self, layer_idx: int | None = None):
        if self.layer_expert_counts is None:
            return None
        if layer_idx is None:
            return {l: int(sum(c.values())) for l, c in self.layer_expert_counts.items()}
        return int(sum(self.layer_expert_counts.get(layer_idx, {}).values()))


def choose_text_with_min_tokens(dataset_type: str, tokenizer, min_tokens: int):
    """
    选择并拼接样本文本，确保分词后至少 min_tokens。
    支持类型：wikitext | winogrande | c4
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

def _iter_dataset_texts(dataset_type: str):
    """
    逐条迭代给定数据集中的文本片段：支持 wikitext | winogrande | c4。
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

def ensure_exact_token_ids(dataset_type: str, tokenizer, max_tokens: int, base_text: str | None = None):
    """
    返回严格长度为 max_tokens 的 input_ids 与 attention_mask（全 1）。
    先使用 base_text 的 token（若给定），不足则从数据集继续追加，超过则截断。
    使用 add_special_tokens=False 以避免长度不确定性。
    """
    # 起始 token 序列
    ids = []
    if base_text is not None and base_text.strip():
        ids = tokenizer(base_text, truncation=False, add_special_tokens=False)["input_ids"]

    # 超过则截断到精确长度
    if len(ids) >= max_tokens:
        ids = ids[:max_tokens]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    # 不足则从数据集追加直到恰好等于 max_tokens
    for piece in _iter_dataset_texts(dataset_type):
        piece_ids = tokenizer(piece, truncation=False, add_special_tokens=False)["input_ids"]
        if not piece_ids:
            continue
        remain = max_tokens - len(ids)
        if len(piece_ids) <= remain:
            ids.extend(piece_ids)
        else:
            ids.extend(piece_ids[:remain])
        if len(ids) == max_tokens:
            break

    if len(ids) != max_tokens:
        raise RuntimeError("Not enough tokens to reach the requested max_length.")

    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

def generate_token_chunks(dataset_type: str, tokenizer, chunk_len: int, times: int):
    """
    构建连续的 token 流，切分为 times 个长度为 chunk_len 的片段，并返回可解码文本。
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


def build_layer_token_experts_yuan(outputs, attention_mask, max_length: int):
    """
    构建每层映射：
    - token_position(int) -> [expert_id...]
    - expert_id(int) -> activation count
    解析自 Yuan 模型 outputs.expert_activations。
    兼容两种结构：
    1) 每层保存 [(selected_experts, routing_weights)]，形状 (B, S, top_k)
    2) 非 MoE 情况：每层保存空激活（每位置 [])
    """
    expert_acts = getattr(outputs, "expert_activations", None)
    layer_dicts = {}
    layer_expert_counts = {}

    # 有效 token 位置
    if isinstance(attention_mask, torch.Tensor):
        am = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
        non_pad_mask = am.to(torch.bool).cpu().numpy()
    else:
        non_pad_mask = np.ones((max_length,), dtype=bool)

    if not expert_acts:
        # 没有捕获到激活，返回空结构（不生成任何层文件）
        return ({}, {})

    # expert_activations 是按层的列表；每层内容可能是：
    # 1) None
    # 2) [(selected_experts, routing_weights)]
    # 3) (selected_experts, routing_weights)
    # 4) { 'expert_activations': { 'selected_experts': ..., 'routing_weights': ... }, 'layer_idx': ... }

    def _extract_sel_wts(layer_act):
        try:
            if layer_act is None:
                return None, None
            # dict 结构（兼容 Qwen 风格）
            if isinstance(layer_act, dict):
                ex_data = layer_act.get("expert_activations", layer_act)
                sel = ex_data.get("selected_experts", None)
                wts = ex_data.get("routing_weights", None)
                return sel, wts
            # tuple 直接是 (sel, wts)
            if isinstance(layer_act, tuple) and len(layer_act) == 2:
                return layer_act[0], layer_act[1]
            # list 可能是 [ (sel, wts) ] 或两元素 (sel, wts)
            if isinstance(layer_act, list):
                if len(layer_act) == 2 and not isinstance(layer_act[0], (list, tuple)):
                    return layer_act[0], layer_act[1]
                if len(layer_act) >= 1 and isinstance(layer_act[0], (list, tuple)) and len(layer_act[0]) == 2:
                    return layer_act[0][0], layer_act[0][1]
            return None, None
        except Exception:
            return None, None

    for layer_idx, layer_act in enumerate(expert_acts):
        token_to_experts = defaultdict(set)
        expert_counts = defaultdict(int)

        if not layer_act:
            # 空层：填充空映射
            layer_dicts[layer_idx] = {int(pos): [] for pos in range(max_length)}
            layer_expert_counts[layer_idx] = {}
            continue

        selected_experts, routing_weights = _extract_sel_wts(layer_act)
        if selected_experts is None:
            layer_dicts[layer_idx] = {int(pos): [] for pos in range(max_length)}
            layer_expert_counts[layer_idx] = {}
            continue

        # 将 selected_experts 规整为形状 (seq_len, top_k)，只处理 batch_size=1
        if isinstance(selected_experts, torch.Tensor):
            se = selected_experts.detach().cpu().numpy()
        else:
            se = np.array(selected_experts)

        if se.ndim == 3:
            # (batch, seq_len, top_k)
            sel = se[0]
        elif se.ndim == 2:
            # (seq_len, top_k)
            sel = se
        elif se.ndim == 1:
            # (top_k,) -> 视为单位置
            sel = se.reshape(1, -1)
        else:
            sel = np.empty((max_length, 0), dtype=int)

        seq_len = sel.shape[0] if hasattr(sel, "shape") else max_length
        valid_len = min(seq_len, len(non_pad_mask), max_length)

        for t in range(valid_len):
            if not non_pad_mask[t]:
                continue
            idxs = sel[t] if t < len(sel) else []
            # idxs 是长度 top_k 的数组/列表
            if hasattr(idxs, "tolist"):
                idx_list = idxs.tolist()
            else:
                idx_list = list(idxs)
            for e in set(int(i) for i in idx_list):
                token_to_experts[t].add(e)
                expert_counts[e] += 1

        # 统一为 0..max_length-1
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
    parser = argparse.ArgumentParser(description="Run Yuan2-MoE once and dump token→experts per layer.")
    parser.add_argument("--dataset_type", type=str, default="wikitext", choices=["wikitext", "winogrande", "c4"],
                        help="Choose one dataset: wikitext | winogrande | c4")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Tokenizer truncation/padding length for a single forward run")
    parser.add_argument("--output_dir", type=str, default="./yuan_outputs",
                        help="Output folder (one JSON file per layer)")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="Hugging Face Yuan model id")
    parser.add_argument("--times", type=int, default=1, help="连续运行的次数；每次消耗 max_length 个 token")
    args = parser.parse_args()

    runner = YuanRunner(model_id=args.model_id)

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


if __name__ == "__main__":
    main()