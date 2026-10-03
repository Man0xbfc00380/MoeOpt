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


# Local Qwen3-MoE model code and weights paths
# 默认改为与可用脚本一致的路径（transformers-download 目录）
QWEN3_MODULAR_PATH = "/home/gpu2-user6/Qwen3/transformers-download/modular_qwen3_moe.py"
QWEN3_MODEL_PATH = "/home/gpu2-user6/Qwen3/model/qwen/Qwen3-30B-A3B"


def _resolve_modular_path(modular_path: str) -> str:
    """Resolve the actual path to modular_qwen3_moe.py with reasonable fallbacks."""
    # If user-specified path exists, use it
    if modular_path and os.path.exists(modular_path):
        return modular_path

    # Try common locations
    candidates = [
        modular_path,
        "/home/gpu2-user6/Qwen3/transformers-download/modular_qwen3_moe.py",
        "/home/gpu2-user6/Qwen3/keycode/modular_qwen3_moe.py",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c

    # Last resort: glob search inside Qwen3 repo
    try:
        import glob
        found = glob.glob("/home/gpu2-user6/Qwen3/**/modular_qwen3_moe.py", recursive=True)
        if found:
            return found[0]
    except Exception:
        pass

    raise FileNotFoundError(
        f"Qwen3 modular file not found. Tried: {candidates}. "
        f"Please pass a valid path via --modular_path."
    )


def _load_qwen3_moe_class(modular_path: str):
    """Dynamically load Qwen3MoeForCausalLM from a local modular file."""
    resolved = _resolve_modular_path(modular_path)
    spec = importlib.util.spec_from_file_location("modular_qwen3_moe", resolved)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3MoeForCausalLM


class QwenRunner:
    """Encapsulate a single forward pass and expert activation dump for Qwen3-MoE."""

    def __init__(self, modular_path: str = QWEN3_MODULAR_PATH, model_path: str = QWEN3_MODEL_PATH):
        self.modular_path = modular_path
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        # Cached results
        self.layer_token_experts = None  # dict[int -> dict[token_pos -> list[int]]]
        self.layer_expert_counts = None  # dict[int -> dict[expert_id -> int]]
        self.last_run_dir = None
        self.last_run_tag = None
        self.last_dataset_type = None
        self.last_max_length = None
        self.last_valid_tokens = None

        # Model config cache
        self.num_experts = None
        self.top_k = None

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        Qwen3MoeForCausalLM = _load_qwen3_moe_class(self.modular_path)

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = Qwen3MoeForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.model = model
        self.tokenizer = tokenizer

        # Cache config values
        self.num_experts = int(getattr(self.model.config, "num_experts", 0) or 0)
        self.top_k = int(getattr(self.model.config, "num_experts_per_tok", 0) or 0)

    def run(self, dataset_type: str, max_length: int, output_dir: str, text: str | None = None, run_tag_override: str | None = None):
        """
        Execute one forward pass, collect per-layer token→experts and activation counts, save to output_dir.
        Returns the subdirectory containing JSON files.
        """
        # Build inputs with token length exactly equal to max_length
        token_ids = ensure_exact_token_ids(dataset_type, self.tokenizer, max_length, base_text=(text or ""))
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        attention_mask = torch.ones((1, len(token_ids)), dtype=torch.long, device=self.model.device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Count valid tokens (non-PAD)
        self.last_valid_tokens = int(inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))[0].sum().item())

        with torch.no_grad():
            outputs = self.model(**inputs, output_router_logits=True)

        layer_token_experts, layer_expert_counts = build_layer_token_experts_qwen(
            outputs, inputs.get("attention_mask", None), max_length
        )

        # Cache
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

    # Accessors
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

    def save_results(self, output_dir: str, run_tag: str | None = None):
        if self.layer_token_experts is None:
            return None
        tag = run_tag or self.last_run_tag or datetime.datetime.now().strftime("manual_%Y%m%d_%H%M%S")
        run_dir = save_layer_dicts(self.layer_token_experts, output_dir, tag)
        self.last_run_dir = run_dir
        self.last_run_tag = tag
        return run_dir


def choose_text_with_min_tokens(dataset_type: str, tokenizer, min_tokens: int):
    """
    Select and concatenate sample text ensuring at least min_tokens after tokenization.
    Supported types: wikitext | winogrande | c4
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

def generate_token_chunks(dataset_type: str, tokenizer, chunk_len: int, times: int):
    """
    Build a continuous token stream, split into `times` chunks of length `chunk_len`, and decode to text.
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
    Yield text pieces sequentially for supported datasets, skipping empty examples.
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
    Concatenate tokenized base_text with dataset tokens until reaching exactly max_tokens.
    Uses add_special_tokens=False to avoid special tokens affecting length.
    """
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


def build_layer_token_experts_qwen(outputs, attention_mask, max_length: int):
    """
    Build per-layer mapping:
    - token_position(int) -> [expert_id...]
    - expert_id(int) -> activation count
    Parse from Qwen3-MoE outputs.expert_activations.
    """
    if not hasattr(outputs, "expert_activations"):
        raise RuntimeError("Model outputs missing 'expert_activations'. Ensure Qwen3 MoE routing hooks are enabled.")

    # Determine valid token positions from attention_mask
    if isinstance(attention_mask, torch.Tensor):
        am = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
        non_pad_mask = am.to(torch.bool).cpu().numpy()
    else:
        # If no attention mask, treat all positions up to max_length as valid
        non_pad_mask = np.ones((max_length,), dtype=bool)

    layer_dicts = {}
    layer_expert_counts = {}

    # outputs.expert_activations is a list per layer; each item is a dict with keys:
    # { 'layer_idx': int, 'expert_activations': { 'selected_experts': (B,S,top_k), 'routing_weights': (B,S,top_k) } }
    for idx, layer_entry in enumerate(outputs.expert_activations):
        if not isinstance(layer_entry, dict):
            # Fallback to enumerate index if dict missing
            layer_dicts[idx] = {}
            layer_expert_counts[idx] = {}
            continue

        # Use the true layer index provided by the model
        layer_idx = int(layer_entry.get("layer_idx", idx))
        expert_data = layer_entry.get("expert_activations", {})
        selected_experts = expert_data.get("selected_experts", None)
        routing_weights = expert_data.get("routing_weights", None)

        token_to_experts = defaultdict(set)
        expert_counts = defaultdict(int)

        if selected_experts is None:
            layer_dicts[layer_idx] = {}
            layer_expert_counts[layer_idx] = {}
            continue

        # Expect shape (batch, seq_len, top_k); we only process batch_size=1
        if isinstance(selected_experts, torch.Tensor):
            sel = selected_experts[0].cpu().numpy()
        else:
            sel = np.array(selected_experts)[0]

        seq_len = sel.shape[0]
        valid_len = min(seq_len, len(non_pad_mask), max_length)

        for t in range(valid_len):
            if not non_pad_mask[t]:
                continue
            idxs = sel[t]
            # idxs is array/list length top_k
            for e in set(int(i) for i in (idxs.tolist() if hasattr(idxs, "tolist") else idxs)):
                token_to_experts[t].add(e)
                expert_counts[e] += 1

        # Convert to regular structure with all positions 0..max_length-1
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
    parser = argparse.ArgumentParser(description="Run Qwen3-MoE once and dump token→experts per layer.")
    parser.add_argument("--dataset_type", type=str, default="wikitext", choices=["wikitext", "winogrande", "c4"],
                        help="Choose one dataset: wikitext | winogrande | c4")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Tokenizer truncation/padding length for a single forward run")
    parser.add_argument("--output_dir", type=str, default="./qwen_expert_token_experts",
                        help="Output folder (one JSON file per layer)")
    parser.add_argument("--modular_path", type=str, default=QWEN3_MODULAR_PATH,
                        help="Path to modular_qwen3_moe.py")
    parser.add_argument("--model_path", type=str, default=QWEN3_MODEL_PATH,
                        help="Path to Qwen3-MoE model weights")
    parser.add_argument("--times", type=int, default=1, help="Number of consecutive runs, each consumes max_length tokens")
    args = parser.parse_args()

    runner = QwenRunner(modular_path=args.modular_path, model_path=args.model_path)

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
    # Print per-layer activation counts and totals
    # all_counts = runner.get_expert_counts()
    # for layer_idx in sorted(all_counts.keys()):
    #     print(f"Layer {layer_idx} expert activation counts:")
    #     counts = all_counts[layer_idx]
    #     if not counts:
    #         print("  (no activations captured)")
    #         continue
    #     for e, c in sorted(counts.items()):
    #         print(f"  Expert {e}: {c}")
    #     total = runner.get_total_activations(layer_idx)
    #     k_est = runner.get_top_k_estimate()
    #     if k_est:
    #         expected = (runner.last_valid_tokens or 0) * k_est
    #         print(f"  Total activations: {total} (expected ~= valid_tokens * top_k)")
    #         print(f"  valid_tokens={runner.last_valid_tokens}, top_k={k_est}, expected≈{expected}")
    #     else:
    #         print("  Unable to estimate top_k (config missing).")


if __name__ == "__main__":
    main()