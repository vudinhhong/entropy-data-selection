"""Tokenizer, prompt format, and model loading.

Every model in the paper, untrained base, teacher, student, sees the same
prompt. That is deliberate: it makes the three accuracies comparable, at the
cost of scoring the base model zero-shot, which is exactly what the denominator
of the proficiency-recovery ratio is supposed to measure.
"""
import random
import time

import numpy as np
import torch

PROMPT = "Question: {q}\nAnswer:"
TARGET = " {a}"
MAX_LENGTH = 512          # question plus answer; GSM8K and MATH both fit
DEFAULT_SEED = 1234


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def load_tokenizer(model_id, padding_side="right"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model(path, train=False, device="cuda"):
    """Load a model, merging a LoRA adapter on the way in if it finds one.

    Checkpoints hold the adapter alone, about 130 MB, instead of the merged
    model at 3.5 GB. Merging at load time is the same arithmetic as merging
    before saving, just moved to the other end.
    """
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM

    adapter_config = Path(str(path)) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        base = json.loads(adapter_config.read_text())["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        model = PeftModel.from_pretrained(model, str(path)).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    if train:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
    else:
        model.eval()
    return model


def add_lora(model, rank, alpha=None, targets=None):
    """Wrap a model in a LoRA adapter.

    Full AdamW on a 0.6B model needs about 12 GiB and fills a card on its own.
    LoRA brings that down to roughly 2.5 GiB, which is what lets several arms
    train side by side. Every arm uses the same configuration, so the
    comparison stays fair; the working point was also run with full
    fine-tuning.
    """
    from peft import LoraConfig, get_peft_model

    targets = targets or ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    return get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))


def base_model_of(model):
    """Reach the real CausalLM through any PEFT wrapper.

    Sniffing for `lm_head` is not enough: a PEFT wrapper forwards every missing
    attribute to the model it wraps, so it appears to have one too, and then
    `model.model` is the whole CausalLM instead of the transformer body --
    calling it returns an output object instead of hidden states.
    """
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    name = type(inner).__name__
    if not name.endswith("ForCausalLM"):
        raise TypeError(f"expected a causal LM, got {name}")
    return inner


def encode(tokenizer, question, answer, max_length=MAX_LENGTH):
    """Tokenize one example. Returns (input_ids, prompt_length).

    Loss is taken on the answer only, which is what prompt_length marks.
    """
    prompt = tokenizer(PROMPT.format(q=question),
                       add_special_tokens=False)["input_ids"]
    target = tokenizer(TARGET.format(a=answer),
                       add_special_tokens=False)["input_ids"]
    target = target + [tokenizer.eos_token_id]
    ids = (prompt + target)[:max_length]
    return ids, min(len(prompt), max_length)


def wait_for_vram(needed_mib, timeout=7200, poll=60, log=log):
    """Block until the visible card has `needed_mib` free.

    The machine these experiments ran on is shared, and a neighbour's job can
    grow by twenty gigabytes between one launch and the next. Starting anyway
    and failing on an allocation throws away whatever this job has already
    done, so wait instead. Returns the free memory once there is enough.

    Reads the physical card that CUDA_VISIBLE_DEVICES points at, since
    nvidia-smi indexes cards and PyTorch indexes the visible subset.
    """
    import os
    import subprocess
    import time

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = int(visible.split(",")[0]) if visible.strip() else 0

    start = time.time()
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.split()
        free = int(out[physical])
        if free >= needed_mib:
            return free
        if time.time() - start > timeout:
            raise RuntimeError(
                f"GPU {physical} has {free} MiB free, {needed_mib} needed, "
                f"waited {timeout}s")
        log(f"  GPU {physical}: {free} MiB free, need {needed_mib}; "
            f"waiting {poll}s")
        time.sleep(poll)


# Peak memory measured for a LoRA fine-tune plus its evaluation, by student
# size. Longer names come first: "Qwen2.5-1.5B" contains "1.5B", so matching
# the smaller key first would under-reserve.
VRAM_BY_SIZE = ((("4B", "4b"), 13000), (("1.8B", "1.8b"), 9500),
                (("1.7B", "1.7b"), 9000), (("1.5B", "1.5b"), 8500),
                (("0.6B", "0.6b"), 3500), (("0.5B", "0.5b"), 3200))


def vram_needed(model_id, default=9000, extra=0):
    """How much room a run on this model wants, in MiB."""
    name = str(model_id)
    for keys, mib in VRAM_BY_SIZE:
        if any(k in name for k in keys):
            return mib + extra
    return default + extra
