"""
03_train.py
从零训练一个 ~0.1B 的古诗翻译 LLaMA 模型
单卡 L40 (46G) 训练 bf16，单阶段 SFT
"""
import json
import os
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from sentencepiece import SentencePieceProcessor
from accelerate import Accelerator
import time
import shutil

# ============ 配置 ============
DATA_JSONL = "/apps/users/xzl/test/data/poetry_translation_pairs.jsonl"
SPM_MODEL = "/apps/users/xzl/test/tokenizer/poetry_spm.model"
OUTPUT_DIR = "/apps/users/xzl/test/checkpoints/poetry_llm_0_1b"

# 模型配置 (~0.1B)
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 2048
NUM_LAYERS = 12
NUM_HEADS = 12
NUM_KV_HEADS = 4           # GQA，省显存
MAX_SEQ_LEN = 768          # 古诗短
VOCAB_SIZE = 8000
DROPOUT = 0.0
TIE_WORD_EMBEDDINGS = True  # 小模型可以共享

# 训练配置
BATCH_SIZE = 32            # 单卡 L40 46G 跑 0.1B 绰绰有余
GRAD_ACCUM = 2             # 等效 batch=64
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
NUM_EPOCHS = 5
LOG_EVERY = 50
SAVE_EVERY = 2000
SEED = 42
MAX_TRAIN_SAMPLES = None   # None = 用全部

# Prompt 模板
SYSTEM_PROMPT = "你是一位古诗翻译专家，请把用户给出的古诗词翻译成通俗易懂的现代白话文。"
INSTRUCTION = "请翻译以下古诗词：\n{original}"

# ============ 数据集 ============
class PoetryDataset(Dataset):
    def __init__(self, data_path, sp, max_seq_len, max_samples=None):
        self.sp = sp
        self.max_seq_len = max_seq_len
        self.samples = []
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                d = json.loads(line)
                # 拼接格式: bos + system/user/assistant + eos
                text = (
                    f"{SYSTEM_PROMPT}\n"
                    f"user\n{INSTRUCTION.format(original=d['original'])}\n"
                    f"assistant\n{d['translation']}"
                )
                self.samples.append(text)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        body_ids = self.sp.encode_as_ids(text)
        ids = [self.sp.bos_id()] + body_ids[:self.max_seq_len - 2] + [self.sp.eos_id()]
        # 找到 assistant 位置，做 loss 掩码
        # 只在 assistant 回答部分计 loss。
        # 用 \nassistant\n 作为分隔符
        asst_marker = self.sp.encode_as_ids("\nassistant\n")
        asst_pos = self._find_subseq(ids, asst_marker)
        labels = [-100] * len(ids)
        if asst_pos is not None:
            start = asst_pos + len(asst_marker)
            for i in range(start, len(ids)):
                labels[i] = ids[i]
        else:
            # 极端截断时不要把 prompt 当成答案训练；保留 eos 避免全 -100 loss。
            labels[-1] = ids[-1]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _find_subseq(seq, sub):
        n, m = len(seq), len(sub)
        for i in range(n - m + 1):
            if seq[i:i+m] == sub:
                return i
        return None

def collate_fn(batch, pad_id=0):
    ids_list, labels_list = zip(*batch)
    max_len = max(len(x) for x in ids_list)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(ids_list, labels_list)):
        input_ids[i, :len(ids)] = ids
        labels[i, :len(lbl)] = lbl
        attn_mask[i, :len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

# ============ 主流程 ============
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 tokenizer
    sp = SentencePieceProcessor(model_file=SPM_MODEL)
    pad_id = sp.pad_id()
    print(f"Tokenizer vocab: {sp.get_piece_size()}, pad_id={pad_id}")

    # 数据
    print("加载数据...")
    dataset = PoetryDataset(DATA_JSONL, sp, MAX_SEQ_LEN, MAX_TRAIN_SAMPLES)
    print(f"训练样本数: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        collate_fn=lambda b: collate_fn(b, pad_id), pin_memory=True, drop_last=True,
    )

    # 配置
    config = LlamaConfig(
        vocab_size=sp.get_piece_size(),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_SEQ_LEN,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=TIE_WORD_EMBEDDINGS,
        dropout=DROPOUT,
        initializer_range=0.02,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    model = LlamaForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.1f}M ({n_params/1e9:.3f}B)")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=WEIGHT_DECAY,
    )
    total_steps = math.ceil(len(loader) / GRAD_ACCUM) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    print(f"总步数: {total_steps} ({NUM_EPOCHS} epochs × {math.ceil(len(loader)/GRAD_ACCUM)} steps)")

    # 设备
    device = torch.device("cuda")
    model.to(device, dtype=torch.bfloat16)

    # 训练循环
    model.train()
    step = 0
    running_loss = 0.0
    running_batches = 0
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            window_start = (batch_idx // GRAD_ACCUM) * GRAD_ACCUM
            window_end = min(window_start + GRAD_ACCUM, len(loader))
            accum_steps = window_end - window_start
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                raw_loss = out.loss
                loss = raw_loss / accum_steps
            loss.backward()

            running_loss += raw_loss.item()
            running_batches += 1

            if batch_idx + 1 == window_end:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % LOG_EVERY == 0:
                    avg_loss = running_loss / max(1, running_batches)
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    sps = (step * BATCH_SIZE * GRAD_ACCUM) / elapsed
                    eta = (total_steps - step) / max(1e-6, sps / (BATCH_SIZE * GRAD_ACCUM))
                    print(
                        f"epoch {epoch+1}/{NUM_EPOCHS}  step {step}/{total_steps}  "
                        f"loss {avg_loss:.4f}  lr {lr_now:.2e}  "
                        f"speed {sps:.1f} samples/s  ETA {eta/60:.1f}min"
                    )
                    running_loss = 0.0
                    running_batches = 0

                if step % SAVE_EVERY == 0 or step == total_steps:
                    ckpt = os.path.join(OUTPUT_DIR, f"step_{step}")
                    model.save_pretrained(ckpt, safe_serialization=True)
                    shutil.copy(SPM_MODEL, os.path.join(ckpt, "tokenizer.model"))  # 备份 tokenizer
                    print(f"  saved → {ckpt}")

    # 最终保存
    final = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final, safe_serialization=True)
    shutil.copy(SPM_MODEL, os.path.join(final, "tokenizer.model"))
    print(f"\n训练完成，最终模型: {final}")

if __name__ == "__main__":
    main()
