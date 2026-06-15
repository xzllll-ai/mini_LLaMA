"""
05a_pretrain.py
Stage 1 预训练：从零训练古诗+白话 LM
torchrun --standalone --nproc_per_node=2 scripts/05a_pretrain.py
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from sentencepiece import SentencePieceProcessor

# =============== 模型规格 (~0.2B) ===============
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 2816
NUM_LAYERS = 14
NUM_HEADS = 16
NUM_KV_HEADS = 4
MAX_SEQ_LEN = 768
TIE_WORD_EMBEDDINGS = True
DROPOUT = 0.0

# =============== 训练超参 ===============
SEED = 42
BATCH_SIZE = 32       # per-GPU
GRAD_ACCUM = 1
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
GRAD_CLIP = 1.0
USE_GRAD_CHECKPOINT = True

EPOCHS = 3
LOG_EVERY = 50
SAVE_EVERY = 1500

OUTPUT_DIR = "/apps/users/xzl/mini_LLaMA/checkpoints/two_stage/stage1"


# =============== 分布式辅助 ===============
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def log(rank, msg):
    if is_main(rank):
        print(msg, flush=True)


# =============== 数据集 ===============
class PretrainDataset(Dataset):
    """
    预训练数据：古诗原文 + \n + 白话译文
    损失 = 全 token LM 损失 (input_ids == labels)

    注意：不再在 __init__ 中 shuffle,留给 DistributedSampler 处理
    (DistributedSampler 通过 set_epoch(epoch) 同步 shuffle,各 rank 看到一致的顺序)
    """
    def __init__(self, data_path, sp, max_seq_len):
        self.sp = sp
        self.max_seq_len = max_seq_len
        self.samples = []
        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                text = f"{d['original']}\n{d['translation']}"
                self.samples.append(text)
        # 不 shuffle! 保持各 rank 数据顺序一致

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        body_ids = self.sp.encode_as_ids(self.samples[idx])
        ids = [self.sp.bos_id()] + body_ids[:self.max_seq_len - 2] + [self.sp.eos_id()]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ids, dtype=torch.long)


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


# =============== 模型 ===============
def build_model(vocab_size):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_SEQ_LEN,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=TIE_WORD_EMBEDDINGS,
        dropout=DROPOUT,
        initializer_range=0.02,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    return LlamaForCausalLM(config)


# =============== 训练主循环 ===============
def train(rank, world_size, local_rank, args):
    # 所有 rank 使用相同 seed,保证:
    # 1) 模型初始权重一致 (DDP 不会自动 broadcast init params)
    # 2) DistributedSampler 的 shuffle 在各 rank 间同步
    random.seed(SEED)
    torch.manual_seed(SEED)

    log(rank, f"\n{'=' * 60}")
    log(rank, f"Stage 1 预训练 | world_size={world_size} | epochs={EPOCHS} | lr={LR}")
    log(rank, f"{'=' * 60}")

    sp = SentencePieceProcessor(model_file=args.spm_model)
    pad_id = sp.pad_id()
    log(rank, f"Tokenizer: vocab={sp.get_piece_size()} pad_id={pad_id}")

    # 数据
    dataset = PretrainDataset(args.data_jsonl, sp, MAX_SEQ_LEN)
    log(rank, f"训练样本数: {len(dataset)}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True, drop_last=True,
    )

    # 模型 (从零)
    model = build_model(sp.get_piece_size())
    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f"模型参数量: {n_params / 1e6:.1f}M ({n_params / 1e9:.3f}B)")

    if USE_GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model = model.to(local_rank, dtype=torch.bfloat16)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if hasattr(model, "module") else model

    # 优化器 + 调度
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    log(rank, f"每 epoch 步数: {steps_per_epoch} | 总步数: {total_steps}")

    # 训练
    model.train()
    step = 0
    running_loss = 0.0
    running_batches = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(local_rank, non_blocking=True) for k, v in batch.items()}
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if is_main(rank) and step % LOG_EVERY == 0:
                    avg_loss = running_loss / max(1, running_batches)
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    sps = (step * BATCH_SIZE * GRAD_ACCUM * world_size) / elapsed
                    eta_sec = (total_steps - step) * (elapsed / step)
                    log(rank,
                        f"epoch {epoch + 1}/{EPOCHS}  step {step}/{total_steps}  "
                        f"loss {avg_loss:.4f}  lr {lr_now:.2e}  "
                        f"speed {sps:.0f} samples/s  ETA {eta_sec / 60:.1f}min"
                    )
                    running_loss = 0.0
                    running_batches = 0

                if is_main(rank) and (step % SAVE_EVERY == 0 or step == total_steps):
                    ckpt = os.path.join(OUTPUT_DIR, f"step_{step}")
                    os.makedirs(ckpt, exist_ok=True)
                    raw_model.save_pretrained(ckpt, safe_serialization=True)
                    shutil.copy(args.spm_model, os.path.join(ckpt, "tokenizer.model"))
                    log(rank, f"  saved → {ckpt}")

    # Final save
    if is_main(rank):
        final = os.path.join(OUTPUT_DIR, "final")
        os.makedirs(final, exist_ok=True)
        raw_model.save_pretrained(final, safe_serialization=True)
        shutil.copy(args.spm_model, os.path.join(final, "tokenizer.model"))
        log(rank, f"\nStage 1 训练完成 → {final}")

    if world_size > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_jsonl", default="/apps/users/xzl/mini_LLaMA/data/poetry_translation_pairs.jsonl")
    parser.add_argument("--spm_model", default="/apps/users/xzl/mini_LLaMA/tokenizer/poetry_spm.model")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    train(rank, world_size, local_rank, args)
    cleanup_distributed()


if __name__ == "__main__":
    main()
