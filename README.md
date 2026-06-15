# mini_LLaMA

从零训练一个轻量级 LLaMA 模型，实现**古诗词 → 现代白话文**翻译。

## 项目概览

本项目是用 HuggingFace Transformers 从零实现 LLaMA 预训练 + 指令微调（SFT）的完整教学示例，涵盖了数据准备、Tokenizer 训练、分布式训练、推理测试的全链路。

| 属性 | 说明 |
|------|------|
| **模型架构** | LLaMA (decoder-only transformer, GQA, RoPE, SwiGLU) |
| **参数量** | ~0.2B (hidden=1024, layers=14, heads=16, KV heads=4) |
| **词表** | SentencePiece BPE, vocab_size=8000 |
| **训练数据** | 50+ 位著名诗人的古诗词及白话译文平行语料 |
| **训练策略** | 两阶段: 预训练 → SFT 指令微调 |
| **硬件** | 2-4 张 L40 (46G), bf16 混合精度, DDP 分布式 |

## 两阶段训练流程

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: 预训练 (from scratch)                       │
│ • 损失: 全 token LM loss (input_ids == labels)       │
│ • 数据: "原文\n译文" 拼接, 学习古诗→白话的语言模式     │
│ • Epochs: 3                                          │
│  ↓                                                    │
│ Stage 2: SFT (基于 Stage 1 权重)                     │
│ • 损失: 仅在 assistant 回答部分计算                    │
│ • 格式: System + User(原文) + Assistant(译文)         │
│ • 训练 loss mask 保证 prompt 部分不计 loss            │
│ • Epochs: 5                                           │
└─────────────────────────────────────────────────────┘
```

## 项目结构

```
mini_LLaMA/
├── scripts/
│   ├── 01_extract_subset.py     # 数据提取: 从语料库中提取古诗-白话平行对
│   ├── 02_train_tokenizer.py    # 训练 SentencePiece tokenizer (BPE, 8k vocab)
│   ├── 03_train.py              # (早期版本) 单阶段 SFT 训练 ~0.1B 模型
│   ├── 04_infer.py              # 推理脚本 (支持交互式 + 批量测试)
│   ├── 05a_pretrain.py          # Stage 1 预训练 (DDP)
│   ├── 05b_sft.py               # Stage 2 SFT 微调 (DDP)
│   ├── 06_test_pretrain.py      # Stage 1 续写测试
│   └── 07_test_sft.py           # Stage 2 翻译效果测试
├── data/
│   └── poetry_translation_pairs.jsonl   # 古诗-白话并行语料
├── tokenizer/
│   ├── poetry_spm.model         # 训练好的 SentencePiece 模型
│   └── poetry_spm.vocab         # 词表 (8000 tokens)
├── checkpoints/
│   └── two_stage/
│       ├── stage1/              # 预训练检查点
│       │   ├── step_1464/
│       │   └── final/
│       └── stage2/              # SFT 检查点
│           ├── step_1220/
│           ├── step_1500/
│           ├── step_2440/
│           └── final/
└── scripts/run_two_stage.sh     # 一键两阶段训练启动脚本
```

## 使用方法

### 1. 环境准备

```bash
# Python 依赖 (使用 conda 环境)
conda create -n mini_llama python=3.10
conda activate mini_llama
pip install torch transformers sentencepiece accelerate
```

### 2. 数据提取

从 [Mobvoi 古文语料库](https://github.com/mobvoi/seq-monkey-data) 中提取 50 位著名诗人的古诗-白话平行对:

```bash
python scripts/01_extract_subset.py
```

需要提前下载语料 `mobvoi_seq_monkey_classical_chs_open_corpus.tar.bz2` 放在项目根目录。

### 3. 训练 Tokenizer

```bash
python scripts/02_train_tokenizer.py
```

训练一个 vocab_size=8000 的 SentencePiece BPE tokenizer，覆盖古汉语常见字词。

### 4. 两阶段训练

**一键启动（推荐）:**

```bash
# 两阶段全程
bash scripts/run_two_stage.sh both

# 或分别执行
bash scripts/run_two_stage.sh stage1   # 只跑预训练
bash scripts/run_two_stage.sh stage2   # 只跑 SFT (需要 stage1 已有权重)
```

**自定义 GPU 数量:**

```bash
GPU_COUNT=4 bash scripts/run_two_stage.sh both    # 4 卡训练
CUDA_VISIBLE_DEVICES=5,6 GPU_COUNT=2 bash scripts/run_two_stage.sh stage1  # 指定 GPU
```

**手动启动（如需调参）:**

```bash
# Stage 1 预训练
torchrun --standalone --nproc_per_node=2 scripts/05a_pretrain.py

# Stage 2 SFT
torchrun --standalone --nproc_per_node=4 scripts/05b_sft.py
```

### 5. 推理测试

```bash
# 交互式翻译
python scripts/04_infer.py --interactive

# 翻译单首古诗
python scripts/04_infer.py --poem "床前明月光，疑是地上霜。举头望明月，低头思故乡。"

# 指定模型路径
python scripts/04_infer.py --model_dir checkpoints/two_stage/stage2/final --interactive

# 测试 Stage 1 续写效果
python scripts/06_test_pretrain.py --model_dir checkpoints/two_stage/stage1/final

# 测试 Stage 2 SFT 翻译效果
python scripts/07_test_sft.py --model_dir checkpoints/two_stage/stage2/final
```

## 推理示例

输入:

```
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
```

输出（Stage 2 SFT 模型）:

```
明亮的月光洒在床前，好像地上铺了一层白霜。
我抬起头望着空中的明月，低下头不禁思念起遥远的故乡。
```

## 模型架构详解

| 参数 | Stage 1 (预训练) | Stage 2 (SFT) | 早期单阶段 |
|------|------------------|---------------|-----------|
| hidden_size | 1024 | 1024 | 768 |
| intermediate_size | 2816 | 2816 | 2048 |
| num_hidden_layers | 14 | 14 | 12 |
| num_attention_heads | 16 | 16 | 12 |
| num_key_value_heads | 4 (GQA) | 4 (GQA) | 4 (GQA) |
| max_seq_len | 768 | 768 | 768 |
| vocab_size | 8000 | 8000 | 8000 |
| tie_word_embeddings | ✅ | ✅ | ✅ |
| 参数量 | ~0.2B | ~0.2B | ~0.1B |

## 数据说明

- **来源**: [Mobvoi Seq Monkey 古典中文开放语料](https://github.com/mobvoi/seq-monkey-data)（古诗+白话文对照）
- **诗人范围**: 李白、杜甫、白居易、苏轼、李清照、辛弃疾等 50 余位历代著名诗人
- **数据量**: 数千首古诗-白话平行对
- **格式**: JSONL，每行含 author（作者）、title（标题）、original（原文）、translation（白话翻译）

## 技术要点

- **GQA (Grouped Query Attention)**: 4 KV heads 共享，显著降低 KV cache 占用
- **Loss Mask**: SFT 阶段仅在 `assistant\n` 之后的部分计算损失，prompt 部分被 mask 掉
- **bf16 混合精度**: 单卡 L40 (46G) 可训练 0.2B 模型
- **DDP 分布式**: 使用 `torchrun` + PyTorch DDP 支持多卡训练
- **梯度检查点**: 开启 `gradient_checkpointing` 以节省显存
- **Weight Tying**: 共享 token embedding 与 lm_head 权重，参数更省

## License

MIT
