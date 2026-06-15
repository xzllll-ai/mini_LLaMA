"""
02_train_tokenizer.py
训练古诗专用 SentencePiece tokenize（BPE, vocab=8000）
用原文+译文一起训练，让它能高效编码古汉语字
"""
import json
import os
import sentencepiece as spm
from pathlib import Path

DATA = "/apps/users/xzl/test/data/poetry_translation_pairs.jsonl"
OUT_DIR = "/apps/users/xzl/test/tokenizer"
CORPUS_TXT = os.path.join(OUT_DIR, "corpus.txt")
VOCAB_SIZE = 8000
MODEL_PREFIX = os.path.join(OUT_DIR, "poetry_spm")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 提取所有文本到 txt，每行一句
    print("提取文本...")
    lines = []
    with open(DATA) as f:
        for line in f:
            d = json.loads(line)
            lines.append(d["original"].strip())
            lines.append(d["translation"].strip())
            # 再加一行"诗题"作为元数据
            if d.get("title"):
                lines.append(d["title"].strip())
            if d.get("author"):
                lines.append(d["author"].strip())

    # 去重 + 过滤过短
    lines = [l for l in set(lines) if len(l) >= 2]
    print(f"去重后 {len(lines)} 行")

    with open(CORPUS_TXT, "w", encoding="utf-8") as out:
        for l in lines:
            out.write(l + "\n")

    # 2. 训练 SentencePiece
    print(f"训练 SentencePiece (vocab={VOCAB_SIZE})...")
    spm.SentencePieceTrainer.train(
        input=CORPUS_TXT,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,   # 中文几乎全覆盖
        num_threads=8,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # 关键: 控制单字 token
        max_sentence_length=2048,
        shuffle_input_sentence=True,
        byte_fallback=True,           # 未知字回退到 byte
        # 标注控制: 不需要
    )
    print(f"训练完成: {MODEL_PREFIX}.model / .vocab")

    # 3. 验证
    sp = spm.SentencePieceProcessor(model_file=MODEL_PREFIX + ".model")
    test = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    tokens = sp.encode_as_pieces(test)
    ids = sp.encode_as_ids(test)
    print(f"\n样本: {test}")
    print(f"Tokens ({len(tokens)}): {tokens[:20]}...")
    print(f"IDs ({len(ids)}): {ids[:20]}...")
    print(f"Vocab size: {sp.get_piece_size()}")

    # 4. 显示前 30 个 token
    print(f"\n前 30 个 tokens:")
    for i in range(min(30, sp.get_piece_size())):
        print(f"  {i:4d}: {sp.id_to_piece(i)!r}")

if __name__ == "__main__":
    main()
