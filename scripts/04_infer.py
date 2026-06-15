"""
04_infer.py
加载训练好的模型，输入古诗，输出白话翻译
"""
# =============== 指定 GPU（必须放在 import torch 之前！） ===============
# 改这里切换用哪几张卡
# 推理只用 1 张卡就够。"4,5" 是把两张都暴露出来,程序只取第一张 cuda:0
# 格式: "4" 单卡; "4,5" 暴露两张但只用第一张; "" 用所有可见卡
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# ===================================================================

import argparse
import json
import os
import torch
from transformers import LlamaForCausalLM, LlamaConfig
from sentencepiece import SentencePieceProcessor

SYSTEM_PROMPT = "你是一位古诗翻译专家，请把用户给出的古诗词翻译成通俗易懂的现代白话文。"
INSTRUCTION_TPL = "请翻译以下古诗词：\n{original}"

def build_prompt(original, sp, with_response_prefix=True):
    text = (
        f"<s>{SYSTEM_PROMPT}\n"
        f"user\n{INSTRUCTION_TPL.format(original=original)}\n"
        f"assistant\n"
    )
    if with_response_prefix:
        return text
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/apps/users/xzl/test/checkpoints/poetry_llm_0_1b/final")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--poem", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # 加载 tokenizer
    spm_path = os.path.join(args.model_dir, "tokenizer.model")
    if not os.path.exists(spm_path):
        spm_path = os.path.join(args.model_dir, "spm.model")
    sp = SentencePieceProcessor(model_file=spm_path)

    # 加载模型
    print(f"加载模型: {args.model_dir}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    eos_id = sp.eos_id()
    print(f"模型加载完成，eos_id={eos_id}")

    def generate(poem, max_new=200, temp=0.7, top_p=0.9):
        prompt = build_prompt(poem, sp, with_response_prefix=True)
        ids = sp.encode_as_ids(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else 1.0,
                top_p=top_p,
                pad_token_id=sp.pad_id(),
                eos_token_id=eos_id,
                repetition_penalty=1.1,
            )
        gen = out[0][input_ids.shape[1]:].tolist()
        # 截断 eos
        if eos_id in gen:
            gen = gen[:gen.index(eos_id)]
        text = sp.decode_ids(gen).strip()
        return text

    if args.interactive:
        print("\n=== 古诗翻译 (输入 q 退出) ===\n")
        while True:
            try:
                poem = input("\n古诗> ")
            except EOFError:
                break
            if poem.strip().lower() in ("q", "quit", "exit"):
                break
            if not poem.strip():
                continue
            print("翻译> ", end="", flush=True)
            out = generate(poem, args.max_new_tokens, args.temperature, args.top_p)
            print(out)
    elif args.poem:
        print(f"\n古诗: {args.poem}\n")
        out = generate(args.poem, args.max_new_tokens, args.temperature, args.top_p)
        print(f"翻译: {out}")
    else:
        # 默认测试几首
        samples = [
            "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
            "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
            "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。",
            "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
            "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
        ]
        for poem in samples:
            print(f"\n古诗: {poem}")
            out = generate(poem, args.max_new_tokens, args.temperature, args.top_p)
            print(f"翻译: {out}")

if __name__ == "__main__":
    main()
