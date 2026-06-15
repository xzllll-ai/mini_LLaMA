"""
06_test_pretrain.py
测试 Stage 1 预训练模型
输入古诗原文,看模型能否"续写"出白话翻译
(不是 SFT,所以不期待严格的翻译格式,但可以看是否学到了语言模式)
"""
import argparse
import os
import torch
from transformers import LlamaForCausalLM
from sentencepiece import SentencePieceProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="/apps/users/xzl/test/checkpoints/two_stage/stage1/final",
    )
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # 加载 tokenizer
    spm_path = os.path.join(args.model_dir, "tokenizer.model")
    sp = SentencePieceProcessor(model_file=spm_path)
    eos_id = sp.eos_id()

    # 加载模型
    print(f"加载模型: {args.model_dir}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    # 测试样本:古诗原文(训练时格式: original \n translation)
    samples = [
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。",
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
        "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
    ]

    print("\n" + "=" * 60)
    print("  Stage 1 续写测试 (输入古诗,看模型能否续出白话)")
    print("=" * 60)

    for poem in samples:
        # 用 \n 触发续写(训练时 original 后面接 \n 然后是 translation)
        prompt = f"{poem}\n"
        ids = sp.encode_as_ids(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=sp.pad_id(),
                eos_token_id=eos_id,
                repetition_penalty=1.1,
            )
        gen = out[0][input_ids.shape[1]:].tolist()
        if eos_id in gen:
            gen = gen[:gen.index(eos_id)]
        # 取第一句(到第一个句号)作为预览
        text = sp.decode_ids(gen).strip()
        preview = text.split("。")[0] + ("。" if "。" in text else "")

        print(f"\n【古诗】{poem}")
        print(f"【续写】{preview}")
        print(f"【完整】{text[:200]}{'...' if len(text) > 200 else ''}")
        print("-" * 60)


if __name__ == "__main__":
    main()
