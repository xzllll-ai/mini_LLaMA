"""
07_test_sft.py
测试 Stage 2 SFT 模型
用与训练一致的 prompt 模板,展示古诗→白话翻译效果
"""
import argparse
import os
import torch
from transformers import LlamaForCausalLM
from sentencepiece import SentencePieceProcessor

# =============== 指定 GPU（必须放在 import torch 之前！） ===============
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# ===================================================================

# =============== Prompt 模板 (必须与 05b_sft.py 训练时一致) ===============
SYSTEM_PROMPT = "你是一位古诗翻译专家，请把用户给出的古诗词翻译成通俗易懂的现代白话文。"
INSTRUCTION_TPL = "请翻译以下古诗词：\n{original}"


def build_prompt(original):
    """构造 SFT 阶段的 prompt,匹配训练时的输入格式"""
    text = (
        f"<s>{SYSTEM_PROMPT}\n"
        f"user\n{INSTRUCTION_TPL.format(original=original)}\n"
        f"assistant\n"
    )
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="/apps/users/xzl/test/checkpoints/two_stage/stage2/final",
    )
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()

    # 加载 tokenizer
    spm_path = os.path.join(args.model_dir, "tokenizer.model")
    sp = SentencePieceProcessor(model_file=spm_path)
    eos_id = sp.eos_id()
    print(f"Tokenizer: vocab={sp.get_piece_size()}, eos_id={eos_id}")

    # 加载模型
    print(f"加载模型: {args.model_dir}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print(f"模型加载完成, 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    # 测试样本:古诗原文
    samples = [
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。",
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
        "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
        "独在异乡为异客，每逢佳节倍思亲。",
        "空山新雨后，天气晚来秋。明月松间照，清泉石上流。",
    ]

    print("=" * 70)
    print("  Stage 2 SFT 翻译测试")
    print("=" * 70)

    for i, poem in enumerate(samples, 1):
        prompt = build_prompt(poem)
        ids = sp.encode_as_ids(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p,
                pad_token_id=sp.pad_id(),
                eos_token_id=eos_id,
                repetition_penalty=args.repetition_penalty,
            )
        gen = out[0][input_ids.shape[1]:].tolist()
        if eos_id in gen:
            gen = gen[:gen.index(eos_id)]
        translation = sp.decode_ids(gen).strip()

        print(f"\n【测试 {i}】")
        print(f"古诗: {poem}")
        print(f"翻译: {translation}")
        print("-" * 70)


if __name__ == "__main__":
    main()
