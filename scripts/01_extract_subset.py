"""
01_extract_subset.py
从 mobvoi_seq_monkey_classical_chs_open_corpu 中提取 ~50 位著名诗人的 (古诗,白话) 平行对
"""
import json
import os
import tarfile
import sys

# 50 位著名诗人，覆盖唐宋元明清等朝代的核心大家
FAMOUS_POETS = [
    # 唐
    ("唐/李白.json", "李白"),
    ("唐/杜甫.json", "杜甫"),
    ("唐/白居易.json", "白居易"),
    ("唐/王维.json", "王维"),
    ("唐/孟浩然.json", "孟浩然"),
    ("唐/李商隐.json", "李商隐"),
    ("唐/杜牧.json", "杜牧"),
    ("唐/王昌龄.json", "王昌龄"),
    ("唐/刘禹锡.json", "刘禹锡"),
    ("唐/李贺.json", "李贺"),
    ("唐/岑参.json", "岑参"),
    ("唐/高适.json", "高适"),
    ("唐/王之涣.json", "王之涣"),
    ("唐/温庭筠.json", "温庭筠"),
    ("唐/韦应物.json", "韦应物"),
    ("唐/韩愈.json", "韩愈"),
    ("唐/柳宗元.json", "柳宗元"),
    ("唐/骆宾王.json", "骆宾王"),
    # 宋
    ("宋/苏轼.json", "苏轼"),
    ("宋/李清照.json", "李清照"),
    ("宋/辛弃疾.json", "辛弃疾"),
    ("宋/陆游.json", "陆游"),
    ("宋/王安石.json", "王安石"),
    ("宋/欧阳修.json", "欧阳修"),
    ("宋/柳永.json", "柳永"),
    ("宋/姜夔.json", "姜夔"),
    ("宋/晏殊.json", "晏殊"),
    ("宋/黄庭坚.json", "黄庭坚"),
    ("宋/秦观.json", "秦观"),
    ("宋/范仲淹.json", "范仲淹"),
    ("宋/文天祥.json", "文天祥"),
    ("宋/杨万里.json", "杨万里"),
    # 元
    ("元/元好问.json", "元好问"),
    ("元/马致远.json", "马致远"),
    ("元/张可久.json", "张可久"),
    ("元/白朴.json", "白朴"),
    # 明
    ("明/高启.json", "高启"),
    ("明/于谦.json", "于谦"),
    ("明/文征明.json", "文征明"),
    ("明/杨慎.json", "杨慎"),
    ("明/唐寅.json", "唐寅"),
    # 清
    ("清/纳兰性德.json", "纳兰性德"),
    ("清/龚自珍.json", "龚自珍"),
    ("清/郑板桥.json", "郑板桥"),
    ("清/袁枚.json", "袁枚"),
    ("清/查慎行.json", "查慎行"),
    # 其他
    ("汉/乐府.json", "乐府"),
    ("汉/曹操.json", "曹操"),
    ("魏晋/陶渊明.json", "陶渊明"),
    ("魏晋/谢灵运.json", "谢灵运"),
    ("南北朝/庾信.json", "庾信"),
    ("南北朝/鲍照.json", "鲍照"),
]

CORPUS_TAR = "/apps/users/xzl/test/mobvoi_seq_monkey_classical_chs_open_corpus.tar.bz2"
OUTPUT_JSONL = "/apps/users/xzl/test/data/poetry_translation_pairs.jsonl"

def main():
    pairs = []
    seen_titles = set()  # 去重
    skipped_no_translate = 0
    skipped_no_paragraph = 0

    print(f"开始提取 {len(FAMOUS_POETS)} 位诗人...")

    with tarfile.open(CORPUS_TAR, "r:bz2") as tar:
        for member, poet in FAMOUS_POETS:
            try:
                f = tar.extractfile(f"./{member}")
                if f is None:
                    print(f"  ⚠️  {member} 不存在")
                    continue
                data = json.load(f)
                added = 0
                for poem in data:
                    para = poem.get("paragraph")
                    trans = poem.get("translate")
                    if not para or not trans:
                        if not para:
                            skipped_no_paragraph += 1
                        else:
                            skipped_no_translate += 1
                        continue

                    # 去重：以作者+标题为 key
                    title = poem.get("title", "").strip()
                    key = f"{poet}::{title}"
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)

                    para_text = "".join(para).strip()
                    trans_text = "".join(trans).strip()

                    # 过滤过短或过长的
                    if len(para_text) < 8 or len(para_text) > 300:
                        continue
                    if len(trans_text) < 10 or len(trans_text) > 600:
                        continue

                    pairs.append({
                        "author": poet,
                        "title": title,
                        "type": poem.get("type", ""),
                        "original": para_text,
                        "translation": trans_text,
                    })
                    added += 1
                print(f"  ✓ {poet}: 提取 {added} 首诗")
            except Exception as e:
                print(f"  ✗ {member}: {e}")

    # 写入 jsonl
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for p in pairs:
            out.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 统计
    print(f"\n========== 提取完成 ==========")
    print(f"总样本数: {len(pairs)}")
    print(f"输出文件: {OUTPUT_JSONL}")
    print(f"文件大小: {os.path.getsize(OUTPUT_JSONL) / 1024 / 1024:.2f} MB")
    print(f"无翻译跳过: {skipped_no_translate}")
    print(f"无原文跳过: {skipped_no_paragraph}")

    # 字符统计
    total_orig = sum(len(p["original"]) for p in pairs)
    total_trans = sum(len(p["translation"]) for p in pairs)
    print(f"原文总字符: {total_orig:,}")
    print(f"译文总字符: {total_trans:,}")
    print(f"原文+译文总字符: {total_orig + total_trans:,}")
    print(f"按 1 token ≈ 1.5 汉字估, 约 {(total_orig + total_trans) / 1.5 / 1e6:.1f}M tokens")

if __name__ == "__main__":
    main()
