#!/usr/bin/env python3
"""
Step 17.4 - Dify / FastGPT 导入包
=================================

从 rag_chunks_v1.jsonl (645 chunks) 生成
平台通用 QA 导入文件 (CSV, question,answer 表头):

- import_qa_full.csv       全量 645
- import_qa_core.csv       核心 76
  (safe + regenerated + multicause + mixed)
- import_qa_singleton.csv  singleton 569

说明:

- Dify: 知识库 -> 导入 -> CSV(问答模式),
  识别 question/answer 两列表头
- FastGPT: 知识库 -> 导入 -> CSV/表格模式,
  导入时选择 question 列为索引/index,
  answer 列为答案/content
- 两平台导入时都会用平台配置的
  embedding 模型重建索引,
  问答模式下问题面权重更高,
  适配客服 query 形态
- 额外列 (part/kb_id/crm_module) 表格模式下
  可选导入为元数据, 问答模式下会被忽略

输出 (新目录, 不覆盖任何既有导出):

    output/platform_import/*.csv
"""

import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CHUNKS_FILE = OUTPUT_DIR / "rag_chunks_v1.jsonl"
IMPORT_DIR = OUTPUT_DIR / "platform_import"

CORE_PARTS = {
    "safe",
    "regenerated",
    "multicause",
    "mixed",
}


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def write_csv(path: Path, rows):

    with open(
        path, "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "question",
                "answer",
                "part",
                "kb_id",
                "cluster_id",
                "crm_module",
                "problem_type",
                "temporal_date",
            ]
        )

        for chunk in rows:
            metadata = chunk["metadata"]

            writer.writerow(
                [
                    chunk["question"],
                    chunk["answer"],
                    metadata["part"],
                    chunk["chunk_id"],
                    metadata["cluster_id"],
                    metadata["crm_module"],
                    metadata["problem_type"],
                    metadata["temporal_date"],
                ]
            )


def main():

    chunks = load_jsonl(CHUNKS_FILE)

    core = [
        chunk for chunk in chunks
        if chunk["metadata"]["part"] in CORE_PARTS
    ]
    singleton = [
        chunk for chunk in chunks
        if chunk["metadata"]["part"]
        == "singleton"
    ]

    if len(core) + len(singleton) != len(chunks):
        raise RuntimeError("part 拆分不完整")

    if len(core) != 76 or len(singleton) != 569:
        raise RuntimeError(
            f"数量异常: core={len(core)} "
            f"singleton={len(singleton)}"
        )

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("import_qa_full.csv", chunks),
        ("import_qa_core.csv", core),
        ("import_qa_singleton.csv", singleton),
    ]

    for filename, rows in targets:
        path = IMPORT_DIR / filename
        write_csv(path, rows)
        print(f"输出: {path} ({len(rows)} 行)")

    print()
    print("Dify 导入:")
    print("  知识库 -> 创建 -> 导入 CSV")
    print("  分段方式选『问答模式』")
    print("  (识别 question/answer 表头,")
    print("   其余列忽略)")
    print()
    print("FastGPT 导入:")
    print("  知识库 -> 创建 -> CSV/表格导入")
    print("  训练模式选『问答对』或表格模式,")
    print("  question 列映射为索引,")
    print("  answer 列映射为答案")
    print()
    print("建议检索参数:")
    print("  top_k: 3-5")
    print("  相似度下限: 0.40 起步")
    print("  (平台相似度归一方式不同,")
    print("   按实测调整; 库内最近邻上限")
    print("   0.892, 低于该值均为不同条目)")


if __name__ == "__main__":
    main()
