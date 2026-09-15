#!/usr/bin/env python3
"""
Step 17.1 - RAG 入库格式导出
============================

从正式 KB v3 (645 条) 导出 RAG-ready chunks:

    1 条 KB = 1 chunk (全部为短条目, 不切分)
    embedding 文本 = "问题：{q}\n答案：{a}"
    metadata 保留权威级 / 模块 / 时效字段

输出 (不修改任何冻结输入):

    output/rag_chunks_v1.jsonl
    output/rag_chunks_v1.xlsx (人工检查用)

守卫:

- chunk 数 = KB v3 条数 (645)
- id 唯一且与 kb_id 一致
- embedding 文本非空
- 答案非空
"""

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

KB_V3_FILE = OUTPUT_DIR / "kb_entries_official_v3.jsonl"

CHUNKS_JSONL = OUTPUT_DIR / "rag_chunks_v1.jsonl"
CHUNKS_XLSX = OUTPUT_DIR / "rag_chunks_v1.xlsx"


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def build_chunk(record):

    payload = record["payload"]

    part = record["part"]

    if part == "singleton":
        crm_module = payload.get("crm_module") or ""
        crm_feature = payload.get("crm_feature") or ""
        problem_type = payload.get("problem_type") or ""
        temporal_date = payload.get("issue_date") or ""
        resolution = payload.get("resolution") or ""
        knowledge_value = (
            payload.get("knowledge_value") or ""
        )
    else:
        crm_module = payload.get("crm_module") or ""
        crm_feature = payload.get("crm_feature") or ""
        problem_type = payload.get("problem_type") or ""
        temporal_date = ""
        resolution = ""
        knowledge_value = (
            payload.get("knowledge_confidence") or ""
        )

    question = record["question"] or ""
    answer = record["answer"] or ""

    return {
        "chunk_id": record["kb_id"],
        "source": "kb_entries_official_v3",
        "text": f"问题：{question}\n答案：{answer}",
        "question": question,
        "answer": answer,
        "metadata": {
            "part": part,
            "cluster_id": record["cluster_id"],
            "provenance": record["provenance"],
            "crm_module": crm_module,
            "crm_feature": crm_feature,
            "problem_type": problem_type,
            "resolution": resolution,
            "knowledge_value": knowledge_value,
            "temporal_date": temporal_date,
        },
    }


def main():

    kb = load_jsonl(KB_V3_FILE)

    errors = []
    chunks = []

    for record in kb:
        chunk = build_chunk(record)

        if not chunk["text"].strip():
            errors.append(
                f"{chunk['chunk_id']} text 为空"
            )

        if not chunk["answer"].strip():
            errors.append(
                f"{chunk['chunk_id']} answer 为空"
            )

        chunks.append(chunk)

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        errors.append("chunk_id 重复")

    if len(chunks) != 645:
        errors.append(
            f"chunk 数 {len(chunks)} != 645"
        )

    if errors:
        for error in errors:
            print(f"守卫失败: {error}")

        raise SystemExit(1)

    with open(
        CHUNKS_JSONL, "w", encoding="utf-8"
    ) as output:
        for chunk in chunks:
            output.write(
                json.dumps(
                    chunk, ensure_ascii=False
                )
                + "\n"
            )

    flat_rows = []

    for chunk in chunks:
        flat_rows.append({
            "chunk_id": chunk["chunk_id"],
            "part": chunk["metadata"]["part"],
            "question": chunk["question"],
            "answer": chunk["answer"],
            "crm_module": chunk["metadata"][
                "crm_module"
            ],
            "problem_type": chunk["metadata"][
                "problem_type"
            ],
            "text_chars": len(chunk["text"]),
        })

    text_lengths = [
        len(chunk["text"]) for chunk in chunks
    ]

    summary = {
        "chunks": len(chunks),
        "source": "kb_entries_official_v3",
        "text_min_chars": min(text_lengths),
        "text_median_chars": sorted(
            text_lengths
        )[len(text_lengths) // 2],
        "text_max_chars": max(text_lengths),
        "embedding_text_format": (
            "问题：{q}\n答案：{a}"
        ),
        "parts": pd.Series(
            [
                chunk["metadata"]["part"]
                for chunk in chunks
            ]
        ).value_counts().to_dict(),
    }

    with pd.ExcelWriter(
        CHUNKS_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame(flat_rows).to_excel(
            writer, sheet_name="chunks", index=False
        )

    print("=" * 60)
    print("RAG chunks 导出完成（Step 17.1）")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    print(f"输出: {CHUNKS_JSONL}")
    print(f"输出: {CHUNKS_XLSX}")


if __name__ == "__main__":
    main()
