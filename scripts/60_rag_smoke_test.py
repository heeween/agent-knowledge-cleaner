#!/usr/bin/env python3
"""
Step 17.3 - RAG 检索冒烟测试
============================

纯 numpy 余弦检索 (无 API):

- 加载 rag_chunks_v1.jsonl + rag_chunk_embeddings_v1.jsonl
- 对查询文本返回 top-k chunk (问题 + 答案 + 元数据)

两种用法:

1. 内置样例集 (客服视角问题, 不在 KB 中的改写):

    .venv/bin/python scripts/60_rag_smoke_test.py

2. 自定义查询 (可多次传入):

    .venv/bin/python scripts/60_rag_smoke_test.py \\
        --query "怎么开通短信" \\
        --query "密码忘记了怎么办"

需要先完成 59 的 embedding。
"""

import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CHUNKS_FILE = OUTPUT_DIR / "rag_chunks_v1.jsonl"
EMBEDDINGS_FILE = (
    OUTPUT_DIR / "rag_chunk_embeddings_v1.jsonl"
)

TOP_K = 5

SAMPLE_QUERIES = [
    "密码忘记了登不上怎么办",
    "怎么开通短信功能",
    "客户离职了账号怎么处理",
    "为什么已经回访的客户还在回访列表里",
    "保养提醒一直弹出来怎么关掉",
    "导入的客户数据查不到",
    "企业微信和系统怎么打通",
    "话机打不出电话",
]


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def search(
    query_vector, matrix, chunks, top_k
):

    similarities = matrix @ query_vector

    top_indices = np.argsort(similarities)[
        ::-1
    ][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk": chunks[int(index)],
            "similarity": float(
                similarities[index]
            ),
        })

    return results


def main():

    queries = []

    if "--query" in sys.argv:
        args = sys.argv[1:]

        for i, arg in enumerate(args):
            if arg == "--query":
                queries.append(
                    args[i + 1]
                )
    else:
        queries = SAMPLE_QUERIES

    chunks = load_jsonl(CHUNKS_FILE)
    embeddings = load_jsonl(EMBEDDINGS_FILE)

    vector_by_id = {
        record["chunk_id"]: record["embedding"]
        for record in embeddings
    }

    missing = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["chunk_id"] not in vector_by_id
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} 个 chunk 缺向量, "
            f"先运行 59。示例: {missing[:3]}"
        )

    matrix = np.array(
        [
            vector_by_id[chunk["chunk_id"]]
            for chunk in chunks
        ],
        dtype=np.float64,
    )

    norms = np.linalg.norm(
        matrix, axis=1, keepdims=True
    )

    matrix = matrix / norms

    print(
        f"索引: {len(chunks)} chunks | "
        f"查询: {len(queries)} 条 | "
        f"top_k={TOP_K}"
    )
    print("=" * 70)

    for query in queries:
        print(f"\n查询: {query}")

        response = None

        try:
            from openai import OpenAI
            import os

            client = OpenAI(
                base_url=(
                    "https://open.bigmodel.cn"
                    "/api/paas/v4"
                ),
                api_key=os.environ.get(
                    "OPENAI_API_KEY"
                ),
                timeout=60,
                max_retries=0,
            )

            response = (
                client.embeddings.create(
                    model="embedding-3",
                    input=query,
                )
            )

            query_vector = np.array(
                response.data[0].embedding,
                dtype=np.float64,
            )

            query_vector = query_vector / (
                np.linalg.norm(query_vector)
                or 1.0
            )

        except Exception as error:
            print(
                f"  查询向量生成失败: {error}"
            )
            continue

        results = search(
            query_vector, matrix, chunks, TOP_K
        )

        for rank, result in enumerate(
            results, start=1
        ):
            chunk = result["chunk"]
            metadata = chunk["metadata"]

            print(
                f"  {rank}. "
                f"[{result['similarity']:.4f}] "
                f"{chunk['chunk_id']} "
                f"({metadata['part']} | "
                f"{metadata['crm_module'] or '-'})"
            )
            print(
                f"     Q: "
                f"{chunk['question'][:70]}"
            )
            print(
                f"     A: "
                f"{chunk['answer'][:70]}"
            )


if __name__ == "__main__":
    main()
