#!/usr/bin/env python3
"""
Step 17.2 - RAG chunk embedding
================================

对 rag_chunks_v1.jsonl (645 chunks) 逐条生成
embedding 向量 (bigmodel embedding-3,
配置与 50/53/54 一致)。

- 向量缓存 output/rag_chunk_embeddings_v1.jsonl
  (chunk_id 去重, 中断重跑只补缺失)
- 逐条请求 (645 条, 单条文本短)

输出:

    output/rag_chunk_embeddings_v1.jsonl
    (chunk_id, model, embedding)

用法:

    .venv/bin/python scripts/59_embed_rag_chunks.py
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CHUNKS_FILE = OUTPUT_DIR / "rag_chunks_v1.jsonl"
EMBEDDINGS_FILE = (
    OUTPUT_DIR / "rag_chunk_embeddings_v1.jsonl"
)

MODEL = "embedding-3"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY 环境变量"
        )

    chunks = load_jsonl(CHUNKS_FILE)

    done = {}

    if EMBEDDINGS_FILE.exists():
        for record in load_jsonl(EMBEDDINGS_FILE):
            if record.get("model") == MODEL:
                done[record["chunk_id"]] = record

    pending = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in done
    ]

    print(
        f"chunks: {len(chunks)} | "
        f"已有向量: {len(done)} | "
        f"待嵌入: {len(pending)}"
    )

    if not pending:
        print("全部已嵌入, 无需请求")
        return

    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )

    with open(
        EMBEDDINGS_FILE, "a", encoding="utf-8"
    ) as output:

        for index, chunk in enumerate(
            pending, start=1
        ):

            last_error = None

            for attempt in range(
                1, MAX_RETRIES + 1
            ):
                try:
                    response = (
                        client.embeddings.create(
                            model=MODEL,
                            input=chunk["text"],
                        )
                    )

                    embedding = (
                        response.data[0].embedding
                    )

                    break

                except Exception as error:

                    last_error = error

                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                    else:
                        raise RuntimeError(
                            f"{chunk['chunk_id']} "
                            f"连续失败: {error}"
                        )

            output.write(
                json.dumps(
                    {
                        "chunk_id": chunk[
                            "chunk_id"
                        ],
                        "model": MODEL,
                        "embedding": embedding,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if index % 50 == 0 or index == len(
                pending
            ):
                output.flush()
                print(
                    f"  {index}/{len(pending)}",
                    flush=True,
                )

    print()
    print(f"输出: {EMBEDDINGS_FILE}")
    print(f"向量总数: {len(done) + len(pending)}")


if __name__ == "__main__":
    main()
