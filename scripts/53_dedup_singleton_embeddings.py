#!/usr/bin/env python3
"""
Step 15.3 - Singleton funnel Layer C (embedding 近邻去重)
=========================================================

对 A+B 幸存者 (output/singleton_survivors_ab_v1.jsonl)
做两层近邻去重:

    C1  与正式 KB v2 (76 条) 的 question
        cosine >= 0.90
        -> DUPLICATE_OF_OFFICIAL_KB (排除,
           记录 kb_id / cluster_id / similarity)

    C2  幸存者内部 cosine >= 0.90
        -> union-find 分组, 每组只留最优:
           confidence desc
           -> knowledge_value (high > medium)
           -> issue_date 新者优先
           -> issue_key 字典序 (确定性兜底)
        其余成员排除, 记录 kept issue_key

embedding:

    模型 text-embedding-v4 (与 Step 8.2 一致)
    base_url dashscope compatible-mode
    单批 <= 10 条
    向量缓存 output/singleton_dedup_embeddings_cache.jsonl
    重跑只补缺失项, 不重复计费

红线:

- 相似度阈值 0.90 高于 Step 8 聚类的 0.88,
  只拦"接近重复", 不重新做意图聚类
- 排除只依据 frozen embedding 相似度,
  不做任何内容改写

输出 (不修改任何冻结输入):

    output/singleton_dedup_v1.jsonl
    output/singleton_dedup_v1.xlsx
    output/singleton_dedup_embeddings_cache.jsonl

用法 (需在终端配置好 embedding API key):

    .venv/bin/python scripts/53_dedup_singleton_embeddings.py
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

SURVIVORS_FILE = (
    OUTPUT_DIR / "singleton_survivors_ab_v1.jsonl"
)
OFFICIAL_KB_FILE = (
    OUTPUT_DIR / "kb_entries_official_v2.jsonl"
)
CACHE_FILE = (
    OUTPUT_DIR
    / "singleton_dedup_embeddings_cache.jsonl"
)

OUTPUT_JSONL = OUTPUT_DIR / "singleton_dedup_v1.jsonl"
OUTPUT_XLSX = OUTPUT_DIR / "singleton_dedup_v1.xlsx"

MODEL = "embedding-3"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
EMBED_BATCH_SIZE = 10
EMBED_MAX_RETRIES = 4

DEFAULT_DUPLICATE_THRESHOLD = 0.90


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def text_fingerprint(text: str) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:24]


def load_cache():

    cache = {}

    if CACHE_FILE.exists():
        with open(
            CACHE_FILE, encoding="utf-8"
        ) as f:
            for line in f:
                record = json.loads(line)

                if record.get("model") != MODEL:
                    continue

                cache[record["fingerprint"]] = record[
                    "embedding"
                ]

    return cache


def append_cache(fingerprint, text, embedding):

    with open(
        CACHE_FILE, "a", encoding="utf-8"
    ) as output:
        output.write(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "model": MODEL,
                    "text_preview": text[:60],
                    "embedding": embedding,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def embed_texts(client, texts, cache):

    pending = {}

    for text in texts:
        fingerprint = text_fingerprint(text)

        if fingerprint not in cache:
            pending[fingerprint] = text

    batches = []

    items = list(pending.items())

    for start in range(
        0, len(items), EMBED_BATCH_SIZE
    ):
        batches.append(
            items[start : start + EMBED_BATCH_SIZE]
        )

    print(
        f"embedding: 总文本 {len(texts)}, "
        f"缓存命中 {len(texts) - len(pending)}, "
        f"待计算 {len(pending)} "
        f"({len(batches)} 批)"
    )

    for index, batch in enumerate(
        batches, start=1
    ):

        fingerprints = [
            fingerprint
            for fingerprint, _ in batch
        ]
        batch_texts = [
            text for _, text in batch
        ]

        last_error = None

        for attempt in range(
            1, EMBED_MAX_RETRIES + 1
        ):
            try:
                response = (
                    client.embeddings.create(
                        model=MODEL,
                        input=batch_texts,
                    )
                )

                embeddings = [
                    item.embedding
                    for item in response.data
                ]

                if len(embeddings) != len(
                    batch_texts
                ):
                    raise RuntimeError(
                        "embedding 数量不匹配"
                    )

                break

            except Exception as error:

                last_error = error

                if attempt < EMBED_MAX_RETRIES:
                    time.sleep(2 ** attempt)

                else:
                    raise RuntimeError(
                        f"embedding 批次 {index} 连续失败: "
                        f"{error}"
                    )

        for fingerprint, embedding in zip(
            fingerprints, embeddings
        ):
            cache[fingerprint] = embedding

            append_cache(
                fingerprint,
                pending[fingerprint],
                embedding,
            )

        if index % 10 == 0 or index == len(batches):
            print(f"  批次 {index}/{len(batches)}")

    return {
        text: cache[text_fingerprint(text)]
        for text in texts
    }


def normalized_matrix(vectors):

    matrix = np.array(vectors, dtype=np.float64)

    norms = np.linalg.norm(
        matrix, axis=1, keepdims=True
    )

    norms[norms == 0] = 1.0

    return matrix / norms


def knowledge_value_rank(value):

    return {"high": 0, "medium": 1}.get(value, 2)


def main():

    duplicate_threshold = (
        DEFAULT_DUPLICATE_THRESHOLD
    )

    if "--threshold" in sys.argv:
        duplicate_threshold = float(
            sys.argv[
                sys.argv.index("--threshold") + 1
            ]
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY 环境变量"
        )

    survivors = load_jsonl(SURVIVORS_FILE)
    official = load_jsonl(OFFICIAL_KB_FILE)

    print(
        f"幸存者: {len(survivors)} | "
        f"官方 KB: {len(official)}"
    )

    singleton_texts = [
        record["question_normalized"]
        or record["question"]
        for record in survivors
    ]
    official_texts = [
        record["question"] for record in official
    ]

    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=120,
        max_retries=0,
    )

    cache = load_cache()

    all_texts = official_texts + singleton_texts

    embeddings = embed_texts(
        client, all_texts, cache
    )

    # ---------- C1: 与官方 KB 去重 ----------

    official_matrix = normalized_matrix(
        [embeddings[text] for text in official_texts]
    )
    singleton_matrix = normalized_matrix(
        [embeddings[text] for text in singleton_texts]
    )

    c1_similarities = (
        singleton_matrix @ official_matrix.T
    )

    best_indices = c1_similarities.argmax(axis=1)
    best_values = c1_similarities.max(axis=1)

    c1_records = []

    for i, record in enumerate(survivors):
        best_kb = official[int(best_indices[i])]

        c1_records.append({
            "issue_key": record["issue_key"],
            "max_official_similarity": round(
                float(best_values[i]), 4
            ),
            "matched_kb_id": best_kb["kb_id"],
            "matched_cluster_id": best_kb[
                "cluster_id"
            ],
            "matched_question": best_kb[
                "question"
            ],
        })

    # ---------- C2: 幸存者内部去重 ----------

    n = len(survivors)

    sim_matrix = (
        singleton_matrix @ singleton_matrix.T
    )

    parent = list(range(n))

    def find(x):

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):

        ra, rb = find(a), find(b)

        if ra != rb:
            parent[rb] = ra

    pair_edges = []

    for i in range(n):
        for j in range(i + 1, n):
            similarity = float(
                sim_matrix[i, j]
            )

            if similarity >= duplicate_threshold:
                union(i, j)
                pair_edges.append(
                    (i, j, similarity)
                )

    # 相似度分布报告:
    # 换 embedding 模型后阈值是否合理看这里
    iu = np.triu_indices(n, k=1)

    intra_sims = sim_matrix[iu]

    print()
    print("幸存者内部相似度分布 (embedding-3):")

    for q in (0.5, 0.9, 0.95, 0.99, 0.999):
        print(
            f"  p{q * 100:g} = "
            f"{float(np.quantile(intra_sims, q)):.4f}"
        )

    print(
        f"  >= {duplicate_threshold}: "
        f"{int((intra_sims >= duplicate_threshold).sum())} 对"
    )

    print(
        "与官方 KB 相似度: "
        f"max={c1_similarities.max():.4f} "
        f">= {duplicate_threshold}: "
        f"{int((c1_similarities >= duplicate_threshold).sum())} 条"
    )

    groups = {}

    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    def sort_key(i):

        record = survivors[i]

        return (
            -float(record["confidence"]),
            knowledge_value_rank(
                record["knowledge_value"]
            ),
            record["issue_date"],
            record["issue_key"],
        )

    kept_index = {}

    for root, members in groups.items():
        ordered = sorted(members, key=sort_key)
        kept_index[root] = ordered[0]

    # ---------- 汇总决策 ----------

    decisions = []

    for i, record in enumerate(survivors):
        c1 = c1_records[i]

        root = find(i)
        group = groups[root]
        kept = kept_index[root]

        decision = {
            "issue_key": record["issue_key"],
            "question": record["question"],
            "issue_date": record["issue_date"],
            "confidence": record["confidence"],
            "knowledge_value": record[
                "knowledge_value"
            ],
            "resolution": record["resolution"],
            "max_official_similarity": c1[
                "max_official_similarity"
            ],
            "matched_kb_id": c1["matched_kb_id"],
            "matched_cluster_id": c1[
                "matched_cluster_id"
            ],
            "group_size": len(group),
        }

        if (
            c1["max_official_similarity"]
            >= duplicate_threshold
        ):
            decision.update({
                "decision": "excluded",
                "reason": (
                    "C1_DUPLICATE_OF_OFFICIAL_KB"
                ),
                "kept_issue_key": "",
                "pair_similarity": c1[
                    "max_official_similarity"
                ],
            })

        elif i != kept:
            kept_record = survivors[kept]

            decision.update({
                "decision": "excluded",
                "reason": (
                    "C2_INTRA_SURVIVOR_NEAR_DUPLICATE"
                ),
                "kept_issue_key": kept_record[
                    "issue_key"
                ],
                "kept_question": kept_record[
                    "question"
                ],
                "pair_similarity": max(
                    (
                        similarity
                        for a, b, similarity in pair_edges
                        if {a, b} == {i, kept}
                    ),
                    default=None,
                ),
            })

        else:
            decision.update({
                "decision": "kept",
                "reason": "",
                "kept_issue_key": record[
                    "issue_key"
                ],
                "pair_similarity": "",
            })

        decisions.append(decision)

    kept_count = sum(
        1 for d in decisions
        if d["decision"] == "kept"
    )

    with open(
        OUTPUT_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in decisions:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    df = pd.DataFrame(decisions)

    summary = {
        "survivors_input": len(survivors),
        "official_kb_size": len(official),
        "duplicate_threshold": duplicate_threshold,
        "kept": kept_count,
        "excluded_c1": int(
            df["reason"]
            .eq("C1_DUPLICATE_OF_OFFICIAL_KB")
            .sum()
        ),
        "excluded_c2": int(
            df["reason"]
            .eq("C2_INTRA_SURVIVOR_NEAR_DUPLICATE")
            .sum()
        ),
        "embedding_model": MODEL,
    }

    with pd.ExcelWriter(
        OUTPUT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        df.to_excel(
            writer, sheet_name="decisions", index=False
        )
        df[df["decision"] == "kept"].to_excel(
            writer, sheet_name="kept", index=False
        )
        df[df["reason"].str.startswith("C1")].to_excel(
            writer, sheet_name="excluded_c1", index=False
        )
        df[df["reason"].str.startswith("C2")].to_excel(
            writer, sheet_name="excluded_c2", index=False
        )

    print()
    print("=" * 60)
    print("Singleton funnel C 层去重完成（Step 15.3）")
    print("=" * 60)
    print(f"输入: {len(survivors)}")
    print(f"kept: {kept_count}")
    print(f"excluded C1 (与官方 KB 重复): "
          f"{summary['excluded_c1']}")
    print(f"excluded C2 (幸存者内部重复): "
          f"{summary['excluded_c2']}")
    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
