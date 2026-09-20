#!/usr/bin/env python3
"""
Step 18.4 - funnel v2 Layer A + 去重
======================================

输入 (71 产物 + 复用层):

    funnel_v2_reused.jsonl     778 行 (含 v1 excluded 标志)
    funnel_v2_extracted.jsonl  新抽取行
    funnel_v2_enriched.jsonl   富集行

流程:

1. Layer A 硬过滤重算 (与 52 同规则):
   A1 confidence<0.7 / A2 knowledge_value=low /
   A3 answer<20字 / A4 temporal=future_plan / A5 temporal=historical
   (Layer B 时效对 after-2026-03 子集恒不触发, 仍保留检查)
2. C1 精确去重: question_normalized 指纹相同
3. C2 近似去重: 组内 cosine >= 0.90 (embedding-3, 阈值同 53)
4. C3 存量查重 (v2 新增): 与 26 条 survivor question
   cosine >= 0.90 -> 视为已有知识, 剔除

输出:

    output/funnel_v2.jsonl                   (全量审计)
    output/funnel_v2_dedup_survivors.jsonl   (进入 LLM 过滤层)

向量缓存: 复用 output/singleton_dedup_embeddings_cache.jsonl
(内容寻址, 追加安全)
"""

import hashlib
import json
import math
import os
from pathlib import Path

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

REUSED_FILE = OUTPUT_DIR / "funnel_v2_reused.jsonl"
EXTRACTED_FILE = OUTPUT_DIR / "funnel_v2_extracted.jsonl"
ENRICHED_FILE = OUTPUT_DIR / "funnel_v2_enriched.jsonl"
BASELINE_V4 = OUTPUT_DIR / "kb_entries_official_v4.jsonl"
CACHE_FILE = OUTPUT_DIR / "singleton_dedup_embeddings_cache.jsonl"

FUNNEL_OUT = OUTPUT_DIR / "funnel_v2.jsonl"
SURVIVORS_OUT = OUTPUT_DIR / "funnel_v2_dedup_survivors.jsonl"

MODEL = "embedding-3"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DUPLICATE_THRESHOLD = 0.90
EMBED_BATCH = 16


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def text_fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_cache():
    cache = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("model") == MODEL:
                cache[record["fingerprint"]] = record["embedding"]
    return cache


def append_cache(fingerprint, text, embedding):
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "fingerprint": fingerprint, "model": MODEL,
            "text_preview": text[:50], "embedding": embedding,
        }, ensure_ascii=False) + "\n")


def embed_texts(api, texts, cache):
    pending = []
    for text in texts:
        fingerprint = text_fingerprint(text)
        if fingerprint not in cache:
            pending.append((fingerprint, text))
    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start:start + EMBED_BATCH]
        response = api.embeddings.create(model=MODEL, input=[text for _, text in batch])
        for (fingerprint, text), item in zip(batch, response.data):
            embedding = item.embedding
            cache[fingerprint] = embedding
            append_cache(fingerprint, text, embedding)
        print(f"  embedding {min(start + EMBED_BATCH, len(pending))}/{len(pending)}", flush=True)
    return [cache[text_fingerprint(text)] for text in texts]


def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def value_rank(value):
    return {"high": 0, "medium": 1}.get(value, 2)


def layer_a_flags(row):
    flags = []
    if float(row.get("confidence") or 0) < 0.7:
        flags.append("A1_CONFIDENCE")
    if row.get("knowledge_value") == "low":
        flags.append("A2_LOW_VALUE")
    if len(row.get("answer") or "") < 20:
        flags.append("A3_SHORT_ANSWER")
    if row.get("temporal_status") == "future_plan":
        flags.append("A4_FUTURE_PLAN")
    if row.get("temporal_status") == "historical":
        flags.append("A5_HISTORICAL")
    if row.get("issue_date", "9999") < "2025-08-06":
        flags.append("B1_STALE")
    return flags


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")
    api = OpenAI(base_url=BASE_URL, api_key=os.environ["OPENAI_API_KEY"])

    rows = load_jsonl(REUSED_FILE) + load_jsonl(ENRICHED_FILE) + load_jsonl(EXTRACTED_FILE)
    seen_keys = set()
    unique_rows = []
    for row in rows:
        key = row["issue_key"]
        if key in seen_keys:
            raise SystemExit(f"守卫失败: issue_key 重复 {key}")
        seen_keys.add(key)
        row["normalized"] = (row.get("question_normalized") or row["question"]).strip()
        unique_rows.append(row)
    print(f"合并输入: {len(unique_rows)} 行 (reused/enriched/extracted)")

    # Layer A
    for row in unique_rows:
        row["layer_a_flags"] = layer_a_flags(row)
        row["excluded_v2"] = bool(row["layer_a_flags"])
    survivors_ab = [row for row in unique_rows if not row["excluded_v2"]]
    print(f"Layer A 后幸存: {len(survivors_ab)} 行")

    # C3 参照: 26 条 survivor question
    baseline = load_jsonl(BASELINE_V4)
    survivor_questions = [record["question"] for record in baseline]

    cache = load_cache()
    print("计算向量 (缓存命中优先)...")
    survivor_vectors = embed_texts(api, survivor_questions, cache)
    question_vectors = embed_texts(api, [row["normalized"] for row in survivors_ab], cache)

    # C1 + C2 + C3
    kept = []
    kept_vectors = []
    dropped_exact, dropped_near, dropped_existing = 0, 0, 0
    for row, vector in zip(survivors_ab, question_vectors):
        fingerprint = text_fingerprint(row["normalized"])
        exact_hit = next((k for k in kept if text_fingerprint(k["normalized"]) == fingerprint), None)
        if exact_hit is not None:
            row["dedup_decision"] = "dropped_exact_duplicate_of"
            row["dedup_target"] = exact_hit["issue_key"]
            dropped_exact += 1
            continue
        near_hit = None
        for kept_row, kept_vector in zip(kept, kept_vectors):
            if cosine(vector, kept_vector) >= DUPLICATE_THRESHOLD:
                near_hit = kept_row
                break
        if near_hit is not None:
            row["dedup_decision"] = "dropped_near_duplicate_of"
            row["dedup_target"] = near_hit["issue_key"]
            dropped_near += 1
            continue
        existing_hit = None
        for index, survivor_vector in enumerate(survivor_vectors):
            if cosine(vector, survivor_vector) >= DUPLICATE_THRESHOLD:
                existing_hit = survivor_questions[index]
                break
        if existing_hit is not None:
            row["dedup_decision"] = "dropped_existing_kb_duplicate"
            row["dedup_target"] = existing_hit
            dropped_existing += 1
            continue
        row["dedup_decision"] = "kept"
        row["dedup_target"] = None
        kept.append(row)
        kept_vectors.append(vector)
    print(f"C1 精确重复 {dropped_exact}, C2 近似重复 {dropped_near}, C3 存量重复 {dropped_existing}")
    print(f"去重后 kept: {len(kept)} 行")

    with open(FUNNEL_OUT, "w", encoding="utf-8") as f:
        for row in unique_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with open(SURVIVORS_OUT, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"写出 {FUNNEL_OUT.name} / {SURVIVORS_OUT.name}")


if __name__ == "__main__":
    main()
