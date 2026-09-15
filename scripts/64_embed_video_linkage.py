#!/usr/bin/env python3
"""Step V3 - 视频问题向量化 / 阈值标定 / 视频↔KB 互挂。

对 Step V2 产出的触发问题（仅 gate=pass）与
官方 KB v3 的 645 条 question（rag_chunks_v1.jsonl）
统一用 bigmodel embedding-3 嵌入（与 59 同款配置），
然后做三类 deterministic 分析：

1. 跨视频撞车：某视频的触发问题与其他视频问题
   的高相似对（>= CROSS_VIDEO_THRESHOLD），标记
   "一个问题可能命中多个视频"的归属风险
2. 视频↔KB 互挂：视频问题 vs KB question 的
   相似对（>= --link-threshold），聚合到视频级
   （每个 video×kb 保留最高相似度的那对问题），
   供"文字答案附视频 / 视频卡附文字答案"使用
3. 阈值标定：三个分布的统计
   - 视频问题 vs 自视频质心（内聚度，info）
   - 视频问题 vs 其他视频问题 max（异视频噪声带）
   - 视频问题 vs KB question max（相关带）
   并给出检索阈值建议；上线前需按 60 的方式
   用真实 query 冒烟校准

向量缓存：output/video_kb_embeddings_v1.jsonl
（text sha256 + model 去重，中断续跑只补缺失；
缓存按文本内容寻址，问题改版后自动失效）

用法：

    .venv/bin/python scripts/64_embed_video_linkage.py --dry-run
    .venv/bin/python scripts/64_embed_video_linkage.py
    .venv/bin/python scripts/64_embed_video_linkage.py --link-threshold 0.80
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from openpyxl import Workbook

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

TRIGGER_FILE = OUTPUT_DIR / "video_trigger_questions_v1.jsonl"
CHUNKS_FILE = OUTPUT_DIR / "rag_chunks_v1.jsonl"
CACHE_FILE = OUTPUT_DIR / "video_kb_embeddings_v1.jsonl"
OUTPUT_JSONL = OUTPUT_DIR / "video_kb_linkage_v1.jsonl"
OUTPUT_XLSX = OUTPUT_DIR / "video_kb_linkage_v1.xlsx"

MODEL = "embedding-3"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120

CROSS_VIDEO_THRESHOLD = 0.90
DEFAULT_LINK_THRESHOLD = 0.80


def load_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def collect_items(trigger_records, chunks):
    """待嵌入条目 [(key, id, text)]。"""
    items = []
    for rec in trigger_records:
        for qi, q in enumerate(rec["questions"]):
            if q["gate"] != "pass":
                continue
            items.append((text_key(q["question"]),
                          f"{rec['video_id']}#{qi}", q["question"]))
    for c in chunks:
        items.append((text_key(c["question"]), c["chunk_id"], c["question"]))
    return items


def embed_missing(client, items, cache):
    pending = [it for it in items if it[0] not in cache]
    print(f"embeddings: {len(items)} total, "
          f"{len(items) - len(pending)} cached, {len(pending)} to embed")

    def flush():
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            for rec in cache.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for i, (key, item_id, text) in enumerate(pending, 1):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(model=MODEL, input=[text])
                cache[key] = {
                    "key": key,
                    "model": MODEL,
                    "text_preview": text[:60],
                    "embedding": resp.data[0].embedding,
                }
                break
            except Exception as e:  # noqa: BLE001
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"{item_id}: 嵌入连续失败: {e}"
                    ) from e
                time.sleep(2 * attempt)
        if i % 50 == 0 or i == len(pending):
            flush()
            print(f"  embedded {i}/{len(pending)}")
    flush()


def percentile(values, p):
    if not values:
        return 0.0
    return round(float(np.percentile(values, p)), 4)


def analyse(trigger_records, chunks, cache, link_threshold):
    video_vecs, kb_vecs = {}, []

    for rec in trigger_records:
        lst = []
        for qi, q in enumerate(rec["questions"]):
            if q["gate"] != "pass":
                continue
            entry = cache.get(text_key(q["question"]))
            if entry is None:
                raise RuntimeError(f"{rec['video_id']}#{qi}: 缓存缺失")
            lst.append((qi, q["question"], np.array(entry["embedding"])))
        video_vecs[rec["video_id"]] = lst

    for c in chunks:
        entry = cache.get(text_key(c["question"]))
        if entry is None:
            raise RuntimeError(f"{c['chunk_id']}: KB 缓存缺失")
        kb_vecs.append((c["chunk_id"], c["question"],
                        np.array(entry["embedding"])))

    kb_matrix = np.array([v for _, _, v in kb_vecs])
    kb_norms = np.linalg.norm(kb_matrix, axis=1)
    vid_ids = sorted(video_vecs)

    # --- 1) 跨视频撞车（问题级两两） ---
    cross_pairs = []
    for vi, vid_a in enumerate(vid_ids):
        for vid_b in vid_ids[vi + 1:]:
            for qi_a, q_a, v_a in video_vecs[vid_a]:
                for qi_b, q_b, v_b in video_vecs[vid_b]:
                    sim = cosine(v_a, v_b)
                    if sim >= CROSS_VIDEO_THRESHOLD:
                        cross_pairs.append({
                            "video_a": vid_a, "question_a": q_a,
                            "video_b": vid_b, "question_b": q_b,
                            "similarity": round(sim, 4),
                        })
    cross_pairs.sort(key=lambda p: -p["similarity"])

    # --- 2) 视频↔KB 互挂 + 3) 标定统计 ---
    cohesion, cross_max_sims, kb_max_sims = [], [], []
    question_stats = []
    qlevel_links = []
    video_best = {}

    for vid in vid_ids:
        qs = video_vecs[vid]
        centroid = np.mean([v for _, _, v in qs], axis=0)
        for qi, question, vec in qs:
            coh = cosine(vec, centroid)
            cohesion.append(coh)

            cross_max = max(
                (cosine(vec, v2)
                 for vid2 in vid_ids if vid2 != vid
                 for _, _, v2 in video_vecs[vid2]),
                default=0.0,
            )
            cross_max_sims.append(cross_max)

            sims = (kb_matrix @ vec) / (kb_norms * np.linalg.norm(vec))
            top_i = int(np.argmax(sims))
            kb_sim = float(sims[top_i])
            kb_max_sims.append(kb_sim)
            kb_id, kb_question, _ = kb_vecs[top_i]

            question_stats.append({
                "video_id": vid, "question_index": qi,
                "question": question,
                "cohesion_own_video": round(coh, 4),
                "max_sim_other_video": round(cross_max, 4),
                "max_sim_kb": round(kb_sim, 4),
                "nearest_kb": kb_id,
            })

            if kb_sim >= link_threshold:
                qlevel_links.append({
                    "video_id": vid, "kb_id": kb_id,
                    "similarity": round(kb_sim, 4),
                    "video_question": question,
                    "kb_question": kb_question,
                })
                prev = video_best.get((vid, kb_id))
                if prev is None or kb_sim > prev["similarity"]:
                    video_best[(vid, kb_id)] = {
                        "video_id": vid, "kb_id": kb_id,
                        "similarity": round(kb_sim, 4),
                        "video_question": question,
                        "kb_question": kb_question,
                    }

    video_kb_links = sorted(
        video_best.values(), key=lambda x: -x["similarity"]
    )

    # 视频级汇总（含 0 link 的视频）
    video_link_summary = []
    for vid in vid_ids:
        links = [l for l in video_kb_links if l["video_id"] == vid]
        video_link_summary.append({
            "video_id": vid,
            "linked_kb_count": len(links),
            "top_kb": links[0]["kb_id"] if links else "",
            "top_kb_similarity": links[0]["similarity"] if links else 0.0,
        })

    calibration = {
        "cohesion_own_video": {
            "p50": percentile(cohesion, 50),
            "p90": percentile(cohesion, 90),
            "min": percentile(cohesion, 0),
        },
        "max_sim_other_video": {
            "p50": percentile(cross_max_sims, 50),
            "p90": percentile(cross_max_sims, 90),
            "p99": percentile(cross_max_sims, 99),
            "max": percentile(cross_max_sims, 100),
            "count_ge_0.90": sum(
                1 for s in cross_max_sims if s >= CROSS_VIDEO_THRESHOLD),
        },
        "max_sim_kb": {
            "p50": percentile(kb_max_sims, 50),
            "p90": percentile(kb_max_sims, 90),
            "max": percentile(kb_max_sims, 100),
        },
        "link_threshold_used": link_threshold,
        "cross_video_threshold_used": CROSS_VIDEO_THRESHOLD,
        "note": "检索阈值上线前需用真实 query 冒烟校准（60 模式）",
    }

    summary = {
        "videos": len(trigger_records),
        "trigger_questions_pass": len(question_stats),
        "kb_questions": len(kb_vecs),
        "embedding_model": MODEL,
        "cross_video_pairs_ge_0.90": len(cross_pairs),
        "video_kb_video_level_links": len(video_kb_links),
        "video_kb_question_level_links": len(qlevel_links),
        "videos_with_no_kb_link": sum(
            1 for s in video_link_summary
            if s["linked_kb_count"] == 0
        ),
    }

    return {
        "summary": summary,
        "calibration": calibration,
        "question_stats": question_stats,
        "cross_pairs": cross_pairs,
        "video_kb_links": video_kb_links,
        "qlevel_links": qlevel_links,
        "video_link_summary": video_link_summary,
    }


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def write_outputs(result):
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        f.write(json.dumps(result["summary"], ensure_ascii=False) + "\n")
        f.write(json.dumps(result["calibration"], ensure_ascii=False) + "\n")
        for link in result["video_kb_links"]:
            f.write(json.dumps({"type": "video_kb_link", **link},
                               ensure_ascii=False) + "\n")
        for pair in result["cross_pairs"]:
            f.write(json.dumps({"type": "cross_video_pair", **pair},
                               ensure_ascii=False) + "\n")

    wb = Workbook()

    ws = wb.active
    ws.title = "summary"
    for k, v in result["summary"].items():
        ws.append([k, v])
    ws.append([])
    ws.append(["calibration"])
    for k, v in result["calibration"].items():
        ws.append([k, json.dumps(v, ensure_ascii=False)
                   if isinstance(v, dict) else v])

    ws = wb.create_sheet("video_link_summary")
    ws.append(["video_id", "linked_kb_count", "top_kb", "top_kb_similarity"])
    for r in result["video_link_summary"]:
        ws.append([r["video_id"], r["linked_kb_count"],
                   r["top_kb"], r["top_kb_similarity"]])

    ws = wb.create_sheet("video_kb_links")
    ws.append(["video_id", "kb_id", "similarity",
               "video_question", "kb_question"])
    for l in result["video_kb_links"]:
        ws.append([l["video_id"], l["kb_id"], l["similarity"],
                   l["video_question"], l["kb_question"]])

    ws = wb.create_sheet("cross_video_pairs")
    ws.append(["video_a", "question_a", "video_b", "question_b",
               "similarity"])
    for p in result["cross_pairs"]:
        ws.append([p["video_a"], p["question_a"],
                   p["video_b"], p["question_b"], p["similarity"]])

    ws = wb.create_sheet("question_stats")
    ws.append(["video_id", "question_index", "question",
               "cohesion_own_video", "max_sim_other_video",
               "max_sim_kb", "nearest_kb"])
    for r in result["question_stats"]:
        ws.append([r["video_id"], r["question_index"], r["question"],
                   r["cohesion_own_video"], r["max_sim_other_video"],
                   r["max_sim_kb"], r["nearest_kb"]])

    wb.save(OUTPUT_XLSX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="不调用 API：统计问题规模")
    ap.add_argument("--link-threshold", type=float,
                    default=DEFAULT_LINK_THRESHOLD)
    args = ap.parse_args()

    if not TRIGGER_FILE.exists():
        raise RuntimeError(f"缺少 {TRIGGER_FILE}，先运行 63")
    if not CHUNKS_FILE.exists():
        raise RuntimeError(f"缺少 {CHUNKS_FILE}")

    trigger_records = load_jsonl(TRIGGER_FILE)
    chunks = load_jsonl(CHUNKS_FILE)
    pass_q = sum(
        1 for r in trigger_records
        for q in r["questions"] if q["gate"] == "pass"
    )
    print(f"videos={len(trigger_records)} "
          f"trigger_questions_pass={pass_q} kb_questions={len(chunks)}")

    if args.dry_run:
        print(f"[dry-run] 待嵌入 {pass_q + len(chunks)} 条 "
              f"(视频问题 {pass_q} + KB 问题 {len(chunks)})")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")

    items = collect_items(trigger_records, chunks)
    cache = {
        r["key"]: r for r in load_jsonl(CACHE_FILE)
        if r.get("model") == MODEL
    }
    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
    )
    embed_missing(client, items, cache)

    result = analyse(trigger_records, chunks, cache, args.link_threshold)
    write_outputs(result)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(result["calibration"], ensure_ascii=False, indent=2))
    print(f"\nwrote: {OUTPUT_JSONL}")
    print(f"wrote: {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
