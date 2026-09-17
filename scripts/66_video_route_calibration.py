#!/usr/bin/env python3
"""
Step V4.1 - 视频路由检索阈值冒烟校准（60 模式）
================================================

回答的问题：真实客户问法进来，top1 相似度落在哪个区间？
阈值定多少才能"宁缺毋滥"地决定是否回复视频链接？

索引：触发问题向量（output/video_kb_embeddings_v1.jsonl 缓存），
排除 video-011（manifest include_in_cs_route=false）。

查询集（三组）：

- A 真实正例：import_video_kb_linkage.csv 里的 KB 问题
  （本身就是聊天里真实出现过的客户问题，已与视频互挂 >= 0.80）。
  向量全部在缓存中，不调 API。
- B 口语正例：每视频一条客服视角口语改写（表面措辞与触发问题
  不同），模拟进线问法。新增文本走 embedding-3，写入同一缓存。
- C 真实负例：与全部触发问题最大相似度 <= 0.70 的 KB 问题
  （真实存在但视频未覆盖的主题），确定性等距抽 20 条。
  向量全部在缓存中，不调 API。

判定口径：

- hit：top1 相似度 >= 阈值 且 top1 视频在期望视频集合内
- 误挂：top1 相似度 >= 阈值 但视频不对（A/B 上最危险的失败）
- 误触：负例 C 的 top1 相似度 >= 阈值

输出：output/video_route_calibration_v1.jsonl（逐条 + summary），
阈值扫描表打印到 stdout。

用法（需要 OPENAI_API_KEY，仅 B 组缺向量时才调 API）：

    set -a && . ./.env && set +a
    .venv/bin/python scripts/66_video_route_calibration.py
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

INVENTORY_FILE = OUTPUT_DIR / "video_inventory_v1.jsonl"
TRIGGER_FILE = OUTPUT_DIR / "video_trigger_questions_v1.jsonl"
LINKAGE_FILE = OUTPUT_DIR / "platform_import" / "import_video_kb_linkage.csv"
MANIFEST_FILE = OUTPUT_DIR / "platform_import" / "video_route_manifest.json"
MESSAGES_FILE = OUTPUT_DIR / "messages.jsonl"
CACHE_FILE = OUTPUT_DIR / "video_kb_embeddings_v1.jsonl"
OUTPUT_JSONL = OUTPUT_DIR / "video_route_calibration_v1.jsonl"

MODEL = "embedding-3"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120

# 与 64 号脚本的互挂阈值保持同一口径
NEGATIVE_MAX_SIM = 0.70
NEGATIVE_SAMPLE_N = 20
GRAY_BAND = (0.70, 0.80)
THRESHOLD_SWEEP = [round(x, 2) for x in np.arange(0.68, 0.93, 0.02)]

# B 组：客服视角口语改写（每视频一条，video-011 不在客服路由故略）。
# 措辞刻意与触发问题不同，只保留同一主题。
COLLOQUIAL_QUERIES = {
    "video-001": "企业微信那个功能咋开通啊 听说要管理员授权",
    "video-002": "客户车子快到保养时间了 在哪里填写邀约跟进",
    "video-003": "保险快到期的客户去哪儿看 怎么跟进",
    "video-004": "客户列表里那些小图标都代表啥意思",
    "video-005": "商机列表那个卡片是干嘛用的 从哪里点进去",
    "video-006": "商机卡片的统计规则能自己改吗 比如里程和时间",
    "video-007": "客户上次说还有项目没做 系统里怎么记录下来提醒",
    "video-008": "想查一个客户名下所有的车在哪里查",
    "video-009": "好久没来店里的客户 系统能帮我筛出来吗",
    "video-010": "客户做完车走了 后续怎么定期关怀",
    "video-012": "首页那个数据能不能切换成我自己门店的",
    "video-013": "用系统直接打电话给客户怎么弄 能录音不",
    "video-014": "新来的员工账号怎么建 权限怎么给",
    "video-015": "客户的跟进任务怎么自动分给店里的人",
    "video-016": "之前给客户做的跟进记录在哪里翻出来看",
    "video-017": "换过的零件还在质保期内吗 系统能提醒吗",
    "video-018": "之前员工留下的客户怎么让别人认领去跟进",
    "video-019": "客户分类的标准能按我们店自己的情况调吗",
    "video-020": "跟客户的通话录音和聊天截图能存进系统吗",
    "video-021": "客户预约了哪天进店 在哪里登记和查询",
    "video-022": "我们店想自己定一个专项跟进活动 系统支持吗",
}


def load_jsonl(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_cache():
    cache = {}
    if CACHE_FILE.exists():
        for record in load_jsonl(CACHE_FILE):
            cache[record["key"]] = record
    return cache


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        for record in cache.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def embed_missing(client, texts, cache):
    missing = [t for t in texts if text_key(t) not in cache]
    if not missing:
        print(f"embedding 缓存命中: 0 缺失 / {len(texts)} 条查询")
        return
    print(f"embedding 缓存缺失 {len(missing)} 条，调用 {MODEL} 补齐…")
    for i, text in enumerate(missing, 1):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(model=MODEL, input=[text])
                cache[text_key(text)] = {
                    "key": text_key(text),
                    "model": MODEL,
                    "text_preview": text[:50],
                    "embedding": resp.data[0].embedding,
                }
                break
            except Exception as error:
                if attempt == MAX_RETRIES:
                    raise
                print(f"  第 {i} 条第 {attempt} 次失败: {error}，重试…")
        if i % 10 == 0 or i == len(missing):
            print(f"  已嵌入 {i}/{len(missing)}")
    save_cache(cache)
    print(f"缓存已写回 {CACHE_FILE.name}")


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def build_index(trigger_rows, cache, excluded_videos):
    questions = []
    for row in trigger_rows:
        video_id = row["video_id"]
        if video_id in excluded_videos:
            continue
        for q in row["questions"]:
            if q["gate"] == "pass":
                questions.append(
                    {
                        "video_id": video_id,
                        "question": q["question"],
                        "vector": np.array(
                            cache[text_key(q["question"])]["embedding"],
                            dtype=np.float64,
                        ),
                    }
                )
    matrix = normalize(np.array([q["vector"] for q in questions]))
    return questions, matrix


def search(query_vector, matrix, index_questions):
    sims = matrix @ query_vector
    order = np.argsort(sims)[::-1]
    ranked = []
    for pos in order[:2]:
        ranked.append(
            {
                "video_id": index_questions[int(pos)]["video_id"],
                "question": index_questions[int(pos)]["question"],
                "similarity": round(float(sims[int(pos)]), 4),
            }
        )
    return ranked


def pct(values, p):
    if not values:
        return None
    return round(float(np.percentile(values, p)), 4)


def main():
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    excluded_videos = set(manifest["cs_route_excluded"])
    video_meta = {
        v["video_id"]: v for v in manifest["videos"]
    }

    trigger_rows = load_jsonl(TRIGGER_FILE)
    cache = load_cache()

    index_questions, index_matrix = build_index(
        trigger_rows, cache, excluded_videos
    )
    print(
        f"路由索引: {len(index_questions)} 条触发问题"
        f"（排除 {sorted(excluded_videos)}），向量全缓存命中"
    )

    # ---- A 组：互挂 KB 问题（真实客户问题，缓存向量） ----
    with open(LINKAGE_FILE, encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))
    expected_by_query = {}
    kb_question_by_id = {}
    for pair in pairs:
        kb_id = pair["kb_id"]
        kb_question_by_id[kb_id] = pair["kb_question"]
        expected_by_query.setdefault(kb_id, set()).add(pair["video_id"])

    set_a = []
    for kb_id, question in sorted(kb_question_by_id.items()):
        vector = cache.get(text_key(question))
        if vector is None:
            raise RuntimeError(f"A 组问题缺向量: {kb_id} {question}")
        set_a.append(
            {
                "set": "A_real_kb_linked",
                "query_id": kb_id,
                "query": question,
                "expected": sorted(expected_by_query[kb_id]),
                "vector": np.array(vector["embedding"], dtype=np.float64),
            }
        )

    # ---- C 组候选：全部 KB 问题对索引的最大相似度 ----
    linkage_kb_ids = set(kb_question_by_id)
    kb_all = [json.loads(line)["question"] for line in open(
        ROOT_DIR / "releases" / "current" / "chunks.jsonl", encoding="utf-8"
    )]
    kb_vectors = {}
    for question in kb_all:
        entry = cache.get(text_key(question))
        if entry is not None:
            kb_vectors[question] = np.array(
                entry["embedding"], dtype=np.float64
            )
    kb_matrix = normalize(
        np.array([kb_vectors[q] for q in kb_all if q in kb_vectors])
    )
    kb_questions_ordered = [q for q in kb_all if q in kb_vectors]
    max_sims = kb_matrix @ index_matrix.T
    max_by_question = {
        q: float(max_sims[i].max())
        for i, q in enumerate(kb_questions_ordered)
    }
    gray = [
        q for q, s in max_by_question.items()
        if GRAY_BAND[0] < s < GRAY_BAND[1] and q not in linkage_kb_ids
    ]
    negative_pool = sorted(
        [
            (s, q)
            for q, s in max_by_question.items()
            if s <= NEGATIVE_MAX_SIM and q not in linkage_kb_ids
        ],
        reverse=True,
    )
    step = max(1, len(negative_pool) // NEGATIVE_SAMPLE_N)
    negatives = negative_pool[::step][:NEGATIVE_SAMPLE_N]
    print(
        f"负例池: {len(negative_pool)} 条（max_sim<= {NEGATIVE_MAX_SIM}），"
        f"等距抽 {len(negatives)} 条；灰区 {len(gray)} 条"
        f"（{GRAY_BAND[0]}–{GRAY_BAND[1]}，不计入正负例）"
    )

    set_c = []
    for sim, question in negatives:
        set_c.append(
            {
                "set": "C_real_kb_negative",
                "query_id": text_key(question)[:8],
                "query": question,
                "expected": [],
                "vector": kb_vectors[question],
                "max_sim_to_index": round(sim, 4),
            }
        )

    # ---- B 组：口语改写（可能需要调 API） ----
    set_b = []
    for video_id, query in sorted(COLLOQUIAL_QUERIES.items()):
        set_b.append(
            {
                "set": "B_colloquial",
                "query_id": video_id,
                "query": query,
                "expected": [video_id],
            }
        )

    client = None
    if any(text_key(q["query"]) not in cache for q in set_b):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "B 组有查询缺向量，需要 OPENAI_API_KEY"
                "（set -a && . ./.env && set +a）"
            )
        client = OpenAI(
            base_url=BASE_URL,
            api_key=api_key,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )
    embed_missing(
        client, [q["query"] for q in set_b], cache
    )
    for q in set_b:
        q["vector"] = np.array(
            cache[text_key(q["query"])]["embedding"], dtype=np.float64
        )

    # ---- 逐条检索 ----
    results = []
    for q in set_a + set_b + set_c:
        ranked = search(normalize(q["vector"][None, :])[0],
                        index_matrix, index_questions)
        top1 = ranked[0]
        top2 = ranked[1] if len(ranked) > 1 else None
        correct = top1["video_id"] in q["expected"] if q["expected"] else None
        results.append(
            {
                "set": q["set"],
                "query_id": q["query_id"],
                "query": q["query"],
                "expected": q["expected"],
                "top1": top1,
                "top2": top2,
                "top1_correct": correct,
                "max_sim_to_index": q.get("max_sim_to_index"),
            }
        )

    # ---- 阈值扫描 ----
    sweep = []
    for threshold in THRESHOLD_SWEEP:
        pos = [r for r in results if r["expected"]]
        neg = [r for r in results if not r["expected"]]
        hits = sum(
            1 for r in pos
            if r["top1"]["similarity"] >= threshold and r["top1_correct"]
        )
        wrong = sum(
            1 for r in pos
            if r["top1"]["similarity"] >= threshold and not r["top1_correct"]
        )
        false_fire = sum(
            1 for r in neg if r["top1"]["similarity"] >= threshold
        )
        sweep.append(
            {
                "threshold": threshold,
                "hit_rate": round(hits / len(pos), 4),
                "wrong_video_fire": wrong,
                "negative_false_fire": false_fire,
            }
        )

    correct_pos_sims = [
        r["top1"]["similarity"]
        for r in results if r["expected"] and r["top1_correct"]
    ]
    wrong_pos_sims = [
        r["top1"]["similarity"]
        for r in results if r["expected"] and not r["top1_correct"]
    ]
    neg_sims = [r["top1"]["similarity"] for r in results if not r["expected"]]
    summary = {
        "record_type": "summary",
        "index_size": len(index_questions),
        "excluded_videos": sorted(excluded_videos),
        "query_counts": {
            "A_real_kb_linked": len(set_a),
            "B_colloquial": len(set_b),
            "C_real_kb_negative": len(set_c),
        },
        "gray_band_count": len(gray),
        "positive_top1_correct": {
            "n": len(correct_pos_sims),
            "p10": pct(correct_pos_sims, 10),
            "p50": pct(correct_pos_sims, 50),
            "min": pct(correct_pos_sims, 0),
        },
        "positive_top1_wrong": {
            "n": len(wrong_pos_sims),
            "sims": sorted(wrong_pos_sims, reverse=True),
        },
        "negative_top1": {
            "n": len(neg_sims),
            "p50": pct(neg_sims, 50),
            "p90": pct(neg_sims, 90),
            "max": pct(neg_sims, 100),
        },
        "threshold_sweep": sweep,
    }

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print("\n=== top1 相似度分布 ===")
    print(
        f"正例-命中正确视频 (n={len(correct_pos_sims)}): "
        f"min={summary['positive_top1_correct']['min']} "
        f"p10={summary['positive_top1_correct']['p10']} "
        f"p50={summary['positive_top1_correct']['p50']}"
    )
    print(
        f"正例-top1 落错视频 (n={len(wrong_pos_sims)}): "
        f"{summary['positive_top1_wrong']['sims']}"
    )
    print(
        f"负例 (n={len(neg_sims)}): "
        f"p50={summary['negative_top1']['p50']} "
        f"p90={summary['negative_top1']['p90']} "
        f"max={summary['negative_top1']['max']}"
    )
    print("\n=== 阈值扫描 ===")
    print("threshold  hit_rate  wrong_video_fire  negative_false_fire")
    for s in sweep:
        print(
            f"{s['threshold']:.2f}      "
            f"{s['hit_rate']:.4f}    "
            f"{s['wrong_video_fire']}                 "
            f"{s['negative_false_fire']}"
        )
    print(f"\n输出: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
