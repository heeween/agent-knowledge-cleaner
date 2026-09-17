#!/usr/bin/env python3
"""
Step V4.2 - 视频路由发布文件组装（确定性，无 LLM，无 API）
==========================================================

把三个冻结产物组装成 release 附带文件（发布态格式）：

- output/video_route.jsonl  154 条路由索引
  （22 视频 162 条触发问题，剔除 video-011 的 8 条；
  字段 = manifest 元数据 × 触发问题 × 向量缓存）
- output/video_linkage.jsonl  50 对视频↔KB 互挂表

自校验（全部通过才写文件）：

- video-011 不出现在路由；条数守卫 154/50
- video_url 全部 https 且与 video_route_manifest.json 一致
- 向量维度 2048、model=embedding-3、content_sha256=sha256(question)
- 互挂 kb_id 必须存在于当前发布快照 chunks.jsonl 且 similarity>=0.80

publish() 会把这两个文件复制进 release 目录并记入 manifest.video 段；
本脚本只负责生成与校验，可重复运行（幂等覆盖）。
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

MANIFEST_FILE = OUTPUT / "platform_import" / "video_route_manifest.json"
TRIGGER_FILE = OUTPUT / "video_trigger_questions_v1.jsonl"
CACHE_FILE = OUTPUT / "video_kb_embeddings_v1.jsonl"
LINKAGE_FILE = OUTPUT / "platform_import" / "import_video_kb_linkage.csv"
CHUNKS_FILE = ROOT / "releases" / "current" / "chunks.jsonl"

ROUTE_OUT = OUTPUT / "video_route.jsonl"
LINKAGE_OUT = OUTPUT / "video_linkage.jsonl"

MODEL = "embedding-3"
DIMENSION = 2048
ROUTE_EXPECTED = 154
LINKAGE_EXPECTED = 50
LINK_MIN_SIMILARITY = 0.80
# 客户可见链接走应用内使用说明页（xcrm 已对 /instruction 前缀免登录）；
# OSS 直链仅是素材源，不进回复
VIDEO_APP_URL_TEMPLATE = "https://a.rcar.vip/instruction/video/{video_id}"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def build_route(manifest, trigger_rows, cache):
    route = []
    for row in trigger_rows:
        video_id = row["video_id"]
        meta = manifest["videos"][video_id]
        if not meta["include_in_cs_route"]:
            continue
        for q in row["questions"]:
            if q["gate"] != "pass":
                continue
            question = q["question"]
            entry = cache[text_key(question)]
            route.append(
                {
                    "video_id": video_id,
                    "question": question,
                    "video_url": VIDEO_APP_URL_TEMPLATE.format(video_id=video_id),
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "duration_hms": meta["duration_hms"],
                    "embedding": entry["embedding"],
                    "model": entry["model"],
                    "content_sha256": hashlib.sha256(
                        question.encode("utf-8")
                    ).hexdigest(),
                }
            )
    return route


def build_linkage(manifest, chunks_kb_ids):
    linkage = []
    with open(LINKAGE_FILE, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            video_id = row["video_id"]
            kb_id = row["kb_id"]
            similarity = float(row["similarity"])
            if kb_id not in chunks_kb_ids:
                raise ValueError(f"互挂指向未知知识条目: {kb_id}")
            if similarity < LINK_MIN_SIMILARITY:
                raise ValueError(f"互挂相似度低于下限: {kb_id} {similarity}")
            meta = manifest["videos"][video_id]
            linkage.append(
                {
                    "video_id": video_id,
                    "kb_id": kb_id,
                    "similarity": similarity,
                    "kb_question": row["kb_question"],
                    "video_url": VIDEO_APP_URL_TEMPLATE.format(video_id=video_id),
                    "title": meta["title"],
                }
            )
    return sorted(linkage, key=lambda x: (x["video_id"], x["kb_id"]))


def validate(route, linkage, manifest):
    excluded = set(manifest["cs_route_excluded"])
    if len(route) != ROUTE_EXPECTED:
        raise ValueError(f"路由条数守卫失败: 期望 {ROUTE_EXPECTED} 实际 {len(route)}")
    if len(linkage) != LINKAGE_EXPECTED:
        raise ValueError(f"互挂条数守卫失败: 期望 {LINKAGE_EXPECTED} 实际 {len(linkage)}")
    for item in route:
        if item["video_id"] in excluded:
            raise ValueError(f"被排除视频进入路由: {item['video_id']}")
        if not item["video_url"].startswith("https://"):
            raise ValueError(f"非 https 链接: {item['video_url']}")
        if len(item["embedding"]) != DIMENSION:
            raise ValueError(f"向量维度不符: {item['question'][:20]}")
        if item["model"] != MODEL:
            raise ValueError(f"向量模型不符: {item['model']}")
        if item["content_sha256"] != hashlib.sha256(
            item["question"].encode("utf-8")
        ).hexdigest():
            raise ValueError(f"content_sha256 不符: {item['question'][:20]}")


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest["videos"] = {v["video_id"]: v for v in manifest["videos"]}

    trigger_rows = [json.loads(line) for line in TRIGGER_FILE.open(encoding="utf-8") if line.strip()]
    cache = {
        record["key"]: record
        for record in (json.loads(line) for line in CACHE_FILE.open(encoding="utf-8") if line.strip())
    }

    chunks_kb_ids = {
        json.loads(line)["kb_id"]
        for line in CHUNKS_FILE.open(encoding="utf-8")
        if line.strip()
    }

    route = build_route(manifest, trigger_rows, cache)
    linkage = build_linkage(manifest, chunks_kb_ids)
    validate(route, linkage, manifest)

    write_jsonl(ROUTE_OUT, route)
    write_jsonl(LINKAGE_OUT, linkage)
    print(
        f"OK video_route.jsonl={len(route)} 条 / "
        f"video_linkage.jsonl={len(linkage)} 对 "
        f"(排除 {manifest['cs_route_excluded']})"
    )


if __name__ == "__main__":
    main()
