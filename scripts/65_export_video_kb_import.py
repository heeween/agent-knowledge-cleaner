#!/usr/bin/env python3
"""Step V4 - 视频知识库平台导入包 + 服务路由清单（deterministic，无 LLM）。

输入：
- output/video_inventory_v1.jsonl        (62, 链接/元数据)
- output/video_trigger_questions_v1.jsonl (63, 触发问题)
- output/video_kb_linkage_v1.jsonl       (64, 可选, 视频↔KB 互挂)

输出（全部新文件，不覆盖既有导出）：

- output/platform_import/import_video_qa.csv
  Dify/FastGPT 视频知识库导入（问答模式），
  每行一条 gate=pass 触发问题，
  answer = 【视频讲解】标题 + 摘要 + OSS 链接 + 时长
- output/platform_import/video_route_manifest.json
  自建服务管线路由清单：每视频 url / 时长 / 分类 /
  include_in_cs_route / 触发问题 / 互挂 KB id 列表
- output/platform_import/import_video_kb_linkage.csv
  视频↔KB 互挂表（64 输出存在时才生成）

护栏：
- 22 个视频必须全部在 63 输出中
- question/answer/url 全列非空
- 跨视频完全重复问题去重（保留 video_id 靠前者）并告警
- video-011（产品价值宣传）include_in_cs_route=false
  （仍进平台 CSV；该 flag 只约束自建管线的主动附挂）

用法：

    .venv/bin/python scripts/65_export_video_kb_import.py
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
IMPORT_DIR = OUTPUT_DIR / "platform_import"

INVENTORY_FILE = OUTPUT_DIR / "video_inventory_v1.jsonl"
TRIGGER_FILE = OUTPUT_DIR / "video_trigger_questions_v1.jsonl"
LINKAGE_FILE = OUTPUT_DIR / "video_kb_linkage_v1.jsonl"

CSV_VIDEO_QA = IMPORT_DIR / "import_video_qa.csv"
CSV_LINKAGE = IMPORT_DIR / "import_video_kb_linkage.csv"
MANIFEST = IMPORT_DIR / "video_route_manifest.json"

# 价值宣传类视频：不参与客服问答的主动视频附挂
CS_ROUTE_EXCLUDED = {"video-011"}

DURATION_FMT = "{:02d}:{:02d}"


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


def fmt_duration(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return DURATION_FMT.format(m, s)


def build_answer(inv):
    return (
        f"【视频讲解】{inv['title']}。{inv['summary']} "
        f"视频链接：{inv['video_url']}"
        f"（时长约 {fmt_duration(inv['duration_sec'])}，含字幕）"
    )


def main():
    errors, warnings = [], []

    if not INVENTORY_FILE.exists():
        raise RuntimeError(f"缺少 {INVENTORY_FILE}，先运行 62")
    if not TRIGGER_FILE.exists():
        raise RuntimeError(f"缺少 {TRIGGER_FILE}，先运行 63")

    inventory = {
        r["video_id"]: r for r in load_jsonl(INVENTORY_FILE)
    }
    trigger = {
        r["video_id"]: r for r in load_jsonl(TRIGGER_FILE)
    }

    if set(inventory) != set(trigger):
        errors.append(
            f"62/63 视频集合不一致: "
            f"inv-only={sorted(set(inventory) - set(trigger))} "
            f"trig-only={sorted(set(trigger) - set(inventory))}"
        )

    # ---- 视频 QA CSV ----
    rows = []
    seen_questions = {}
    for vid in sorted(trigger):
        inv = inventory.get(vid, trigger[vid])
        answer = build_answer(inv)
        if not inv.get("video_url"):
            errors.append(f"{vid}: video_url 为空")
        passed = [q for q in trigger[vid]["questions"]
                  if q["gate"] == "pass"]
        if not passed:
            warnings.append(f"{vid}: 无 gate=pass 触发问题")
        for q in passed:
            qn = re.sub(r"\s", "", q["question"])
            if qn in seen_questions:
                warnings.append(
                    f"{vid}: 问题跨视频重复 "
                    f"(与 {seen_questions[qn]} 相同): {q['question']}"
                )
                continue
            seen_questions[qn] = vid
            rows.append({
                "question": q["question"],
                "answer": answer,
                "video_id": vid,
                "category": inv.get("category", ""),
                "duration_sec": inv.get("duration_sec", ""),
                "video_url": inv["video_url"],
            })

    if not rows:
        errors.append("没有可导出的问题行")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_VIDEO_QA, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "video_id",
                        "category", "duration_sec", "video_url"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # ---- 视频↔KB 互挂 CSV（可选） ----
    linkage_written = False
    if LINKAGE_FILE.exists():
        video_links = []
        for line in load_jsonl(LINKAGE_FILE):
            if isinstance(line, dict) and line.get("type") == "video_kb_link":
                video_links.append(line)
        if video_links:
            with open(CSV_LINKAGE, "w", encoding="utf-8-sig",
                      newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["video_id", "kb_id", "similarity",
                                "video_question", "kb_question"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(video_links)
            linkage_written = True

    # ---- 服务路由清单 ----
    linked_kb = {}
    if linkage_written:
        for line in video_links:
            linked_kb.setdefault(line["video_id"], []).append(
                {"kb_id": line["kb_id"], "similarity": line["similarity"]}
            )

    manifest_videos = []
    for vid in sorted(inventory):
        inv = inventory[vid]
        passed = [q["question"] for q in trigger[vid]["questions"]
                  if q["gate"] == "pass"]
        manifest_videos.append({
            "video_id": vid,
            "title": inv["title"],
            "category": inv.get("category", ""),
            "summary": inv["summary"],
            "duration_sec": inv["duration_sec"],
            "duration_hms": fmt_duration(inv["duration_sec"]),
            "video_url": inv["video_url"],
            "transcript_url": inv["transcript_url"],
            "matched_document_id": inv.get("matched_document_id", ""),
            "include_in_cs_route": vid not in CS_ROUTE_EXCLUDED,
            "trigger_questions": passed,
            "linked_kb": linked_kb.get(vid, []),
        })

    manifest = {
        "version": "video_route_manifest_v1",
        "videos_total": len(manifest_videos),
        "cs_route_excluded": sorted(CS_ROUTE_EXCLUDED),
        "qa_csv_rows": len(rows),
        "videos": manifest_videos,
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"videos: {len(manifest_videos)}")
    print(f"video qa csv rows: {len(rows)} -> {CSV_VIDEO_QA.name}")
    print(f"linkage csv: {'written' if linkage_written else 'skipped (64 输出不存在)'}")
    print(f"manifest: {len(manifest_videos)} videos -> {MANIFEST.name}")
    print(f"cs_route_excluded: {sorted(CS_ROUTE_EXCLUDED)}")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(" -", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
