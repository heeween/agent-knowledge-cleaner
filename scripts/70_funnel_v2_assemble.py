#!/usr/bin/env python3
"""
Step 18.2 - 组装 funnel v2 输入（2026-03-01 之后切片）
========================================================

来源分层 (全部可复用优先, 缺口才进 LLM):

  747 after 切片
    = 566 已在 singleton_funnel_v1  -> 778 行逐字复用
    + 61  抽取过但走了聚类路径       -> 元数据富集 worklist
      (extracted_issues.xlsx 有 QA, 缺 confidence/
       knowledge_value/temporal_status 等漏斗字段)
    + 120 完全未抽取                 -> 全量抽取 worklist
      (source_messages 取自候选切片文件)

输出 (全部新文件, 不修改冻结输入):

    output/funnel_v2_reused.jsonl            (778 行)
    output/funnel_v2_enrich_worklist.jsonl   (61 切片的行)
    output/funnel_v2_extract_worklist.jsonl  (120 切片)
"""

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATES_FILE = OUTPUT_DIR / "issue_candidates_after_20260301.jsonl"
FUNNEL_V1_FILE = OUTPUT_DIR / "singleton_funnel_v1.jsonl"
EXTRACTED_XLSX = OUTPUT_DIR / "extracted_issues.xlsx"

REUSED_FILE = OUTPUT_DIR / "funnel_v2_reused.jsonl"
ENRICH_FILE = OUTPUT_DIR / "funnel_v2_enrich_worklist.jsonl"
EXTRACT_FILE = OUTPUT_DIR / "funnel_v2_extract_worklist.jsonl"

EXPECTED_CANDIDATES = 747
EXPECTED_REUSED_SLICES = 566
EXPECTED_REUSED_ROWS = 778
EXPECTED_ENRICH_SLICES = 61
EXPECTED_EXTRACT_SLICES = 120

ENRICH_METADATA_FIELDS = [
    "confidence", "knowledge_value", "temporal_status",
    "resolution", "crm_module", "crm_feature",
]


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    candidates = load_jsonl(CANDIDATES_FILE)
    if len(candidates) != EXPECTED_CANDIDATES:
        raise SystemExit(f"守卫失败: after 切片 {len(candidates)} != {EXPECTED_CANDIDATES}")
    after_ids = {record["issue_id"] for record in candidates}

    # 1) 逐字复用层
    funnel_v1 = load_jsonl(FUNNEL_V1_FILE)
    reused = [row for row in funnel_v1 if row["issue_key"].split("#")[0] in after_ids]
    reused_slices = {row["issue_key"].split("#")[0] for row in reused}
    if len(reused) != EXPECTED_REUSED_ROWS or len(reused_slices) != EXPECTED_REUSED_SLICES:
        raise SystemExit(
            f"守卫失败: 复用层 {len(reused)} 行 / {len(reused_slices)} 切片 != "
            f"{EXPECTED_REUSED_ROWS} 行 / {EXPECTED_REUSED_SLICES} 切片"
        )
    with open(REUSED_FILE, "w", encoding="utf-8") as f:
        for row in reused:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"写出 {REUSED_FILE.name}: {len(reused)} 行 / {len(reused_slices)} 切片")

    # 2) 剩余切片
    remaining = after_ids - reused_slices
    extracted = pd.read_excel(EXTRACTED_XLSX, sheet_name="extracted_issues")
    extracted["source_candidate_id"] = extracted["source_candidate_id"].astype(str)
    by_slice = {sid: group for sid, group in extracted.groupby("source_candidate_id")}

    enrich_rows, extract_slices = [], []
    for slice_id in sorted(remaining):
        group = by_slice.get(slice_id)
        if group is None or group.empty:
            candidate = next(record for record in candidates if record["issue_id"] == slice_id)
            extract_slices.append(candidate)
            continue
        for _, row in group.iterrows():
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not question or not answer:
                continue
            issue_local = int(row.get("issue_local_index") or 0)
            enrich_rows.append({
                "issue_key": f"{slice_id}#EXT-{issue_local:04d}",
                "source_candidate_id": slice_id,
                "document_id": str(row.get("document_id") or ""),
                "source_filename": str(row.get("source_filename") or ""),
                "issue_date": next(
                    record["start_time"][:10] for record in candidates
                    if record["issue_id"] == slice_id
                ),
                "question": question,
                "question_normalized": str(row.get("question_normalized") or "").strip() or question,
                "answer": answer,
                "problem_type": str(row.get("problem_type") or "").strip(),
                **{field: None for field in ENRICH_METADATA_FIELDS},
            })

    enrich_slice_ids = {row["source_candidate_id"] for row in enrich_rows}
    if len(enrich_slice_ids) != EXPECTED_ENRICH_SLICES:
        raise SystemExit(
            f"守卫失败: 富集层 {len(enrich_slice_ids)} 切片 != {EXPECTED_ENRICH_SLICES}"
        )
    if len(extract_slices) != EXPECTED_EXTRACT_SLICES:
        raise SystemExit(
            f"守卫失败: 抽取层 {len(extract_slices)} 切片 != {EXPECTED_EXTRACT_SLICES}"
        )
    if len(reused_slices) | len(enrich_slice_ids) | len(extract_slices) != after_ids and \
            reused_slices | enrich_slice_ids | {s["issue_id"] for s in extract_slices} != after_ids:
        raise SystemExit("守卫失败: 三层切片并集 != after 全集")

    with open(ENRICH_FILE, "w", encoding="utf-8") as f:
        for row in enrich_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with open(EXTRACT_FILE, "w", encoding="utf-8") as f:
        for record in extract_slices:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"写出 {ENRICH_FILE.name}: {len(enrich_rows)} 行 / {len(enrich_slice_ids)} 切片")
    print(f"写出 {EXTRACT_FILE.name}: {len(extract_slices)} 切片")
    print("三层覆盖校验通过: 566 + 61 + 120 = 747")


if __name__ == "__main__":
    main()
