#!/usr/bin/env python3
"""
Step 18.1 - 过滤 2026-03-01 之后的聊天切片
============================================

用户决策 (2026-09-20): 重立基线后, 新知识条目只允许
来源 2026-03-01 之后的聊天切片。

确定性过滤, 不修改冻结输入:

    output/issue_candidates.jsonl (3606 条, 冻结)
      -> start_time >= 2026-03-01
      -> output/issue_candidates_after_20260301.jsonl (747 条)

守卫:

- 输入 3606 条, 全部带 start_time
- 输出全部满足 start_time >= 2026-03-01
- 输出条数与逐月分布写进 summary, 供人工核对
"""

import json
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATES_FILE = OUTPUT_DIR / "issue_candidates.jsonl"
FILTERED_FILE = OUTPUT_DIR / "issue_candidates_after_20260301.jsonl"

CUTOFF = "2026-03-01"
EXPECTED_TOTAL = 3606
EXPECTED_AFTER = 747


def main():
    records = []
    with open(CANDIDATES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) != EXPECTED_TOTAL:
        raise SystemExit(f"守卫失败: 输入 {len(records)} 条 != 冻结总数 {EXPECTED_TOTAL}")
    if any(not record.get("start_time") for record in records):
        raise SystemExit("守卫失败: 存在缺少 start_time 的切片")

    after = [record for record in records if record["start_time"] >= CUTOFF]
    if len(after) != EXPECTED_AFTER:
        raise SystemExit(f"守卫失败: 过滤结果 {len(after)} 条 != 预期 {EXPECTED_AFTER}")

    months = Counter(record["start_time"][:7] for record in after)
    with open(FILTERED_FILE, "w", encoding="utf-8") as f:
        for record in after:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"写出 {FILTERED_FILE.name}: {len(after)} 条")
    for month in sorted(months):
        print(f"  {month}: {months[month]}")


if __name__ == "__main__":
    main()
