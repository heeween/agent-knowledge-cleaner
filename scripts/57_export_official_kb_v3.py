#!/usr/bin/env python3
"""
Step 16.2 - 正式 KB v3 导出
===========================

official KB v2 (76 条) + singleton gate publish
(569 条) -> official KB v3 (645 条)。

全部内容逐字来自已冻结/已验证输出:

- v2 76 条原样继承 (payload 不变)
- singleton 569 条:
  question/answer 来自 singleton_candidates_v1
  (Step 7 抽取原文), verdict 溯源来自
  singleton_publishability_v1

守卫:

- 总数 76 + publish = 645
- kb_id 连续唯一 (KB-0001..)
- singleton issue_key 与 v2 cluster_id 空间无交集
- 发布前手机号全量复扫 (0 允许)
- publish 条目答案非空且 >= 20 字
- publish 无 guard_downgrades
  (前置标记 PRIVACY_PHONE / ANSWER_TOO_SHORT
   必须不在 pre_flags)

输出 (新文件, 不覆盖任何既有导出):

    output/kb_entries_official_v3.jsonl / .xlsx
"""

import json
import re
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

V2_FILE = OUTPUT_DIR / "kb_entries_official_v2.jsonl"
CANDIDATES_FILE = (
    OUTPUT_DIR / "singleton_candidates_v1.jsonl"
)
GATE_FILE = (
    OUTPUT_DIR / "singleton_publishability_v1.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR / "kb_entries_official_v3.jsonl"
)
OUTPUT_XLSX = (
    OUTPUT_DIR / "kb_entries_official_v3.xlsx"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)1[3-9]\d{9}(?!\d)"
)


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():

    v2 = load_jsonl(V2_FILE)
    candidates = {
        record["issue_key"]: record
        for record in load_jsonl(CANDIDATES_FILE)
    }
    gate = {
        record["issue_key"]: record
        for record in load_jsonl(GATE_FILE)
    }

    publish_keys = sorted(
        key
        for key, record in gate.items()
        if record["verdict_final"] == "publish"
    )

    # ---------- 守卫 ----------

    errors = []

    if len(v2) != 76:
        errors.append(
            f"v2 数量 {len(v2)} != 76"
        )

    v2_ids = {record["cluster_id"] for record in v2}

    candidate_keys = set(candidates.keys())

    if v2_ids & candidate_keys:
        errors.append("v2 与 singleton id 空间有交集")

    for key in publish_keys:
        gate_record = gate[key]
        candidate = candidates.get(key)

        if candidate is None:
            errors.append(f"{key} candidate 缺失")
            continue

        if gate_record["guard_downgrades"]:
            errors.append(
                f"{key} 带 guard_downgrades 却是 publish"
            )

        if gate_record["pre_flags"]:
            errors.append(
                f"{key} 带前置标记 {gate_record['pre_flags']}"
                f" 却是 publish"
            )

        answer = candidate["answer"]

        if len(answer.strip()) < 20:
            errors.append(f"{key} 答案过短")

        if PHONE_PATTERN.search(
            candidate["question"] + "\n" + answer
        ):
            errors.append(f"{key} 含手机号")

    if errors:
        for error in errors:
            print(f"导出守卫失败: {error}")

        raise SystemExit(1)

    # ---------- 组装 ----------

    envelope = []

    for record in v2:
        envelope.append(record)

    for key in publish_keys:
        candidate = candidates[key]
        gate_record = gate[key]

        envelope.append({
            "kb_id": "",
            "part": "singleton",
            "cluster_id": key,
            "question": candidate["question"],
            "answer": candidate["answer"],
            "source_issue_keys": [key],
            "provenance": (
                "singleton_publishability_v1.jsonl#publish"
            ),
            "payload": {
                **candidate,
                "gate_verdict": gate_record[
                    "verdict_final"
                ],
                "gate_reason": gate_record["reason"],
                "gate_notes": gate_record["notes"],
                "gate_self_contained": gate_record[
                    "self_contained"
                ],
                "gate_model": gate_record["model"],
            },
        })

    for index, record in enumerate(
        envelope, start=1
    ):
        record["kb_id"] = f"KB-{index:04d}"

    total = len(envelope)

    # ---------- 导出 ----------

    with open(
        OUTPUT_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in envelope:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    flat_rows = [
        {
            "kb_id": record["kb_id"],
            "part": record["part"],
            "cluster_id": record["cluster_id"],
            "question": record["question"],
            "answer": record["answer"],
            "provenance": record["provenance"],
        }
        for record in envelope
    ]

    singleton_rows = []

    for record in envelope:
        if record["part"] != "singleton":
            continue

        payload = record["payload"]

        singleton_rows.append({
            "kb_id": record["kb_id"],
            "issue_key": record["cluster_id"],
            "question": record["question"],
            "answer": record["answer"],
            "resolution": payload["resolution"],
            "temporal_status": payload[
                "temporal_status"
            ],
            "knowledge_value": payload[
                "knowledge_value"
            ],
            "issue_date": payload["issue_date"],
            "self_contained": payload[
                "gate_self_contained"
            ],
            "gate_reason": payload["gate_reason"],
            "funnel_stable_limitation": payload[
                "funnel_stable_limitation"
            ],
        })

    parts = pd.Series(
        [record["part"] for record in envelope]
    ).value_counts().to_dict()

    summary = {
        "official_kb_version": "v3",
        "total": total,
        **{f"part_{k}": int(v) for k, v in parts.items()},
        "previous_official_count": 76,
        "singleton_publish": len(publish_keys),
        "singleton_manual_review": sum(
            1 for r in gate.values()
            if r["verdict_final"] == "manual_review"
        ),
        "singleton_reject": sum(
            1 for r in gate.values()
            if r["verdict_final"] == "reject"
        ),
    }

    with pd.ExcelWriter(
        OUTPUT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame(flat_rows).to_excel(
            writer, sheet_name="all_entries", index=False
        )
        pd.DataFrame(singleton_rows).to_excel(
            writer,
            sheet_name="singleton_publish",
            index=False,
        )

    print("=" * 60)
    print("正式 KB v3 导出完成（Step 16.2）")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
