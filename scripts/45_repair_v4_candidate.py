#!/usr/bin/env python3
"""
Step 12.3.1 - v4 deterministic repair
=====================================

Step 11.5 v4 验证 (10 entries) 中出现两个 flag:

1. QCLUSTER-0029 needs_fix (hard INVENTED_PRODUCT_RULE)
   - 真缺陷 (v3 验证漏检, v4 抓到):
     summary_answer 说 "主要有两种路径",
     但 entry 自身 UNIT 2 就有 source 支撑的
     第三条路径 (管理员账号参照指引图片创建)
   - deterministic 修复: summary 重写为三条路径,
     clause 1/3 保留原验证文本,
     clause 2 使用 SOURCE 2 (0002446#ROW-02718)
     原文片段

2. QCLUSTER-0002 manual_review
   (medium INVENTED_PROCEDURE_STEP, minor)
   - unit 2 的步骤 "检查历史与本次保养记录的
     机油型号格式" 是推断步骤, 不在 SOURCE 5
     原文中 (validator 自评 minor 且与来源
     含义一致)
   - deterministic 修复: 删除该推断步骤,
     保留两条与 SOURCE 5 逐字对应的步骤

不调用 LLM。
v4 逐字保留, 其余 8 条 entry 不动。

输出:

    output/kb_entries_mixed_candidate_v5.jsonl
    output/kb_entries_mixed_candidate_v5.xlsx
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATE_V4 = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v4.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v5.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v5.xlsx"
)


# ============================================================
# 定向修复定义 (与 40 同模式: 显式、可审计)
# ============================================================

REPAIR_0029_SUMMARY_OLD = (
    "创建新账号主要有两种路径：一是通过权限中心自行创建，"
    "二是联系客服提供手机号由客服代为创建。"
)

REPAIR_0029_SUMMARY_NEW = (
    "创建新账号主要有三种路径：一是通过权限中心自行创建；"
    "二是使用管理员账号登录系统，"
    "参照系统提供的操作指引图片完成账号创建流程；"
    "三是联系客服提供手机号由客服代为创建。"
)

REPAIR_0002_STEP_DROP = (
    "检查历史与本次保养记录的机油型号格式"
)


def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float):
        if pd.isna(value):
            return ""

    return str(value).strip()


def file_md5(path):

    return hashlib.md5(path.read_bytes()).hexdigest()


def repair_entry(entry):

    """
    返回 (repaired_entry, repair_fields)。
    只允许改定义中列出的字段;
    前置文本不匹配直接 abort。
    """

    cluster_id = clean_text(entry.get("cluster_id"))

    repair_fields = []

    if cluster_id == "QCLUSTER-0029":

        old = clean_text(
            entry.get("summary_answer")
        )

        if old != REPAIR_0029_SUMMARY_OLD:

            raise RuntimeError(
                "QCLUSTER-0029 summary 与预期不符:\n"
                f"{old}"
            )

        entry["summary_answer"] = (
            REPAIR_0029_SUMMARY_NEW
        )

        repair_fields.append("summary_answer")

    elif cluster_id == "QCLUSTER-0002":

        units = entry.get("units") or []

        unit = units[1]

        steps = [
            clean_text(step)
            for step in unit.get("steps") or []
        ]

        if REPAIR_0002_STEP_DROP not in steps:

            raise RuntimeError(
                "QCLUSTER-0002 unit 2 未找到"
                f"待删除步骤: {steps}"
            )

        unit["steps"] = [
            step
            for step in steps
            if step != REPAIR_0002_STEP_DROP
        ]

        repair_fields.append(
            "units[1].steps"
        )

    return entry, repair_fields


def main():

    print("=" * 70)
    print("Step 12.3.1 - v4 deterministic repair")
    print("=" * 70)

    checksum_before = file_md5(CANDIDATE_V4)

    lines = [
        line
        for line in CANDIDATE_V4.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    repaired_lines = []

    repaired_ids = set()

    for line in lines:

        entry = json.loads(line)

        cluster_id = clean_text(
            entry.get("cluster_id")
        )

        if cluster_id in (
            "QCLUSTER-0029", "QCLUSTER-0002"
        ):

            entry, fields = repair_entry(entry)

            entry["repair_applied"] = True

            entry["repair_fields"] = fields

            entry["repair_reason"] = (
                "step_12_3_1_deterministic_repair"
            )

            repaired_ids.add(cluster_id)

            print()
            print(f"repaired {cluster_id}: {fields}")

            if cluster_id == "QCLUSTER-0029":

                print(
                    "  new summary: "
                    f"{entry['summary_answer']}"
                )

            if cluster_id == "QCLUSTER-0002":

                print(
                    "  unit 2 steps: "
                    + json.dumps(
                        entry["units"][1]["steps"],
                        ensure_ascii=False,
                    )
                )

            repaired_lines.append(
                json.dumps(
                    entry, ensure_ascii=False
                )
            )

            continue

        repaired_lines.append(line)

    if repaired_ids != {
        "QCLUSTER-0029", "QCLUSTER-0002"
    }:

        raise RuntimeError(
            f"修复目标缺失: {repaired_ids}"
        )

    OUTPUT_JSONL.write_text(
        "\n".join(repaired_lines) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 完整性校验
    # --------------------------------------------------------

    v4_by_id = {
        json.loads(line)["cluster_id"]: json.loads(line)
        for line in lines
    }

    v5_by_id = {
        json.loads(line)["cluster_id"]: json.loads(line)
        for line in repaired_lines
    }

    checks = []

    checks.append(
        {
            "check": "entry_count",
            "expected": 10,
            "actual": len(v5_by_id),
            "passed": len(v5_by_id) == 10,
        }
    )

    untouched_ok = True

    for cluster_id, entry in v4_by_id.items():

        if cluster_id in repaired_ids:
            continue

        if json.dumps(
            v5_by_id[cluster_id],
            ensure_ascii=False,
        ) != json.dumps(entry, ensure_ascii=False):
            untouched_ok = False

    checks.append(
        {
            "check": "other_8_entries_unchanged",
            "expected": True,
            "actual": untouched_ok,
            "passed": untouched_ok,
        }
    )

    e29 = v5_by_id["QCLUSTER-0029"]

    units_29_ok = (
        json.dumps(e29["units"], ensure_ascii=False)
        == json.dumps(
            v4_by_id["QCLUSTER-0029"]["units"],
            ensure_ascii=False,
        )
    )

    checks.append(
        {
            "check": "0029_units_unchanged",
            "expected": True,
            "actual": units_29_ok,
            "passed": units_29_ok,
        }
    )

    e02 = v5_by_id["QCLUSTER-0002"]

    unit2 = e02["units"][1]

    steps_ok = (
        REPAIR_0002_STEP_DROP
        not in unit2["steps"]
        and len(unit2["steps"]) == 2
    )

    content_ok = (
        unit2["content"]
        == v4_by_id["QCLUSTER-0002"]["units"][1][
            "content"
        ]
    )

    checks.append(
        {
            "check": "0002_only_steps_changed",
            "expected": True,
            "actual": (
                f"steps_ok={steps_ok} "
                f"content_ok={content_ok}"
            ),
            "passed": steps_ok and content_ok,
        }
    )

    checksum_after = file_md5(CANDIDATE_V4)

    frozen_ok = checksum_before == checksum_after

    checks.append(
        {
            "check": "v4_frozen_unmodified",
            "expected": True,
            "actual": frozen_ok,
            "passed": frozen_ok,
        }
    )

    failed = [
        c for c in checks if not c["passed"]
    ]

    if failed:

        for c in failed:
            print(f"[FAIL] {c['check']}: {c['actual']}")

        raise RuntimeError("完整性校验失败")

    # --------------------------------------------------------
    # 输出 xlsx
    # --------------------------------------------------------

    rows = []

    for entry in (
        json.loads(line)
        for line in repaired_lines
    ):

        rows.append(
            {
                "cluster_id": entry.get(
                    "cluster_id"
                ),
                "canonical_question": entry.get(
                    "canonical_question"
                ),
                "generation_mode": entry.get(
                    "generation_mode"
                ),
                "summary_answer": entry.get(
                    "summary_answer"
                ),
                "unit_count": entry.get(
                    "unit_count"
                ),
                "candidate_status": entry.get(
                    "candidate_status"
                ),
                "repair_applied": entry.get(
                    "repair_applied"
                ),
                "repair_fields": " | ".join(
                    entry.get("repair_fields") or []
                ),
            }
        )

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        pd.DataFrame(rows).to_excel(
            writer,
            sheet_name="entries",
            index=False,
        )

        pd.DataFrame(checks).to_excel(
            writer,
            sheet_name="repair_integrity",
            index=False,
        )

    print()
    print("=" * 70)
    print("完整性校验全部通过:")
    for check in checks:
        print(f"  [PASS] {check['check']}")
    print("=" * 70)
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Excel: {OUTPUT_XLSX}")
    print()
    print("下一步: 39 v5 验证")
    print(
        ".venv/bin/python "
        "scripts/39_validate_mixed_candidate_grounding.py "
        "--candidate output/kb_entries_mixed_candidate_v5.jsonl"
    )
    print()
    print(
        f"校验时间: "
        f"{datetime.now().isoformat(timespec='seconds')}"
    )


if __name__ == "__main__":
    main()
