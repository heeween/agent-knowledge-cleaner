#!/usr/bin/env python3
"""
Step 11.6 - Cross-entry dedup / merge decision
==============================================

处理 frozen Step 8.4 判定 same_intent 的
跨 entry 重复对:

    QCLUSTER-0021  vs  QCLUSTER-0066
    similarity 0.8942 / 0.8921
    entry centroid cosine 0.9175

两条 entry 内容级 grounding 在 Step 11.5 v2
已经全部干净 (summary / units / notes 全 grounded，
conflict_wrongly_claimed = false)。

唯一 hard flag 是 CROSS_ENTRY_SAME_INTENT，
所以两条不能分别发布。

本脚本做 deterministic merge:

- survivor: QCLUSTER-0021
  选择规则 (可复算):
    1. usable source 多者优先 (3 > 2)
    2. 平局时 knowledge_confidence 高者优先 (0.9 > 0.75)
- absorbed: QCLUSTER-0066
- units: 两个 parent 的 unit 逐字保留
  (0021 的 3 个 cause + 0066 的 2 个 cause)
- summary_answer: 由两个 parent 已通过 v2 验证的
  summary 子句 deterministic 重编号拼装
- notes / limitations: deterministic 合并，
  并为 partial / temporary / unresolved source
  补齐时效提示 (消除 TEMPORAL_CAUTION_MISSING)

不调用 LLM。
不修改任何 frozen 输入。
输出新文件:

    output/kb_entries_mixed_candidate_v3.jsonl
    output/kb_entries_mixed_candidate_v3.xlsx
    output/cross_entry_merge_decisions.jsonl

不支持 merge 以外的决策路径: primary-secondary
或 drop 会丢失 0066 已通过 grounding 的 2 个
cause unit，与"保留 grounded knowledge"原则冲突。
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATE_V2 = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v2.jsonl"
)

VALIDATION_V2 = (
    OUTPUT_DIR
    / "kb_entry_grounding_validations_mixed_v2.jsonl"
)

GROUNDING_FILE = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

STRUCTURE_FILE = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.xlsx"
)

PAIR_FILE = (
    OUTPUT_DIR
    / "pair_classifications.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v3.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v3.xlsx"
)

DECISION_JSONL = (
    OUTPUT_DIR
    / "cross_entry_merge_decisions.jsonl"
)

DRY_RUN = "--dry-run" in sys.argv

# 与 39 的 has_temporal_caution 保持一致
CAUTION_SOURCE_STATUSES = {"partial", "unresolved"}


# ============================================================
# 工具
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float):
        if pd.isna(value):
            return ""

    return str(value).strip()


def file_md5(path):

    return hashlib.md5(path.read_bytes()).hexdigest()


def load_jsonl(path):

    records = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def parse_summary_clauses(summary, expected_count):

    """
    从 parent 已验证的 summary 中
    deterministic 抽取原因子句。

    0021 格式:
      "...主要有三个原因：1. xxx；2. yyy；3. zzz。"
    0066 格式:
      "原因一：xxx；原因二：yyy。"

    抽取失败直接 abort，不做任何猜测。
    """

    text = clean_text(summary)

    body = text.split("：", 1)[1] if "：" in text else text

    body = body.strip().rstrip("。")

    clauses = []

    for chunk in body.split("；"):

        chunk = chunk.strip()

        chunk = re.sub(r"^[0-9]+\.\s*", "", chunk)

        chunk = re.sub(r"^原因[一二三四五六七八九十]+[:：]\s*", "", chunk)

        if chunk:
            clauses.append(chunk)

    if len(clauses) != expected_count:

        raise RuntimeError(
            f"summary 子句解析失败: 期望 {expected_count}，"
            f"实际 {len(clauses)}: {clauses}"
        )

    return clauses


# ============================================================
# 输入加载与前置校验
# ============================================================

def load_and_verify():

    inputs = [
        CANDIDATE_V2,
        VALIDATION_V2,
        GROUNDING_FILE,
        STRUCTURE_FILE,
        PAIR_FILE,
    ]

    for path in inputs:

        if not path.exists():
            raise FileNotFoundError(path)

    checksums = {
        path.name: file_md5(path)
        for path in inputs
    }

    entries = load_jsonl(CANDIDATE_V2)

    validations = {
        clean_text(v.get("cluster_id")): v
        for v in load_jsonl(VALIDATION_V2)
    }

    by_id = {
        clean_text(e.get("cluster_id")): e
        for e in entries
    }

    left_id = "QCLUSTER-0021"

    right_id = "QCLUSTER-0066"

    for cluster_id in (left_id, right_id):

        if cluster_id not in by_id:
            raise RuntimeError(f"缺少 entry: {cluster_id}")

        if cluster_id not in validations:
            raise RuntimeError(
                f"缺少 v2 validation: {cluster_id}"
            )

    frozen_pairs = load_jsonl(PAIR_FILE)

    return (
        entries,
        by_id,
        validations,
        frozen_pairs,
        checksums,
    )


def collect_cross_entry_pairs(
    frozen_pairs,
    entry_a,
    entry_b,
):

    """
    重算两个 entry 之间的跨条目 same_intent 证据，
    必须与 v2 validation 中记录的 flags 一致。
    """

    keys_a = set()

    keys_b = set()

    for entry, keys in (
        (entry_a, keys_a),
        (entry_b, keys_b),
    ):

        for unit in entry.get("units") or []:

            for key in unit.get("source_issue_keys") or []:
                keys.add(clean_text(key))

    hits = []

    for pair in frozen_pairs:

        key_a = clean_text(pair.get("issue_key_a"))

        key_b = clean_text(pair.get("issue_key_b"))

        in_a = key_a in keys_a or key_b in keys_a

        in_b = key_a in keys_b or key_b in keys_b

        if not (in_a and in_b):
            continue

        if clean_text(pair.get("relation")) != "same_intent":
            continue

        hits.append(
            {
                "pair_key": clean_text(pair.get("pair_key")),
                "similarity": pair.get("similarity"),
                "question_a": clean_text(
                    pair.get("question_a")
                ),
                "question_b": clean_text(
                    pair.get("question_b")
                ),
            }
        )

    return hits


def verify_preconditions(
    entry_a,
    entry_b,
    validation_a,
    validation_b,
    pair_hits,
):

    errors = []

    for entry in (entry_a, entry_b):

        if clean_text(
            entry.get("candidate_status")
        ) != "candidate_ok":

            errors.append(
                f"{entry.get('cluster_id')} "
                "candidate_status != candidate_ok"
            )

        if entry.get("hard_flags"):

            errors.append(
                f"{entry.get('cluster_id')} "
                "存在 hard_flags，不属于纯跨条目重复"
            )

    for validation in (validation_a, validation_b):

        if validation.get("conflict_wrongly_claimed"):
            errors.append(
                f"{validation.get('cluster_id')} "
                "conflict_wrongly_claimed = true，"
                "需要先处理 conflict，不能直接 merge"
            )

        hard_flags = [
            item
            for item in validation.get("validation_flags") or []
            if item.get("severity") == "hard"
        ]

        non_cross = [
            item
            for item in hard_flags
            if item.get("flag")
            != "CROSS_ENTRY_SAME_INTENT"
        ]

        if non_cross:
            errors.append(
                f"{validation.get('cluster_id')} "
                "存在非 cross-entry 的 hard flag: "
                f"{[item.get('flag') for item in non_cross]}"
            )

    if not pair_hits:

        errors.append(
            "frozen pair_classifications 中找不到"
            "两条 entry 之间的 same_intent pair"
        )

    if errors:

        raise RuntimeError(
            "前置校验失败:\n- " + "\n- ".join(errors)
        )


def load_usable_sources_by_cluster():

    df = pd.read_excel(
        GROUNDING_FILE,
        sheet_name="all_sources",
    )

    usable = df[df["usable_for_kb"] == True]

    by_cluster = {}

    for cluster_id, group in usable.groupby(
        usable["cluster_id"].astype(str)
    ):

        by_cluster[cluster_id] = group.to_dict(
            orient="records"
        )

    return by_cluster


def load_gate_cause_counts():

    df = pd.read_excel(
        STRUCTURE_FILE,
        sheet_name="all_clusters",
    )

    counts = {}

    for row in df.to_dict(orient="records"):

        counts[clean_text(row.get("cluster_id"))] = int(
            row.get("independent_cause_count") or 0
        )

    return counts


# ============================================================
# Merge 构造（全部 deterministic）
# ============================================================

def select_survivor(entry_a, entry_b):

    """
    survivor 选择规则:
      1. usable_source_count 多者优先
      2. 平局时 knowledge_confidence 高者优先
    """

    candidates = sorted(
        (entry_a, entry_b),
        key=lambda e: (
            int(e.get("usable_source_count") or 0),
            float(e.get("knowledge_confidence") or 0),
        ),
        reverse=True,
    )

    return candidates[0], candidates[1]


def build_merged_units(survivor, absorbed):

    merged_units = []

    for parent in (survivor, absorbed):

        for unit in parent.get("units") or []:

            merged_units.append(
                {
                    "unit_type": clean_text(
                        unit.get("unit_type")
                    ),
                    "title": unit.get("title"),
                    "condition": unit.get("condition"),
                    "content": unit.get("content"),
                    "steps": list(
                        unit.get("steps") or []
                    ),
                    "source_issue_keys": list(
                        unit.get("source_issue_keys")
                        or []
                    ),
                }
            )

    return merged_units


def build_merged_summary(
    survivor,
    absorbed,
    merged_unit_count,
):

    clauses = parse_summary_clauses(
        survivor.get("summary_answer"),
        int(survivor.get("unit_count") or 0),
    )

    clauses = clauses + parse_summary_clauses(
        absorbed.get("summary_answer"),
        int(absorbed.get("unit_count") or 0),
    )

    if len(clauses) != merged_unit_count:

        raise RuntimeError(
            "合并 summary 子句数与 unit 数不一致"
        )

    numbered = "；".join(
        f"{index}. {clause}"
        for index, clause in enumerate(
            clauses, start=1
        )
    )

    intro = (
        "已回访的工单或客户仍出现在回访列表中，"
        f"主要有以下{merged_unit_count}个原因："
    )

    return intro + numbered + "。"


def build_risky_source_cautions(
    source_rows,
    merged_units,
):

    """
    为每个 risky source (partial / unresolved /
    temporary) 找到引用它的 merged unit，
    deterministic 生成时效提示。
    """

    cautions = []

    for row in source_rows:

        resolution = clean_text(
            row.get("resolution")
        )

        temporal = clean_text(
            row.get("temporal_status")
        )

        if (
            resolution not in CAUTION_SOURCE_STATUSES
            and temporal != "temporary"
        ):
            continue

        key = clean_text(row.get("issue_key"))

        unit_index = None

        for index, unit in enumerate(
            merged_units, start=1
        ):

            if key in (
                unit.get("source_issue_keys") or []
            ):
                unit_index = index

                break

        if unit_index is None:

            raise RuntimeError(
                f"risky source 未被任何 unit 引用: {key}"
            )

        title = clean_text(
            merged_units[unit_index - 1].get("title")
        )

        reasons = []

        if resolution in CAUTION_SOURCE_STATUSES:
            reasons.append(
                f"resolution={resolution}"
            )

        if temporal == "temporary":
            reasons.append(
                "temporal_status=temporary"
            )

        cautions.append(
            {
                "unit_index": unit_index,
                "text": (
                    f"原因{unit_index}（{title}）对应 "
                    f"source（{key}）的 "
                    f"{'/'.join(reasons)}，"
                    "该解释仅部分确认或依赖当前系统版本，"
                    "可能随后续处理或优化而变化。"
                ),
            }
        )

    cautions.sort(key=lambda item: item["unit_index"])

    return [item["text"] for item in cautions]


def build_merged_entry(
    survivor,
    absorbed,
    source_rows,
    gate_cause_counts,
    pair_hits,
    validation_a,
    validation_b,
):

    merged_unit_count = (
        int(survivor.get("unit_count") or 0)
        + int(absorbed.get("unit_count") or 0)
    )

    gate_expected = (
        gate_cause_counts.get(
            clean_text(survivor.get("cluster_id")), 0
        )
        + gate_cause_counts.get(
            clean_text(absorbed.get("cluster_id")), 0
        )
    )

    if merged_unit_count != gate_expected:

        raise RuntimeError(
            f"merged unit 数 {merged_unit_count} != "
            f"gate independent_cause_count 之和 "
            f"{gate_expected}"
        )

    merged_units = build_merged_units(
        survivor, absorbed
    )

    summary_answer = build_merged_summary(
        survivor,
        absorbed,
        merged_unit_count,
    )

    notes = [
        (
            f"以上{merged_unit_count}个原因"
            "（"
            + "；".join(
                clean_text(unit.get("title"))
                for unit in merged_units
            )
            + "）是并列的独立情况，并非冲突关系。"
        )
    ]

    limitations = build_risky_source_cautions(
        source_rows,
        merged_units,
    )

    source_keys = [
        clean_text(row.get("issue_key"))
        for row in source_rows
    ]

    risky_count = sum(
        1
        for row in source_rows
        if clean_text(row.get("resolution"))
        in CAUTION_SOURCE_STATUSES
        or clean_text(row.get("temporal_status"))
        == "temporary"
    )

    avg_grounding = round(
        (
            float(survivor.get("avg_lexical_grounding") or 0)
            * int(survivor.get("usable_source_count") or 0)
            + float(absorbed.get("avg_lexical_grounding") or 0)
            * int(absorbed.get("usable_source_count") or 0)
        )
        / len(source_rows),
        4,
    )

    merged_entry = {
        "cluster_id": clean_text(
            survivor.get("cluster_id")
        ),
        "canonical_question": survivor.get(
            "canonical_question"
        ),
        "question_variants": [
            absorbed.get("canonical_question")
        ],
        "knowledge_structure": survivor.get(
            "knowledge_structure"
        ),
        "generation_mode": survivor.get(
            "generation_mode"
        ),
        "summary_answer": summary_answer,
        "units": merged_units,
        "unit_count": merged_unit_count,
        "notes": notes,
        "limitations": limitations,
        "crm_module": survivor.get("crm_module"),
        "crm_feature": survivor.get("crm_feature"),
        "problem_type": survivor.get("problem_type"),
        "knowledge_confidence": min(
            float(
                survivor.get("knowledge_confidence")
                or 0
            ),
            float(
                absorbed.get("knowledge_confidence")
                or 0
            ),
        ),
        "source_issue_keys": source_keys,
        "usable_source_count": len(source_rows),
        "cited_source_count": len(source_keys),
        "risky_source_count": risky_count,
        "min_lexical_grounding": min(
            float(
                survivor.get("min_lexical_grounding")
                or 0
            ),
            float(
                absorbed.get("min_lexical_grounding")
                or 0
            ),
        ),
        "avg_lexical_grounding": avg_grounding,
        "candidate_status": "candidate_ok",
        "hard_flag_count": 0,
        "hard_flags": [],
        "soft_flags": [],
        "structural_flags": [],
        "request_seconds": 0.0,
        "recheck_applied": False,
        "repair_applied": False,
        "repair_fields": [],
        "repair_reason": "",
        "repair_source_status": (
            "merged_in_step_11_6"
        ),
        "validator_false_positive_reviewed": False,
        "validator_false_positive_reason": "",
        "merged_from_clusters": [
            clean_text(absorbed.get("cluster_id"))
        ],
        "merge_decision": {
            "step": "11.6",
            "decision": "merge",
            "survivor": clean_text(
                survivor.get("cluster_id")
            ),
            "absorbed": clean_text(
                absorbed.get("cluster_id")
            ),
            "selection_rule": (
                "usable_source_count 多者优先，"
                "平局取 knowledge_confidence 高者"
            ),
            "frozen_same_intent_pairs": pair_hits,
            "entry_centroid_cosine": 0.9175,
            "conflict_check": (
                "两个 parent 的 v2 validation "
                "conflict_wrongly_claimed 均为 false；"
                "5 个 cause 条件互不相同，"
                "不存在同一条件下的互斥结论"
            ),
            "content_rule": (
                "units 逐字保留；summary 子句取自"
                "两个 parent 已通过 v2 验证的 "
                "summary；notes 由 unit titles "
                "deterministic 生成；limitations "
                "由 risky source 元数据 deterministic 生成"
            ),
            "validated_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "script": "scripts/41_merge_cross_entry_duplicates.py",
        },
    }

    return merged_entry


# ============================================================
# 输出
# ============================================================

def export_excel(records, decision_record):

    rows = []

    for entry in records:

        rows.append(
            {
                "cluster_id": entry.get("cluster_id"),
                "canonical_question": entry.get(
                    "canonical_question"
                ),
                "question_variants": " | ".join(
                    entry.get("question_variants") or []
                ),
                "knowledge_structure": entry.get(
                    "knowledge_structure"
                ),
                "generation_mode": entry.get(
                    "generation_mode"
                ),
                "summary_answer": entry.get(
                    "summary_answer"
                ),
                "unit_count": entry.get("unit_count"),
                "usable_source_count": entry.get(
                    "usable_source_count"
                ),
                "risky_source_count": entry.get(
                    "risky_source_count"
                ),
                "candidate_status": entry.get(
                    "candidate_status"
                ),
                "hard_flags": " | ".join(
                    entry.get("hard_flags") or []
                ),
                "soft_flags": " | ".join(
                    entry.get("soft_flags") or []
                ),
                "limitations": " | ".join(
                    entry.get("limitations") or []
                ),
                "source_issue_keys": " | ".join(
                    entry.get("source_issue_keys") or []
                ),
                "merged_from_clusters": " | ".join(
                    entry.get("merged_from_clusters")
                    or []
                ),
            }
        )

    entries_df = pd.DataFrame(rows)

    decision = decision_record

    decision_df = pd.DataFrame(
        [
            {
                "field": key,
                "value": json.dumps(
                    value, ensure_ascii=False
                )
                if isinstance(value, (dict, list))
                else value,
            }
            for key, value in decision.items()
        ]
    )

    integrity_df = pd.DataFrame(
        decision.get("integrity_checks")
    )

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        entries_df.to_excel(
            writer,
            sheet_name="entries",
            index=False,
        )

        decision_df.to_excel(
            writer,
            sheet_name="merge_decision",
            index=False,
        )

        integrity_df.to_excel(
            writer,
            sheet_name="merge_integrity",
            index=False,
        )


def verify_outputs(
    v2_entries_by_id,
    v2_lines_by_id,
    merged_entry,
):

    checks = []

    v3_records = load_jsonl(OUTPUT_JSONL)

    v3_by_id = {
        clean_text(e.get("cluster_id")): e
        for e in v3_records
    }

    checks.append(
        {
            "check": "entry_count",
            "expected": "9 (10 - 1 absorbed)",
            "actual": len(v3_records),
            "passed": len(v3_records) == 9,
        }
    )

    checks.append(
        {
            "check": "absorbed_entry_removed",
            "expected": "QCLUSTER-0066 不存在",
            "actual": (
                "QCLUSTER-0066" in v3_by_id
            ),
            "passed": (
                "QCLUSTER-0066" not in v3_by_id
            ),
        }
    )

    untouched_ok = True

    survivor_id = clean_text(
        merged_entry.get("cluster_id")
    )

    absorbed_id = (
        merged_entry.get("merge_decision") or {}
    ).get("absorbed", "")

    for cluster_id, entry in v2_entries_by_id.items():

        if cluster_id in (survivor_id, absorbed_id):
            continue

        original_text = v2_lines_by_id[cluster_id]

        current_text = json.dumps(
            v3_by_id.get(cluster_id),
            ensure_ascii=False,
        )

        original_norm = json.dumps(
            entry, ensure_ascii=False
        )

        if current_text != original_norm:
            untouched_ok = False

    checks.append(
        {
            "check": "other_8_entries_unchanged",
            "expected": "8 条 entry 内容字段完全一致",
            "actual": untouched_ok,
            "passed": untouched_ok,
        }
    )

    parent_units = []

    for cluster_id in (
        "QCLUSTER-0021",
        "QCLUSTER-0066",
    ):

        for unit in v2_entries_by_id[
            cluster_id
        ].get("units") or []:
            parent_units.append(unit)

    units_ok = len(
        merged_entry.get("units") or []
    ) == len(parent_units)

    if units_ok:

        for merged_unit, parent_unit in zip(
            merged_entry.get("units") or [],
            parent_units,
        ):

            for field in (
                "title",
                "condition",
                "content",
                "steps",
                "source_issue_keys",
            ):

                if (
                    merged_unit.get(field)
                    != parent_unit.get(field)
                ):
                    units_ok = False

    checks.append(
        {
            "check": "units_verbatim",
            "expected": (
                "5 个 unit 与 parent unit 逐字一致"
            ),
            "actual": units_ok,
            "passed": units_ok,
        }
    )

    failed = [
        item for item in checks if not item["passed"]
    ]

    if failed:

        for item in failed:

            print(
                f"[FAIL] {item['check']}: "
                f"expected={item['expected']} "
                f"actual={item['actual']}"
            )

        raise RuntimeError(
            "输出完整性校验失败，详见上方 FAIL"
        )

    return checks


# ============================================================
# 主流程
# ============================================================

def main():

    print("=" * 70)
    print("Step 11.6 - Cross-entry dedup / merge")
    print("QCLUSTER-0021 vs QCLUSTER-0066")
    print("=" * 70)

    if DRY_RUN:
        print("DRY RUN - 只打印计划，不写任何输出")

    (
        entries,
        by_id,
        validations,
        frozen_pairs,
        checksums,
    ) = load_and_verify()

    entry_a = by_id["QCLUSTER-0021"]

    entry_b = by_id["QCLUSTER-0066"]

    pair_hits = collect_cross_entry_pairs(
        frozen_pairs,
        entry_a,
        entry_b,
    )

    print()
    print("frozen same_intent pairs:")
    for hit in pair_hits:
        print(
            f"  {hit['similarity']:.4f}  "
            f"{hit['question_a']} / {hit['question_b']}"
        )

    verify_preconditions(
        entry_a,
        entry_b,
        validations["QCLUSTER-0021"],
        validations["QCLUSTER-0066"],
        pair_hits,
    )

    print()
    print("前置校验通过:")
    print("  - 两条 entry 均无内容级 hard flag")
    print("  - conflict_wrongly_claimed 均为 false")
    print("  - frozen same_intent 证据存在")

    survivor, absorbed = select_survivor(
        entry_a, entry_b
    )

    print()
    print(
        f"survivor = {survivor.get('cluster_id')} "
        f"(sources={survivor.get('usable_source_count')}, "
        f"confidence="
        f"{survivor.get('knowledge_confidence')})"
    )
    print(
        f"absorbed = {absorbed.get('cluster_id')} "
        f"(sources={absorbed.get('usable_source_count')}, "
        f"confidence="
        f"{absorbed.get('knowledge_confidence')})"
    )

    sources_by_cluster = load_usable_sources_by_cluster()

    source_rows = (
        sources_by_cluster.get(
            clean_text(survivor.get("cluster_id")), []
        )
        + sources_by_cluster.get(
            clean_text(absorbed.get("cluster_id")), []
        )
    )

    if len(source_rows) != 5:

        raise RuntimeError(
            f"usable source 并集应为 5，实际 "
            f"{len(source_rows)}"
        )

    gate_cause_counts = load_gate_cause_counts()

    merged_entry = build_merged_entry(
        survivor,
        absorbed,
        source_rows,
        gate_cause_counts,
        pair_hits,
        validations["QCLUSTER-0021"],
        validations["QCLUSTER-0066"],
    )

    print()
    print("merged entry:")
    print(
        f"  cluster_id = "
        f"{merged_entry.get('cluster_id')} "
        f"(absorbs "
        f"{merged_entry.get('merged_from_clusters')[0]})"
    )
    print(
        f"  units = {merged_entry.get('unit_count')} "
        f"(gate 3 + 2)"
    )
    print(
        f"  sources = "
        f"{merged_entry.get('usable_source_count')} "
        f"(risky "
        f"{merged_entry.get('risky_source_count')})"
    )
    print()
    print(f"summary: {merged_entry.get('summary_answer')}")
    print()
    print("notes:")
    for note in merged_entry.get("notes") or []:
        print(f"  - {note}")
    print("limitations:")
    for item in merged_entry.get("limitations") or []:
        print(f"  - {item}")

    if DRY_RUN:
        print()
        print("DRY RUN 结束，未写任何输出。")
        return

    v2_lines = CANDIDATE_V2.read_text(
        encoding="utf-8"
    ).splitlines()

    v2_lines_by_id = {}

    v2_entries_by_id = {}

    for line in v2_lines:

        if not line.strip():
            continue

        obj = json.loads(line)

        cluster_id = clean_text(obj.get("cluster_id"))

        v2_entries_by_id[cluster_id] = obj

        v2_lines_by_id[cluster_id] = json.dumps(
            obj, ensure_ascii=False
        )

    output_lines = []

    for cluster_id, line_text in v2_lines_by_id.items():

        if cluster_id == clean_text(
            absorbed.get("cluster_id")
        ):
            continue

        if cluster_id == clean_text(
            survivor.get("cluster_id")
        ):
            output_lines.append(
                json.dumps(
                    merged_entry, ensure_ascii=False
                )
            )

            continue

        output_lines.append(line_text)

    OUTPUT_JSONL.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8",
    )

    decision_record = dict(
        merged_entry.get("merge_decision") or {}
    )

    integrity_checks = verify_outputs(
        v2_entries_by_id,
        v2_lines_by_id,
        merged_entry,
    )

    decision_record["integrity_checks"] = (
        integrity_checks
    )

    decision_record["input_checksums"] = checksums

    for path in [
        CANDIDATE_V2,
        VALIDATION_V2,
        GROUNDING_FILE,
        STRUCTURE_FILE,
        PAIR_FILE,
    ]:

        current = file_md5(path)

        if checksums[path.name] != current:

            raise RuntimeError(
                f"frozen 输入被修改: {path.name}"
            )

    decision_record["frozen_inputs_unmodified"] = True

    DECISION_JSONL.write_text(
        json.dumps(
            decision_record, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    export_excel(
        load_jsonl(OUTPUT_JSONL),
        decision_record,
    )

    print()
    print("=" * 70)
    print("完整性校验全部通过:")
    for item in integrity_checks:
        print(f"  [PASS] {item['check']}")
    print("=" * 70)
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Excel: {OUTPUT_XLSX}")
    print(f"决策记录: {DECISION_JSONL}")
    print()
    print("下一步 (Step 11.6.1): 39 v3 revalidation")
    print(
        ".venv/bin/python "
        "scripts/39_validate_mixed_candidate_grounding.py "
        "--candidate output/kb_entries_mixed_candidate_v3.jsonl"
    )


if __name__ == "__main__":
    main()
