"""
Step 11.3 v3.1 - Scenario Grounding Revalidation

目的：

scripts/36_classify_mixed_knowledge_structure_v3.py
的 validate_scenarios 之前只统计
role == "scenario" 的 evidence，

导致 QCLUSTER-0004 这类
"role=procedure 但带 grounded scenario_label"
的 cluster 被误判为
SCENARIO_GROUNDING_FAIL 而 blocked。

本脚本：

- 不调用任何 LLM
- 不修改、不覆盖 v3 输出
- 直接读取 output/mixed_structure_classifications_v3.jsonl
- 用已修复的 hard rules 重新做 deterministic 推导
- 输出 output/mixed_structure_classifications_v3_1.jsonl / .xlsx

推导原则：

apply_hard_rules 只会把
generation_ready 从 true 改成 false，
永远不会反向放宽。

因此可以从最宽松基线
（generation_ready = true，
mode = 结构对应的标准 mode）
重新跑一遍 hard rules，
得到 deterministic 的最终状态。

任何无法用
"scenario grounding 修复"
解释的变化，
都会标记为
changed_review_required，
不会静默通过。
"""

import json
import importlib.util
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "output"

SCRIPTS_DIR = ROOT_DIR / "scripts"

CLASSIFIER_SCRIPT = (
    SCRIPTS_DIR
    / "36_classify_mixed_knowledge_structure_v3.py"
)

MIXED_AUDIT_FILE = (
    OUTPUT_DIR
    / "mixed_cluster_audits_v2.xlsx"
)

GROUNDING_FILE = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

V3_JSONL = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.xlsx"
)


# 与 36 的 hard rule 10 保持一致
EXPECTED_MODES = {
    "direct_synthesis": "single_answer",
    "structured_howto": "howto_sections",
    "multiple_causes": "cause_items",
    "scenario_branches": "scenario_sections",
    "troubleshooting": "troubleshooting_steps",
}

SCENARIO_FLAG_PREFIX = "SCENARIO_GROUNDING_FAIL"


def load_classifier_module():

    spec = importlib.util.spec_from_file_location(
        "mixed_structure_classifier_v3",
        CLASSIFIER_SCRIPT,
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float):
        if pd.isna(value):
            return ""

    return str(value).strip()


def load_v3_records():

    if not V3_JSONL.exists():
        raise FileNotFoundError(
            f"缺少 v3 输出: {V3_JSONL}"
        )

    records = []

    with V3_JSONL.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    if not records:
        raise RuntimeError("v3 JSONL 为空")

    return records


def load_usable_sources():

    grounding_df = pd.read_excel(
        GROUNDING_FILE,
        sheet_name="all_sources",
    )

    usable_df = (
        grounding_df[
            grounding_df["usable_for_kb"] == True
        ]
        .copy()
    )

    sources_by_cluster = {}

    for cluster_id, group in usable_df.groupby(
        usable_df["cluster_id"].astype(str)
    ):

        sources_by_cluster[cluster_id] = (
            group.to_dict(orient="records")
        )

    return sources_by_cluster


def rebuild_baseline(obj):

    """
    从 v3 记录还原 hard rules 之前的宽松基线。
    """

    hard_flags = set(
        obj.get("hard_rule_flags") or []
    )

    risk_flags = [
        flag
        for flag in (obj.get("risk_flags") or [])
        if flag not in hard_flags
    ]

    structure = obj["knowledge_structure"]

    payload = {
        "knowledge_structure": structure,
        "confidence": obj["structure_confidence"],
        "canonical_question": obj["canonical_question"],
        "evidence_roles": obj.get("evidence_roles") or [],
        "independent_cause_count": obj[
            "independent_cause_count"
        ],
        "scenario_count": obj["scenario_count"],
        "has_true_conflict": obj["has_true_conflict"],
        "conflict_facts": obj.get("conflict_facts") or [],
        "temporal_dependency": obj["temporal_dependency"],
        "current_validity_resolved": obj[
            "current_validity_resolved"
        ],
        "generation_ready": True,
        "recommended_generation_mode": EXPECTED_MODES.get(
            structure,
            "do_not_generate",
        ),
        "reason": obj["reason"],
        "risk_flags": risk_flags,
    }

    return payload


def state_of(
    structure,
    ready,
    mode,
    conflict,
    flags,
):

    return {
        "knowledge_structure": structure,
        "generation_ready": bool(ready),
        "recommended_generation_mode": mode,
        "has_true_conflict": bool(conflict),
        "hard_rule_flags": sorted(flags or []),
    }


def diff_states(before, after):

    diffs = []

    for key in before:

        if before[key] != after[key]:

            diffs.append(
                {
                    "field": key,
                    "before": before[key],
                    "after": after[key],
                }
            )

    return diffs


def classify_change(
    stored_flags,
    diffs,
    after,
):

    if not diffs:
        return "unchanged", ""

    scenario_only = bool(stored_flags) and all(
        flag.startswith(SCENARIO_FLAG_PREFIX)
        for flag in stored_flags
    )

    recovered = all(
        item["field"] in
        {
            "generation_ready",
            "recommended_generation_mode",
            "hard_rule_flags",
        }
        for item in diffs
    )

    if (
        scenario_only
        and recovered
        and after["generation_ready"]
        and not after["hard_rule_flags"]
    ):

        return (
            "unblocked_scenario_grounding",
            "stored hard flags 全部来自 "
            "role 过滤导致的 scenario grounding 误判，"
            "修复后重新推导通过",
        )

    return (
        "changed_review_required",
        "; ".join(
            f"{item['field']}: "
            f"{item['before']} -> {item['after']}"
            for item in diffs
        ),
    )


def build_scenario_grounding_rows(
    module,
    obj,
    source_rows,
):

    source_map = {
        clean_text(row.get("issue_key")): row
        for row in source_rows
    }

    rows = []

    for item in obj.get("evidence_roles") or []:

        label = clean_text(item.get("scenario_label"))

        evidence = clean_text(
            item.get("scenario_evidence")
        )

        role = clean_text(item.get("role"))

        if not (
            role == "scenario" or label or evidence
        ):
            continue

        source = source_map.get(
            clean_text(item.get("issue_key")),
            {},
        )

        source_text = (
            module.clean_text(
                source.get("supported_answer")
            )
            + "\n"
            + module.clean_text(
                source.get("supported_solution")
            )
        )

        rows.append(
            {
                "cluster_id": obj["cluster_id"],
                "issue_key": clean_text(
                    item.get("issue_key")
                ),
                "role": role,
                "scenario_label": label,
                "scenario_evidence": evidence,
                "grounded_in_source": (
                    module.contains_grounded_phrase(
                        evidence,
                        source_text,
                    )
                    if evidence
                    else False
                ),
            }
        )

    return rows


def revalidate():

    module = load_classifier_module()

    records = load_v3_records()

    sources_by_cluster = load_usable_sources()

    mixed_df = pd.read_excel(
        MIXED_AUDIT_FILE,
        sheet_name="all_clusters",
    )

    mixed_ids = set(
        mixed_df["cluster_id"].astype(str).tolist()
    )

    v3_ids = [
        clean_text(record["cluster_id"])
        for record in records
    ]

    if set(v3_ids) != mixed_ids:
        raise RuntimeError(
            "v3 cluster 集合与 "
            "mixed_cluster_audits_v2 不一致。"
            f" v3={sorted(set(v3_ids))}"
            f" audit={sorted(mixed_ids)}"
        )

    out_records = []

    change_rows = []

    scenario_rows = []

    for obj in records:

        cluster_id = clean_text(obj["cluster_id"])

        source_rows = sources_by_cluster.get(
            cluster_id,
            [],
        )

        scenario_rows.extend(
            build_scenario_grounding_rows(
                module,
                obj,
                source_rows,
            )
        )

        if not source_rows:

            out_records.append(
                {
                    **obj,
                    "revalidation_status":
                        "no_usable_source",
                    "revalidation_note":
                        "grounding v2 中没有 usable source",
                    "v3_generation_ready": bool(
                        obj.get("generation_ready")
                    ),
                }
            )

            change_rows.append(
                {
                    "cluster_id": cluster_id,
                    "revalidation_status":
                        "no_usable_source",
                    "field": "",
                    "before": "",
                    "after": "",
                }
            )

            continue

        if len(source_rows) != int(obj["source_count"]):

            change_rows.append(
                {
                    "cluster_id": cluster_id,
                    "revalidation_status":
                        "source_count_mismatch",
                    "field": "source_count",
                    "before": obj["source_count"],
                    "after": len(source_rows),
                }
            )

        payload = rebuild_baseline(obj)

        result = module.StructureClassification.model_validate(
            payload
        )

        try:

            result, hard_flags = module.apply_hard_rules(
                result,
                source_rows,
            )

        except RuntimeError as exc:

            out_records.append(
                {
                    **obj,
                    "revalidation_status":
                        "source_coverage_error",
                    "revalidation_note": str(exc),
                    "v3_generation_ready": bool(
                        obj.get("generation_ready")
                    ),
                }
            )

            change_rows.append(
                {
                    "cluster_id": cluster_id,
                    "revalidation_status":
                        "source_coverage_error",
                    "field": "evidence_roles",
                    "before": "",
                    "after": str(exc),
                }
            )

            continue

        before = state_of(
            obj["knowledge_structure"],
            obj["generation_ready"],
            obj["recommended_generation_mode"],
            obj["has_true_conflict"],
            obj.get("hard_rule_flags") or [],
        )

        after = state_of(
            result.knowledge_structure,
            result.generation_ready,
            result.recommended_generation_mode,
            result.has_true_conflict,
            hard_flags,
        )

        diffs = diff_states(before, after)

        status, note = classify_change(
            before["hard_rule_flags"],
            diffs,
            after,
        )

        record = {
            "cluster_id": cluster_id,
            "canonical_question": result.canonical_question,
            "source_count": len(source_rows),
            "knowledge_structure": result.knowledge_structure,
            "structure_confidence": result.confidence,
            "independent_cause_count":
                result.independent_cause_count,
            "scenario_count": result.scenario_count,
            "has_true_conflict": result.has_true_conflict,
            "conflict_fact_count": len(result.conflict_facts),
            "conflict_facts": [
                item.model_dump()
                for item in result.conflict_facts
            ],
            "temporal_dependency": result.temporal_dependency,
            "current_validity_resolved":
                result.current_validity_resolved,
            "generation_ready": result.generation_ready,
            "recommended_generation_mode":
                result.recommended_generation_mode,
            "reason": result.reason,
            "risk_flags": result.risk_flags,
            "hard_rule_flag_count": len(hard_flags),
            "hard_rule_flags": hard_flags,
            "evidence_roles": [
                item.model_dump()
                for item in result.evidence_roles
            ],
            "request_seconds": obj.get("request_seconds"),
            "revalidation_status": status,
            "revalidation_note": note,
            "v3_generation_ready": bool(
                obj["generation_ready"]
            ),
            "v3_hard_rule_flags": list(
                obj.get("hard_rule_flags") or []
            ),
        }

        out_records.append(record)

        if diffs:

            for item in diffs:

                change_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "revalidation_status": status,
                        "field": item["field"],
                        "before": item["before"],
                        "after": item["after"],
                    }
                )

        else:

            change_rows.append(
                {
                    "cluster_id": cluster_id,
                    "revalidation_status": status,
                    "field": "",
                    "before": "",
                    "after": "",
                }
            )

    return out_records, change_rows, scenario_rows


def export_excel(records, change_rows, scenario_rows):

    df = pd.DataFrame(records)

    cluster_cols = [
        "cluster_id",
        "canonical_question",
        "source_count",
        "knowledge_structure",
        "structure_confidence",
        "independent_cause_count",
        "scenario_count",
        "has_true_conflict",
        "conflict_fact_count",
        "temporal_dependency",
        "current_validity_resolved",
        "generation_ready",
        "recommended_generation_mode",
        "hard_rule_flag_count",
        "revalidation_status",
        "reason",
        "risk_flags",
        "hard_rule_flags",
    ]

    all_clusters = df[
        [col for col in cluster_cols if col in df.columns]
    ].copy()

    for col in ["risk_flags", "hard_rule_flags"]:

        if col in all_clusters.columns:

            all_clusters[col] = all_clusters[col].apply(
                lambda value: " | ".join(value)
                if isinstance(value, list)
                else value
            )

    evidence_rows = []

    conflict_rows = []

    flag_rows = []

    for record in records:

        for item in record.get("evidence_roles") or []:

            evidence_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    "issue_key": item.get("issue_key"),
                    "role": item.get("role"),
                    "scenario_label": item.get(
                        "scenario_label"
                    ),
                    "scenario_evidence": item.get(
                        "scenario_evidence"
                    ),
                    "reason": item.get("reason"),
                }
            )

        for item in record.get("conflict_facts") or []:

            conflict_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    **item,
                }
            )

        for flag in record.get("hard_rule_flags") or []:

            flag_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    "hard_rule_flag": flag,
                }
            )

    ready_df = all_clusters[
        all_clusters["generation_ready"] == True
    ]

    blocked_df = all_clusters[
        all_clusters["generation_ready"] == False
    ]

    affected_df = all_clusters[
        all_clusters["hard_rule_flag_count"] > 0
    ]

    status_counts = (
        df["revalidation_status"]
        .value_counts()
        .rename_axis("revalidation_status")
        .reset_index(name="count")
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "cluster_count",
                "value": len(df),
            },
            {
                "metric": "generation_ready_count",
                "value": int(
                    df["generation_ready"].sum()
                ),
            },
            {
                "metric": "blocked_count",
                "value": int(
                    (~df["generation_ready"]).sum()
                ),
            },
            {
                "metric": "true_conflict_count",
                "value": int(
                    df["has_true_conflict"].sum()
                ),
            },
            {
                "metric": "material_temporal_count",
                "value": int(
                    (
                        df["temporal_dependency"]
                        == "material"
                    ).sum()
                ),
            },
            {
                "metric":
                    "hard_rule_affected_cluster_count",
                "value": int(
                    (
                        df["hard_rule_flag_count"] > 0
                    ).sum()
                ),
            },
            {
                "metric": "conflict_fact_count",
                "value": int(
                    df["conflict_fact_count"].sum()
                ),
            },
            {
                "metric": "v3_generation_ready_count",
                "value": int(
                    df["v3_generation_ready"].sum()
                )
                if "v3_generation_ready" in df.columns
                else None,
            },
            {
                "metric": "unblocked_scenario_grounding",
                "value": int(
                    (
                        df["revalidation_status"]
                        == "unblocked_scenario_grounding"
                    ).sum()
                ),
            },
            {
                "metric": "changed_review_required",
                "value": int(
                    (
                        df["revalidation_status"]
                        == "changed_review_required"
                    ).sum()
                ),
            },
            {
                "metric": "unchanged",
                "value": int(
                    (
                        df["revalidation_status"]
                        == "unchanged"
                    ).sum()
                ),
            },
        ]
    )

    structure_stats = (
        df["knowledge_structure"]
        .value_counts()
        .rename_axis("knowledge_structure")
        .reset_index(name="count")
    )

    temporal_stats = (
        df["temporal_dependency"]
        .value_counts()
        .rename_axis("temporal_dependency")
        .reset_index(name="count")
    )

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        summary.to_excel(
            writer,
            index=False,
            sheet_name="summary",
        )

        structure_stats.to_excel(
            writer,
            index=False,
            sheet_name="structure_stats",
        )

        temporal_stats.to_excel(
            writer,
            index=False,
            sheet_name="temporal_stats",
        )

        all_clusters.to_excel(
            writer,
            index=False,
            sheet_name="all_clusters",
        )

        pd.DataFrame(evidence_rows).to_excel(
            writer,
            index=False,
            sheet_name="evidence_roles",
        )

        pd.DataFrame(conflict_rows).to_excel(
            writer,
            index=False,
            sheet_name="conflict_facts",
        )

        pd.DataFrame(flag_rows).to_excel(
            writer,
            index=False,
            sheet_name="hard_rule_flags",
        )

        ready_df.to_excel(
            writer,
            index=False,
            sheet_name="generation_ready",
        )

        blocked_df.to_excel(
            writer,
            index=False,
            sheet_name="blocked",
        )

        affected_df.to_excel(
            writer,
            index=False,
            sheet_name="hard_rule_affected",
        )

        status_counts.to_excel(
            writer,
            index=False,
            sheet_name="revalidation_stats",
        )

        pd.DataFrame(change_rows).to_excel(
            writer,
            index=False,
            sheet_name="revalidation_changes",
        )

        pd.DataFrame(scenario_rows).to_excel(
            writer,
            index=False,
            sheet_name="scenario_grounding",
        )


def main():

    print("=" * 70)
    print(
        "Step 11.3 v3.1 - "
        "Scenario Grounding Revalidation"
    )
    print("=" * 70)

    print("LLM 调用: 无")

    print(f"输入: {V3_JSONL}")

    assert OUTPUT_JSONL != V3_JSONL

    assert OUTPUT_XLSX.name != "mixed_structure_classifications_v3.xlsx"

    records, change_rows, scenario_rows = revalidate()

    with OUTPUT_JSONL.open("w", encoding="utf-8") as fout:

        for record in records:

            fout.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    export_excel(records, change_rows, scenario_rows)

    df = pd.DataFrame(records)

    print()
    print("-" * 70)
    print("Revalidation status")
    print("-" * 70)

    for status, count in (
        df["revalidation_status"].value_counts().items()
    ):

        print(f"{status:35} {count}")

    print()
    print("-" * 70)
    print("Generation ready")
    print("-" * 70)

    print(
        f"v3   ready: "
        f"{int(df['v3_generation_ready'].sum())} / "
        f"{len(df)}"
    )

    print(
        f"v3.1 ready: "
        f"{int(df['generation_ready'].sum())} / "
        f"{len(df)}"
    )

    changed = df[
        df["revalidation_status"] != "unchanged"
    ]

    if len(changed):

        print()
        print("-" * 70)
        print("Changed clusters")
        print("-" * 70)

        for row in changed.itertuples():

            print(
                f"{row.cluster_id} "
                f"| {row.revalidation_status} "
                f"| {row.knowledge_structure} "
                f"| ready "
                f"{row.v3_generation_ready} -> "
                f"{row.generation_ready} "
                f"| mode "
                f"{row.recommended_generation_mode}"
            )

            if row.revalidation_note:

                print(f"    note: {row.revalidation_note}")

    review = df[
        df["revalidation_status"]
        == "changed_review_required"
    ]

    print()

    if len(review):

        print(
            f"[WARNING] {len(review)} 个 cluster "
            f"出现无法用 scenario grounding 修复解释的变化，"
            f"需要人工复核: "
            f"{sorted(review['cluster_id'].tolist())}"
        )

    else:

        print(
            "[OK] 所有变化都可以由 "
            "scenario grounding role 过滤修复解释"
        )

    print()
    print(f"JSONL: {OUTPUT_JSONL}")

    print(f"Excel: {OUTPUT_XLSX}")

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
