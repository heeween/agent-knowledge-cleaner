#!/usr/bin/env python3
"""
Step 13.4 - blocked cluster 仲裁结案
====================================

基于冻结输出对 5 个目标 cluster 做正式仲裁记录:

    QCLUSTER-0006  短信开通
    QCLUSTER-0074  回访录音
    QCLUSTER-0139  企微车主画像
    QCLUSTER-0194  验证过期
    QCLUSTER-0008  无法登录

输入 (全部冻结, 只读):

- output/formal_doc_evidence_links_v1.jsonl   (Step 13.3)
- output/mixed_structure_classifications_v3_1 (Step 11.3)
- output/temporal_validity_decisions.jsonl    (Step 12.2)
- output/formal_doc_evidence_chunks_v1 / paragraphs_v1 (Step 13.2)

决策表说明:

decision 表是人工复核后的仲裁结论
(依据 13.3 覆盖结论 + 12.2 temporal 决策),
脚本不产生新判断, 只做三类守卫:

1. 一致性守卫: 决策必须与冻结硬事实兼容
   (temporal unblock 标志 / 结构判定 / direct 关联数)
2. 引用守卫: reason 中引用的 coverage_note
   必须与 13.3 输出逐字一致
3. 取证守卫: gate input 的 quote 必须逐字来自
   冻结 chunk / 13.3 links

原则:

- 本步骤不生成 KB 内容, 不修改任何冻结输出
- 0008 不改写 v5 条目,
  官方确认仅归档为 publishability gate 输入
- 0139 不允许把"授权前置"推断为
  "画像空白的原因" (来源未陈述该因果)

输出:

    output/cluster_arbitration_v1.jsonl
    output/cluster_arbitration_v1.xlsx

用法:

    .venv/bin/python scripts/49_arbitrate_formal_doc_clusters.py
"""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

LINKS_FILE = (
    OUTPUT_DIR / "formal_doc_evidence_links_v1.jsonl"
)
STRUCTURE_FILE = (
    OUTPUT_DIR / "mixed_structure_classifications_v3_1.jsonl"
)
TEMPORAL_FILE = (
    OUTPUT_DIR / "temporal_validity_decisions.jsonl"
)
AUDITS_FILE = OUTPUT_DIR / "mixed_cluster_audits_v2.jsonl"
CHUNKS_FILE = (
    OUTPUT_DIR / "formal_doc_evidence_chunks_v1.jsonl"
)
PARAGRAPHS_FILE = (
    OUTPUT_DIR / "formal_doc_paragraphs_v1.jsonl"
)

OUTPUT_JSONL = OUTPUT_DIR / "cluster_arbitration_v1.jsonl"
OUTPUT_XLSX = OUTPUT_DIR / "cluster_arbitration_v1.xlsx"

# 0008 gate input 的机制取证:
# 从其关联 chunk 的段落中按关键词确定性选取
MECHANISM_KEYWORDS = ["密码"]


# ============================================================
# 仲裁决策表 (人工复核, 脚本只做守卫)
# ============================================================

ARBITRATION_TABLE = {
    "QCLUSTER-0006": {
        "decision": "KEEP_BLOCKED_INSUFFICIENT_EVIDENCE",
        "decision_rule": (
            "NO_TEMPORAL_DECISION_PLUS_"
            "INSUFFICIENT_EVIDENCE_NO_DIRECT_DOC"
        ),
    },
    "QCLUSTER-0074": {
        "decision": "KEEP_BLOCKED_INCIDENT_TEMPORARY",
        "decision_rule": (
            "TEMPORAL_UNBLOCK_FALSE_AND_"
            "NO_DIRECT_DOC_ARBITRATION"
        ),
    },
    "QCLUSTER-0139": {
        "decision": "KEEP_BLOCKED_INCIDENT_TEMPORARY",
        "decision_rule": (
            "TEMPORAL_UNBLOCK_FALSE_AND_"
            "NO_DIRECT_DOC_ARBITRATION"
        ),
        "regeneration_rejected": {
            "reason_code": (
                "DOC_EVIDENCE_DOES_NOT_STATE_BLANK_CAUSE"
            ),
            "explanation": (
                "官方文档 direct 证据仅覆盖画像功能定义与"
                "企微授权前置条件；来源未陈述"
                "『未授权导致画像空白』这一因果。"
                "将其补生成为新 cause 属于发明条件，"
                "按 grounding 原则拒绝。"
            ),
        },
    },
    "QCLUSTER-0194": {
        "decision": "KEEP_BLOCKED_STALE",
        "decision_rule": "TEMPORAL_STALE_NO_RECENT",
    },
    "QCLUSTER-0008": {
        "decision": "DOC_MECHANISM_CONFIRMED_GATE_INPUT",
        "decision_rule": (
            "CURRENT_CONFIRMED_PLUS_MECHANISM_DOC_EVIDENCE"
        ),
    },
}


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():

    links = {
        record["cluster_id"]: record
        for record in load_jsonl(LINKS_FILE)
    }
    structures = {
        record["cluster_id"]: record
        for record in load_jsonl(STRUCTURE_FILE)
    }
    temporal = {
        record["cluster_id"]: record
        for record in load_jsonl(TEMPORAL_FILE)
    }
    audits = {
        record["cluster_id"]: record
        for record in load_jsonl(AUDITS_FILE)
    }
    chunks = {
        record["chunk_id"]: record
        for record in load_jsonl(CHUNKS_FILE)
    }
    paragraphs = load_jsonl(PARAGRAPHS_FILE)

    errors = []
    records = []

    for cluster_id, table_row in (
        ARBITRATION_TABLE.items()
    ):
        link_record = links.get(cluster_id)

        if link_record is None:
            errors.append(
                f"{cluster_id}: 13.3 关联记录缺失"
            )
            continue

        structure = structures.get(cluster_id, {})
        temporal_record = temporal.get(cluster_id)
        audit = audits.get(cluster_id, {})

        kept_links = link_record["links"]
        direct_links = [
            link for link in kept_links
            if link["relevance"] == "direct"
        ]

        final_decision = (
            temporal_record.get("final_decision")
            if temporal_record
            else None
        )
        unblock = (
            temporal_record.get("unblock_for_candidate")
            if temporal_record
            else None
        )

        # ---------- 一致性守卫 ----------

        if cluster_id == "QCLUSTER-0006":
            ok = (
                temporal_record is None
                and structure.get("knowledge_structure")
                == "insufficient_evidence"
                and structure.get("generation_ready")
                is False
                and len(direct_links) == 0
            )

            if not ok:
                errors.append(
                    f"{cluster_id}: 决策与硬事实不一致"
                )

        elif cluster_id == "QCLUSTER-0074":
            ok = (
                final_decision == "INCIDENT_TEMPORARY"
                and unblock is False
                and len(direct_links) == 0
            )

            if not ok:
                errors.append(
                    f"{cluster_id}: 决策与硬事实不一致"
                )

        elif cluster_id == "QCLUSTER-0139":
            ok = (
                final_decision == "INCIDENT_TEMPORARY"
                and unblock is False
                and len(kept_links) >= 1
            )

            if not ok:
                errors.append(
                    f"{cluster_id}: 决策与硬事实不一致"
                )

        elif cluster_id == "QCLUSTER-0194":
            ok = (
                final_decision
                == "STALE_NO_RECENT_CONFIRMATION"
                and unblock is False
            )

            if not ok:
                errors.append(
                    f"{cluster_id}: 决策与硬事实不一致"
                )

        elif cluster_id == "QCLUSTER-0008":
            ok = (
                final_decision == "CURRENT_CONFIRMED"
                and unblock is True
                and len(kept_links) >= 1
            )

            if not ok:
                errors.append(
                    f"{cluster_id}: 决策与硬事实不一致"
                )

        # ---------- 引用守卫:
        # coverage_note 原文进入 reason ----------

        coverage_note = link_record["coverage_note"]

        # ---------- 0008 gate input 取证 ----------

        gate_inputs = []

        if cluster_id == "QCLUSTER-0008":
            chunk_id = kept_links[0]["chunk_id"]
            chunk = chunks[chunk_id]
            chunk_paragraphs = [
                p for p in paragraphs
                if p["document_id"] == chunk["document_id"]
                and chunk["para_start"]
                <= p["para_index"]
                <= chunk["para_end"]
            ]

            for p in chunk_paragraphs:
                if any(
                    keyword in p["text"]
                    for keyword in MECHANISM_KEYWORDS
                ):
                    gate_inputs.append({
                        "input_type": (
                            "DOC_MECHANISM_CONFIRMATION"
                        ),
                        "chunk_id": chunk_id,
                        "document_id": p["document_id"],
                        "line_number": p["line_number"],
                        "quote": p["text"],
                        "target_entry": (
                            "QCLUSTER-0008 "
                            "(kb_entries_mixed_candidate_v5)"
                        ),
                        "usage": (
                            "final publishability gate "
                            "验证输入; 不修改 v5 条目"
                        ),
                    })

        # ---------- 0139 拒绝补生成取证 ----------

        regeneration_rejected = (
            table_row.get("regeneration_rejected")
        )

        authorization_quote = None

        if cluster_id == "QCLUSTER-0139":
            for link in kept_links:
                for quote in link["quotes"]:
                    if "授权" in quote:
                        authorization_quote = {
                            "chunk_id": link["chunk_id"],
                            "quote": quote,
                        }
                        break

                if authorization_quote:
                    break

        record = {
            "cluster_id": cluster_id,
            "core_question": audit.get(
                "core_question"
            ),
            "knowledge_structure": structure.get(
                "knowledge_structure"
            ),
            "generation_ready": structure.get(
                "generation_ready"
            ),
            "temporal_final_decision": final_decision,
            "temporal_unblock": unblock,
            "kept_link_count": link_record[
                "kept_link_count"
            ],
            "direct_link_count": len(direct_links),
            "linked_chunks": [
                link["chunk_id"]
                for link in kept_links
            ],
            "coverage_note_verbatim": coverage_note,
            "arbitration_decision": table_row[
                "decision"
            ],
            "decision_rule": table_row[
                "decision_rule"
            ],
            "regeneration_rejected": (
                regeneration_rejected
            ),
            "authorization_quote": authorization_quote,
            "publishability_gate_inputs": gate_inputs,
        }

        records.append(record)

    if errors:
        for error in errors:
            print(f"守卫失败: {error}")

        raise SystemExit(1)

    with open(
        OUTPUT_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in records:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    summary_rows = [
        {
            "cluster_id": record["cluster_id"],
            "arbitration_decision": record[
                "arbitration_decision"
            ],
            "kept_link_count": record[
                "kept_link_count"
            ],
            "direct_link_count": record[
                "direct_link_count"
            ],
            "temporal_final_decision": record[
                "temporal_final_decision"
            ],
        }
        for record in records
    ]

    gate_rows = []

    for record in records:
        for gate_input in record[
            "publishability_gate_inputs"
        ]:
            gate_rows.append({
                "cluster_id": record["cluster_id"],
                **gate_input,
            })

    regeneration_rows = []

    for record in records:
        if record["regeneration_rejected"]:
            regeneration_rows.append({
                "cluster_id": record["cluster_id"],
                "reason_code": record[
                    "regeneration_rejected"
                ]["reason_code"],
                "explanation": record[
                    "regeneration_rejected"
                ]["explanation"],
                "authorization_quote": (
                    record["authorization_quote"] or {}
                ).get("quote"),
                "authorization_chunk": (
                    record["authorization_quote"] or {}
                ).get("chunk_id"),
                "coverage_note_verbatim": record[
                    "coverage_note_verbatim"
                ],
            })

    with pd.ExcelWriter(
        OUTPUT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame(records).drop(
            columns=[
                "publishability_gate_inputs",
                "authorization_quote",
            ]
        ).to_excel(
            writer, sheet_name="arbitration", index=False
        )
        pd.DataFrame(gate_rows).to_excel(
            writer, sheet_name="gate_inputs", index=False
        )
        pd.DataFrame(regeneration_rows).to_excel(
            writer,
            sheet_name="regeneration_review",
            index=False,
        )

    print("=" * 60)
    print("仲裁结案完成（Step 13.4）")
    print("=" * 60)

    for row in summary_rows:
        print(
            f"  {row['cluster_id']}  "
            f"→ {row['arbitration_decision']}"
        )

    print()
    print(f"0008 gate inputs: {len(gate_rows)} 条")
    print(f"补生成复核记录: {len(regeneration_rows)} 条")
    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
