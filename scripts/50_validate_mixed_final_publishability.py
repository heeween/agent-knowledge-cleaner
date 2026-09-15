#!/usr/bin/env python3
"""
Step 14 - Mixed candidate final publishability gate
====================================================

对 kb_entries_mixed_candidate_v5 的 10 条 candidate
做最终发布判定:

    8 条原 mixed
    + QCLUSTER-0021 (合并吸收 QCLUSTER-0066)
    + QCLUSTER-0008 (temporal unblock, 6 条时效 limitations)

三 verdict (同 20 / 28):

    publish / manual_review / reject

输入 (全部冻结, 只读):

- output/kb_entries_mixed_candidate_v5.jsonl
- output/kb_entry_grounding_validations_mixed_v5.jsonl
- output/mixed_structure_classifications_v3_1.jsonl
- output/cluster_arbitration_v1.jsonl        (0008 gate inputs)
- output/cross_entry_merge_decisions.jsonl   (0021 溯源)

判定原则:

- grounding 已由 39 v5 全部通过,
  本 gate 不重新纠结 quote 级 grounding,
  聚焦发布维度:
  question-unit 匹配 / scope / 可复用性 /
  单次事故是否被写成通用规则 /
  时效 limitations 是否充分 / 是否有未决冲突
- resolution / temporal_status 不自动 reject
- LLM verdict 之后有 deterministic 守卫:
  publish 遇以下情况降级 manual_review
    - grounding_status != grounding_pass
    - hard_flag_count > 0
    - material temporal 且 limitations 为空
    - 0008 缺文档确认 gate inputs
    - 合并条目缺 merged_from 溯源
- info 级 flag 只进 notes, 不拦截

输出 (不修改任何冻结输入):

    output/mixed_final_publishability_v1.jsonl
    output/mixed_final_publishability_v1.xlsx

用法:

    .venv/bin/python scripts/50_validate_mixed_final_publishability.py
    .venv/bin/python scripts/50_validate_mixed_final_publishability.py --dry-run
    .venv/bin/python scripts/50_validate_mixed_final_publishability.py --parse-check
    .venv/bin/python scripts/50_validate_mixed_final_publishability.py --only QCLUSTER-0008
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATE_FILE = (
    OUTPUT_DIR / "kb_entries_mixed_candidate_v5.jsonl"
)
VALIDATION_FILE = (
    OUTPUT_DIR
    / "kb_entry_grounding_validations_mixed_v5.jsonl"
)
STRUCTURE_FILE = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.jsonl"
)
ARBITRATION_FILE = (
    OUTPUT_DIR / "cluster_arbitration_v1.jsonl"
)
MERGE_FILE = (
    OUTPUT_DIR / "cross_entry_merge_decisions.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR / "mixed_final_publishability_v1.jsonl"
)
OUTPUT_XLSX = (
    OUTPUT_DIR / "mixed_final_publishability_v1.xlsx"
)
ERROR_LOG = (
    OUTPUT_DIR
    / "mixed_final_publishability_llm_errors.log"
)

MODEL = "glm-5.3-flash"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 180

SUMMARY_PREVIEW_CHARS = 600
NOTE_PREVIEW_CHARS = 200


# ============================================================
# 输入加载与组装
# ============================================================

def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def load_entry_contexts():

    candidates = {
        record["cluster_id"]: record
        for record in load_jsonl(CANDIDATE_FILE)
    }
    validations = {
        record["cluster_id"]: record
        for record in load_jsonl(VALIDATION_FILE)
    }
    structures = {
        record["cluster_id"]: record
        for record in load_jsonl(STRUCTURE_FILE)
    }
    arbitrations = {
        record["cluster_id"]: record
        for record in load_jsonl(ARBITRATION_FILE)
    }

    merge_survivor = None
    merge_absorbed = None

    for record in load_jsonl(MERGE_FILE):
        if record.get("decision") == "merge":
            merge_survivor = record.get("survivor")
            merge_absorbed = record.get("absorbed")

    contexts = {}

    for cluster_id, entry in candidates.items():
        validation = validations.get(cluster_id, {})
        structure = structures.get(cluster_id, {})
        arbitration = arbitrations.get(cluster_id)

        gate_inputs = (
            arbitration.get("publishability_gate_inputs")
            if arbitration
            else []
        )

        validation_flags = [
            {
                "flag": flag.get("flag"),
                "severity": flag.get("severity"),
                "detail": str(
                    flag.get("detail")
                )[:NOTE_PREVIEW_CHARS],
            }
            for flag in (
                validation.get("validation_flags")
                or []
            )
        ]

        contexts[cluster_id] = {
            "cluster_id": cluster_id,
            "canonical_question": entry.get(
                "canonical_question"
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
            "units": entry.get("units") or [],
            "limitations": entry.get("limitations")
            or [],
            "notes": (entry.get("notes") or [])[
                :NOTE_PREVIEW_CHARS
            ],
            "usable_source_count": entry.get(
                "usable_source_count"
            ),
            "cited_source_count": entry.get(
                "cited_source_count"
            ),
            "risky_source_count": entry.get(
                "risky_source_count"
            ),
            "merged_from_clusters": entry.get(
                "merged_from_clusters"
            ),
            "temporal_validity": entry.get(
                "temporal_validity"
            ),
            "temporal_dependency": structure.get(
                "temporal_dependency"
            ),
            "grounding_status": validation.get(
                "grounding_status"
            ),
            "hard_flag_count": validation.get(
                "hard_flag_count"
            ),
            "medium_flag_count": validation.get(
                "medium_flag_count"
            ),
            "info_flag_count": validation.get(
                "info_flag_count"
            ),
            "validation_flags": validation_flags,
            "candidate_soft_flags": validation.get(
                "candidate_soft_flags"
            ),
            "doc_gate_input_count": len(gate_inputs),
            "doc_gate_quotes": [
                gate_input["quote"]
                for gate_input in gate_inputs
            ],
            "merge_survivor": merge_survivor,
            "merge_absorbed": merge_absorbed,
        }

    return contexts


# ============================================================
# Prompt 构建
# ============================================================

def build_prompt(context: dict) -> str:

    units_block = json.dumps(
        context["units"],
        ensure_ascii=False,
        indent=1,
    )

    limitations_block = (
        "\n".join(
            f"- {limitation}"
            for limitation in context["limitations"]
        )
        or "（无）"
    )

    flags_block = (
        "\n".join(
            f"- [{flag['severity']}] {flag['flag']}: "
            f"{flag['detail']}"
            for flag in context["validation_flags"]
        )
        or "（无）"
    )

    soft_block = (
        "\n".join(
            f"- {soft}"
            for soft in (
                context["candidate_soft_flags"] or []
            )
        )
        or "（无）"
    )

    merge_block = "（非合并条目）"

    if context["merged_from_clusters"]:
        merge_block = (
            f"由跨条目 merge 产生，absorbed: "
            f"{context['merged_from_clusters']}"
        )

    doc_block = "（无）"

    if context["doc_gate_quotes"]:
        doc_block = "\n".join(
            f"- 「{quote}」"
            for quote in context["doc_gate_quotes"]
        )

    return f"""你是一个严格的知识库发布审核员。

任务：对一条已通过全部 grounding 校验的 Candidate KB 条目做最终发布判定。

## 判定只关注发布维度（不要重新纠结 quote 级 grounding，那已由 gate 全部通过）

publish 要求整体满足：

1. canonical_question 范围清晰，不宽于内容支持的范围
2. 所有 unit 都真正回答这个问题（无 scope mismatch）
3. 内容具备可复用性（不是只对某一客户/某一次会话成立）
4. 没有把单次事故写成通用规则
5. 时效风险已被 limitations 充分提示
6. 没有未决的互斥冲突
7. 没有危险的无依据时间/菜单/流程

注意：

- resolution/temporal_status 不自动 reject；
  例如"系统目前不支持某功能"若来源明确
  属于稳定产品能力，仍可 publish
- 不同原因 ≠ 冲突；不同 workaround ≠ 冲突；
  temporary ≠ conflict
- 已知 info 级 flag 不拦截，但要确认其在
  发布语境下确实无害
- 不要为了提高发布率强行 publish

manual_review：知识有价值但存在必须人工确认的风险
（强时效 / 版本适用性不确定 / 特定场景限定等）。

reject：问题与内容不匹配 / 只是一次性事故 /
无法形成可复用知识 / 核心结论冲突 / 范围明显过宽。

## 待审条目（全部字段来自冻结 pipeline 输出）

- cluster_id: {context['cluster_id']}
- canonical_question: {context['canonical_question']}
- 知识结构: {context['knowledge_structure']}（generation_mode = {context['generation_mode']}）
- v3.1 temporal_dependency: {context['temporal_dependency']}
- grounding 验证: {context['grounding_status']}（hard={context['hard_flag_count']} medium={context['medium_flag_count']} info={context['info_flag_count']}）
- 来源统计: usable={context['usable_source_count']} cited={context['cited_source_count']} risky={context['risky_source_count']}
- 合并溯源: {merge_block}
- 文档机制确认（仅 0008 有，来自正式文档逐字引用）:
{doc_block}

summary_answer:
{context['summary_answer'][:SUMMARY_PREVIEW_CHARS]}

units（{len(context['units'])} 个）:
{units_block}

limitations（{len(context['limitations'])} 条）:
{limitations_block}

validation flags:
{flags_block}

candidate soft flags:
{soft_block}

## 输出 JSON 格式（不要输出 JSON 以外的任何文字）

{{
  "cluster_id": "{context['cluster_id']}",
  "verdict": "publish | manual_review | reject",
  "reusable_knowledge": "yes | partial | no",
  "incident_as_rule_risk": "none | low | high",
  "temporal_risk": "none | low | high",
  "reason": "判定核心理由",
  "notes": ["审核备注，可为空数组"]
}}"""


# ============================================================
# LLM schema 与调用（模式同 43/48）
# ============================================================

class PublishabilityVerdict(BaseModel):

    cluster_id: str
    verdict: Literal[
        "publish", "manual_review", "reject"
    ]
    reusable_knowledge: Literal[
        "yes", "partial", "no"
    ]
    incident_as_rule_risk: Literal[
        "none", "low", "high"
    ]
    temporal_risk: Literal["none", "low", "high"]
    reason: str
    notes: List[str] = Field(default_factory=list)


ENUM_VALUES = {
    "verdict": [
        "publish", "manual_review", "reject"
    ],
    "reusable_knowledge": ["yes", "partial", "no"],
    "incident_as_rule_risk": ["none", "low", "high"],
    "temporal_risk": ["none", "low", "high"],
}


def build_client():

    return OpenAI(
        base_url=(
            "https://open.bigmodel.cn/api/paas/v4"
        ),
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


def strip_json_fences(raw):

    text = raw.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def log_llm_error(cluster_id, raw, error):

    with ERROR_LOG.open("a", encoding="utf-8") as f:

        f.write(
            "=" * 70
            + "\n"
            + f"[{datetime.now().isoformat()}] "
            + f"{cluster_id}\n"
            + f"ERROR: {error}\n"
            + "RAW RESPONSE:\n"
            + raw
            + "\n"
        )


def build_repair_note(error):

    enum_hint = "; ".join(
        f"{field}: {'/'.join(values)}"
        for field, values in ENUM_VALUES.items()
    )

    return (
        "你上一次的输出未通过 schema 校验，错误如下:\n"
        f"{error}\n\n"
        "请重新输出完整 JSON，要求:\n"
        "1. 字段名与类型与要求完全一致;\n"
        f"2. 枚举取值必须严格使用以下之一 ({enum_hint});\n"
        "3. notes 是字符串数组;\n"
        "4. 不要输出 JSON 以外的任何文字。"
    )


def judge_entry(client, context: dict):

    prompt = build_prompt(context)

    cluster_id = context["cluster_id"]

    last_error = None
    repair_note = None
    last_raw = None

    for attempt in range(1, MAX_RETRIES + 1):

        started = time.time()

        messages = [
            {"role": "user", "content": prompt}
        ]

        if repair_note and last_raw:

            messages.append(
                {
                    "role": "assistant",
                    "content": last_raw,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": repair_note,
                }
            )

        response = (
            client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={
                    "type": "json_object"
                },
                temperature=0,
                extra_body={
                    "reasoning_effort": "low"
                },
            )
        )

        elapsed = time.time() - started

        raw = strip_json_fences(
            response.choices[0].message.content
        )

        last_raw = raw

        try:

            data = json.loads(raw)
            data["cluster_id"] = cluster_id

            verdict = (
                PublishabilityVerdict.model_validate(
                    data
                )
            )

            if attempt > 1:
                log_llm_error(
                    cluster_id,
                    raw,
                    f"RECOVERED_ON_ATTEMPT_{attempt}",
                )

            return {
                "verdict": verdict,
                "model": MODEL,
                "attempt": attempt,
                "request_seconds": round(
                    elapsed, 2
                ),
            }

        except Exception as error:

            last_error = error
            log_llm_error(cluster_id, raw, error)

            repair_note = build_repair_note(error)

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"{cluster_id} LLM schema 校验连续 "
        f"{MAX_RETRIES} 次失败: {last_error}"
    )


# ============================================================
# deterministic 守卫（publish 降级）
# ============================================================

def apply_guards(context: dict, verdict):

    downgrades = []

    if verdict.verdict != "publish":
        return verdict.verdict, downgrades

    if context["grounding_status"] != (
        "grounding_pass"
    ):
        downgrades.append(
            "GUARD_NOT_GROUNDING_PASS"
        )

    if (context["hard_flag_count"] or 0) > 0:
        downgrades.append("GUARD_HARD_FLAGS_PRESENT")

    if (
        context["temporal_dependency"] == "material"
        and not context["limitations"]
    ):
        downgrades.append(
            "GUARD_MATERIAL_TEMPORAL_NO_LIMITATIONS"
        )

    if context["temporal_validity"] and (
        context["doc_gate_input_count"] == 0
    ):
        downgrades.append(
            "GUARD_NO_DOC_CONFIRMATION"
        )

    if context["merge_survivor"] == (
        context["cluster_id"]
    ) and not context["merged_from_clusters"]:
        downgrades.append("GUARD_NO_MERGE_TRACE")

    if downgrades:
        return "manual_review", downgrades

    return "publish", downgrades


# ============================================================
# 主流程
# ============================================================

def main():

    dry_run = "--dry-run" in sys.argv
    parse_check = "--parse-check" in sys.argv

    only = None

    if "--only" in sys.argv:
        only = sys.argv[
            sys.argv.index("--only") + 1
        ]

    contexts = load_entry_contexts()

    cluster_ids = sorted(contexts.keys())

    if only:
        if only not in contexts:
            raise ValueError(
                f"--only 仅支持: {cluster_ids}"
            )

        cluster_ids = [only]

    print(f"待审条目: {len(cluster_ids)}")

    if parse_check:
        fake = PublishabilityVerdict(
            cluster_id="QCLUSTER-0008",
            verdict="publish",
            reusable_knowledge="yes",
            incident_as_rule_risk="none",
            temporal_risk="low",
            reason="parse-check",
            notes=[],
        )

        final_verdict, downgrades = apply_guards(
            contexts["QCLUSTER-0008"], fake
        )

        assert final_verdict == "publish"
        assert downgrades == []

        broken = contexts["QCLUSTER-0008"].copy()
        broken["doc_gate_input_count"] = 0

        final_verdict, downgrades = apply_guards(
            broken, fake
        )

        assert final_verdict == "manual_review"
        assert downgrades == [
            "GUARD_NO_DOC_CONFIRMATION"
        ]

        for cluster_id in cluster_ids:
            prompt = build_prompt(contexts[cluster_id])
            print(
                f"  prompt {cluster_id}: "
                f"{len(prompt)} chars"
            )

        print("parse-check 通过")
        return

    if dry_run:
        for cluster_id in cluster_ids:
            context = contexts[cluster_id]
            prompt = build_prompt(context)

            print(
                f"{cluster_id}  "
                f"units={len(context['units'])}  "
                f"limitations="
                f"{len(context['limitations'])}  "
                f"grounding="
                f"{context['grounding_status']}  "
                f"temporal_dep="
                f"{context['temporal_dependency']}  "
                f"doc_inputs="
                f"{context['doc_gate_input_count']}  "
                f"prompt={len(prompt)} chars"
            )

        print(
            "dry-run 完成（未调用 LLM，未写输出）"
        )
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY 环境变量"
        )

    client = build_client()

    output_records = []

    for index, cluster_id in enumerate(
        cluster_ids, start=1
    ):
        context = contexts[cluster_id]

        print(
            f"[{index}/{len(cluster_ids)}] "
            f"{cluster_id} 判定中..."
        )

        judged = judge_entry(client, context)
        final_verdict, downgrades = apply_guards(
            context, judged["verdict"]
        )

        record = {
            "cluster_id": cluster_id,
            "canonical_question": context[
                "canonical_question"
            ],
            "knowledge_structure": context[
                "knowledge_structure"
            ],
            "temporal_dependency": context[
                "temporal_dependency"
            ],
            "grounding_status": context[
                "grounding_status"
            ],
            "verdict_raw": judged[
                "verdict"
            ].verdict,
            "verdict_final": final_verdict,
            "guard_downgrades": downgrades,
            "reusable_knowledge": judged[
                "verdict"
            ].reusable_knowledge,
            "incident_as_rule_risk": judged[
                "verdict"
            ].incident_as_rule_risk,
            "temporal_risk": judged[
                "verdict"
            ].temporal_risk,
            "reason": judged["verdict"].reason,
            "notes": judged["verdict"].notes,
            "model": judged["model"],
            "attempt": judged["attempt"],
            "request_seconds": judged[
                "request_seconds"
            ],
        }

        output_records.append(record)

        downgrade_text = (
            f" ↓{downgrades}"
            if downgrades
            else ""
        )

        print(
            f"  {judged['verdict'].verdict}"
            f" → {final_verdict}{downgrade_text}"
        )

    with open(
        OUTPUT_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in output_records:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    verdict_rows = [
        {
            "cluster_id": record["cluster_id"],
            "canonical_question": record[
                "canonical_question"
            ],
            "verdict_raw": record["verdict_raw"],
            "verdict_final": record[
                "verdict_final"
            ],
            "guard_downgrades": ";".join(
                record["guard_downgrades"]
            ),
            "reusable_knowledge": record[
                "reusable_knowledge"
            ],
            "incident_as_rule_risk": record[
                "incident_as_rule_risk"
            ],
            "temporal_risk": record[
                "temporal_risk"
            ],
            "reason": record["reason"],
        }
        for record in output_records
    ]

    note_rows = []

    for record in output_records:
        for note in record["notes"]:
            note_rows.append({
                "cluster_id": record[
                    "cluster_id"
                ],
                "verdict_final": record[
                    "verdict_final"
                ],
                "note": note,
            })

    with pd.ExcelWriter(
        OUTPUT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame(verdict_rows).to_excel(
            writer, sheet_name="verdicts", index=False
        )
        pd.DataFrame(note_rows).to_excel(
            writer, sheet_name="llm_notes", index=False
        )
        pd.DataFrame(
            [
                {
                    "cluster_id": record[
                        "cluster_id"
                    ],
                    **record,
                }
                for record in output_records
            ]
        ).drop(
            columns=["notes", "cluster_id"],
        ).to_excel(
            writer,
            sheet_name="full_records",
            index=False,
        )

    print()
    print("=" * 60)
    print("Mixed final publishability gate 完成")
    print("=" * 60)

    counts = {"publish": 0, "manual_review": 0, "reject": 0}

    for record in output_records:
        counts[record["verdict_final"]] += 1
        print(
            f"  {record['cluster_id']}  "
            f"→ {record['verdict_final']}"
        )

    print()
    print(
        f"publish={counts['publish']}  "
        f"manual_review={counts['manual_review']}  "
        f"reject={counts['reject']}"
    )
    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
