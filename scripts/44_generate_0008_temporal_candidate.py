#!/usr/bin/env python3
"""
Step 12.3 - Temporal-Unblocked Candidate 生成
=============================================

为 Step 12.2 判定 CURRENT_CONFIRMED 的
QCLUSTER-0008 (登录失败 troubleshooting)
生成 1 条 Candidate KB entry。

与 38 的关系:

    38 (frozen) 处理 generation_ready=ture 的
    10 个 cluster; 0008 当时 blocked
    (MATERIAL_TEMPORAL_UNRESOLVED),
    recommended_generation_mode=do_not_generate,
    independent_cause_count=0
    (troubleshooting 结构不按 cause 计数)。

    Step 12.2 解除 temporal block 后,
    本脚本按 cause_items 生成 4 个 unit
    (每个 usable source 一个)。

结构:

1. deterministic 证据组装
   (4 个 usable source 按时间排序 +
    Step 12.2 的 temporal 标签与注记)
2. LLM (glm-5.3-flash) 生成 summary + units
   - notes / limitations 不由 LLM 生成:
     Step 12.2 的时效注记逐字注入 +
     historical 成员自动生成历史提示
3. deterministic 门:
   - unit 数 = source 数, 覆盖完整
   - 每个 unit 引用合法 source
   - 时效注记逐字在 limitations 中
   - unit 内容词面 grounding (与 38 同规则)
4. 输出 candidate v4
   (v3 的 9 条逐字保留 + 0008 新增)

输出 (不修改任何冻结输入):

    output/kb_entries_mixed_candidate_v4.jsonl
    output/kb_entries_mixed_candidate_v4.xlsx

用法:

    .venv/bin/python scripts/44_generate_0008_temporal_candidate.py --dry-run
    .venv/bin/python scripts/44_generate_0008_temporal_candidate.py
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATE_V3 = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v3.jsonl"
)

GROUNDING_FILE = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

STRUCTURE_FILE = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.xlsx"
)

TEMPORAL_FILE = (
    OUTPUT_DIR
    / "temporal_validity_decisions.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v4.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v4.xlsx"
)

ERROR_LOG = OUTPUT_DIR / "temporal_candidate_gen_errors.log"

TARGET_CLUSTER = "QCLUSTER-0008"

MODEL = "glm-5.3-flash"

MAX_RETRIES = 4

REQUEST_TIMEOUT = 180

DRY_RUN = "--dry-run" in sys.argv


# ============================================================
# 工具 (词面 grounding 与 38/39/43 同规则)
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float):
        if pd.isna(value):
            return ""

    return str(value).strip()


def normalize_text(text):

    text = clean_text(text)

    text = re.sub(r"\s+", "", text)

    text = re.sub(
        r"[，。！？；：、,.!?;:\-_()（）\[\]【】\"'“”‘’]",
        "",
        text,
    )

    return text.lower()


def shingles(text, size=2):

    text = normalize_text(text)

    if len(text) < size:
        return set([text]) if text else set()

    return set(
        text[index:index + size]
        for index in range(len(text) - size + 1)
    )


def lexical_grounded_ratio(content, source_text):

    content_shingles = shingles(content)

    if not content_shingles:
        return 0.0

    source_shingles = shingles(source_text)

    if not source_shingles:
        return 0.0

    hit = len(content_shingles & source_shingles)

    return hit / len(content_shingles)


def parse_ts(value):

    text = clean_text(value)

    if not text:
        return None

    try:
        return datetime.strptime(
            text, "%Y-%m-%d %H:%M:%S"
        )

    except ValueError:
        return None


def load_jsonl_file(path):

    records = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


# ============================================================
# LLM Schema
# ============================================================

class GeneratedUnit(BaseModel):

    title: str

    condition: str = ""

    content: str

    steps: List[str] = []

    source_issue_key: str


class GeneratedEntry(BaseModel):

    summary_answer: str

    units: List[GeneratedUnit]

    crm_feature: str

    problem_type: str

    knowledge_confidence: float

    @field_validator(
        "knowledge_confidence",
        mode="before",
    )
    @classmethod
    def coerce_confidence(cls, value):

        if isinstance(value, str):

            text = value.strip().rstrip("%")

            try:
                return float(text)

            except ValueError:
                return 0.0

        return value


# ============================================================
# 证据组装
# ============================================================

def load_sources_for_cluster():

    df = pd.read_excel(
        GROUNDING_FILE,
        sheet_name="all_sources",
    )

    rows = df[
        (df["cluster_id"] == TARGET_CLUSTER)
        & (df["usable_for_kb"] == True)
    ]

    sources = []

    for row in rows.to_dict(orient="records"):

        sources.append(
            {
                "issue_key": clean_text(
                    row.get("issue_key")
                ),
                "question": clean_text(
                    row.get("question")
                ),
                "supported_answer": clean_text(
                    row.get("supported_answer")
                ),
                "supported_solution": clean_text(
                    row.get("supported_solution")
                ),
                "temporal_status": clean_text(
                    row.get("temporal_status")
                ),
                "resolution": clean_text(
                    row.get("resolution")
                ),
            }
        )

    return sources


def load_temporal_decision():

    for record in load_jsonl_file(
        TEMPORAL_FILE
    ):

        if record.get("cluster_id") == (
            TARGET_CLUSTER
        ):

            return record

    raise RuntimeError(
        f"缺少 Step 12.2 判定: {TARGET_CLUSTER}"
    )


def assemble_evidence():

    sources = load_sources_for_cluster()

    temporal = load_temporal_decision()

    if not sources:

        raise RuntimeError("无 usable source")

    labels = {
        clean_text(a.get("issue_key")): clean_text(
            a.get("temporal_label")
        )
        for a in (
            temporal.get("decision") or {}
        ).get("member_assessments") or []
    }

    # 时间信息来自 Step 12.1
    ts_by_key = {
        clean_text(row.get("issue_key")): row
        for row in load_jsonl_file(
            OUTPUT_DIR / "issue_timestamps.jsonl"
        )
    }

    for source in sources:

        ts = ts_by_key.get(
            source["issue_key"], {}
        )

        source["issue_date"] = clean_text(
            ts.get("issue_time_min")
        )[:10]

        source["temporal_label"] = labels.get(
            source["issue_key"], "uncertain"
        )

    sources.sort(
        key=lambda s: s["issue_date"]
    )

    structure_df = pd.read_excel(
        STRUCTURE_FILE,
        sheet_name="all_clusters",
    )

    gate_row = structure_df[
        structure_df["cluster_id"]
        == TARGET_CLUSTER
    ].iloc[0].to_dict()

    evidence_roles = pd.read_excel(
        STRUCTURE_FILE,
        sheet_name="evidence_roles",
    )

    roles = evidence_roles[
        evidence_roles["cluster_id"]
        == TARGET_CLUSTER
    ]

    role_by_key = {
        clean_text(row.get("issue_key")): {
            "role": clean_text(row.get("role")),
            "reason": clean_text(
                row.get("reason")
            ),
        }
        for row in roles.to_dict(
            orient="records"
        )
    }

    annotations = list(
        (temporal.get("decision") or {}).get(
            "required_annotations"
        ) or []
    )

    return {
        "cluster_id": TARGET_CLUSTER,
        "canonical_question": clean_text(
            gate_row.get("canonical_question")
        ),
        "structure": clean_text(
            gate_row.get("knowledge_structure")
        ),
        "sources": sources,
        "role_by_key": role_by_key,
        "annotations": annotations,
        "temporal_decision": temporal,
    }


# ============================================================
# notes / limitations deterministic 注入
# ============================================================

def build_notes_and_limitations(
    evidence,
):
    """
    全部 deterministic, LLM 不参与:

    - limitations = Step 12.2 时效注记 (逐字)
      + historical 成员自动历史提示
      + temporary 成员自动时间盒提示
    - notes = 多路径并列说明 (由 titles 生成,
      在 entry 组装阶段完成)
    """

    annotations = list(
        evidence["annotations"]
    )

    label_text = {
        "historical": "历史证据 (historical)",
        "temporary_incident": (
            "临时故障证据 (temporary_incident)"
        ),
        "current": "当前证据",
        "uncertain": "时效未确认证据",
    }

    for source in evidence["sources"]:

        label = source["temporal_label"]

        if label == "current":
            continue

        annotations.append(
            f"{source['issue_key']} "
            f"({source['issue_date']}): "
            f"{source['question']} — "
            f"该来源为 {label_text.get(label, label)}，"
            "时效边界以该日期证据为准，"
            "当前适用性需以正式 CRM 文档为准。"
        )

    return annotations


# ============================================================
# Prompt
# ============================================================

def build_prompt(evidence):

    source_blocks = []

    for index, source in enumerate(
        evidence["sources"], start=1
    ):

        role = evidence["role_by_key"].get(
            source["issue_key"], {}
        )

        source_blocks.append(
            f"""
------------------------------------------------------------
SOURCE {index}
------------------------------------------------------------

issue_key:
{source['issue_key']}

证据日期: {source['issue_date']}
temporal_label (Step 12.2 判定): {source['temporal_label']}
gate role: {role.get('role', '')}

question:
{source['question']}

supported_answer:
{source['supported_answer']}

supported_solution:
{source['supported_solution'] or '(无)'}
""".strip()
        )

    return f"""
请为下面这个 cluster 生成一条 Candidate KB entry。

============================================================
CLUSTER
============================================================

cluster_id:
{evidence['cluster_id']}

canonical_question (必须原样使用):
{evidence['canonical_question']}

gate 结构判定:
{evidence['structure']} (按 4 条独立排查路径处理)

Step 12.2 temporal 判定:
CURRENT_CONFIRMED (知识在最新证据点成立, 但必须带时效注记)

============================================================
GROUNDED SOURCES (按证据日期升序)
============================================================

{chr(10).join(source_blocks)}

============================================================
生成要求
============================================================

1. units 必须恰好 4 个, 每个 source 一个,
   source_issue_key 填对应 issue_key。

2. unit 的 content 必须紧贴该 source 的
   supported_answer / supported_solution,
   逐字或最小改写;
   禁止引入 source 中不存在的
   执行者、菜单路径、按钮、版本号、URL。

3. unit 的 title 概括该条排查路径;
   condition 如无明确来源条件则留空字符串。

4. steps 只能来自 supported_answer /
   supported_solution 中的操作内容。

5. summary_answer 概括 4 条路径
   (密码重置 / 访问链接 / 浏览器兼容 /
   临时故障等待处理),
   并说明它们是并列的不同原因。
   summary 中不得引入来源之外的细节。

6. 不要生成 notes / limitations /
   时效判断 — 这些由系统另行注入。

7. 输出 JSON:
{{
  "summary_answer": "...",
  "units": [
    {{
      "title": "...",
      "condition": "",
      "content": "...",
      "steps": ["..."],
      "source_issue_key": "ISSUE-CAND-..."
    }}
  ],
  "crm_feature": "回访/登录相关功能名",
  "problem_type": "troubleshooting",
  "knowledge_confidence": 0.0
}}
""".strip()


# ============================================================
# LLM 调用 (schema 修复重试, 与 43 同机制)
# ============================================================

def log_llm_error(raw, error):

    with ERROR_LOG.open(
        "a", encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70
            + "\n"
            + f"[{datetime.now().isoformat()}]\n"
            + f"ERROR: {error}\n"
            + "RAW RESPONSE:\n"
            + raw
            + "\n"
        )


def strip_json_fences(raw):

    text = raw.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*", "", text
        )

        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def build_client():

    return OpenAI(
        base_url=(
            "https://open.bigmodel.cn/api/paas/v4"
        ),
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


def judge_entry(client, evidence):

    prompt = build_prompt(evidence)

    last_error = None

    repair_note = None

    last_raw = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            started = time.time()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是 CRM 知识库的 KB 生成器。"
                        "只使用提供的 grounded source "
                        "文本，禁止发明任何来源之外的"
                        "内容。输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
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

            data = json.loads(raw)

            entry = GeneratedEntry.model_validate(
                data
            )

            if attempt > 1:
                log_llm_error(
                    raw,
                    f"RECOVERED_ON_ATTEMPT_{attempt}",
                )

            return entry, elapsed

        except Exception as exc:

            last_error = exc

            log_llm_error(
                last_raw or "(no response)",
                repr(exc),
            )

            repair_note = (
                "你上一次的输出未通过 schema 校验:\n"
                f"{exc}\n请修正字段名与类型后"
                "重新输出完整 JSON，"
                "不要输出 JSON 以外的文字。"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise last_error


# ============================================================
# deterministic 门
# ============================================================

def validate_generated(
    evidence, generated,
):

    flags = []

    valid_keys = {
        s["issue_key"]
        for s in evidence["sources"]
    }

    used_keys = set()

    unit_texts = []

    for index, unit in enumerate(
        generated.units, start=1
    ):

        key = clean_text(
            unit.source_issue_key
        )

        used_keys.add(key)

        if key not in valid_keys:

            flags.append(
                {
                    "flag": "INVENTED_SOURCE_KEY",
                    "severity": "hard",
                    "detail": (
                        f"unit {index}: {key}"
                    ),
                }
            )

            continue

        source = next(
            s
            for s in evidence["sources"]
            if s["issue_key"] == key
        )

        source_text = (
            source["supported_answer"]
            + "\n"
            + source["supported_solution"]
        )

        ratio = lexical_grounded_ratio(
            unit.content, source_text
        )

        unit_texts.append(
            (unit, source, ratio)
        )

        if ratio < 0.6:

            flags.append(
                {
                    "flag": "UNIT_LOW_GROUNDING",
                    "severity": "hard",
                    "detail": (
                        f"unit {index} ({key}) "
                        f"lexical ratio {ratio:.2f}"
                    ),
                }
            )

    missing = valid_keys - used_keys

    if missing:

        flags.append(
            {
                "flag": "SOURCE_COVERAGE_INCOMPLETE",
                "severity": "hard",
                "detail": ", ".join(
                    sorted(missing)
                ),
            }
        )

    if len(generated.units) != len(
        evidence["sources"]
    ):

        flags.append(
            {
                "flag": "UNIT_COUNT_MISMATCH",
                "severity": "hard",
                "detail": (
                    f"units={len(generated.units)} "
                    f"sources="
                    f"{len(evidence['sources'])}"
                ),
            }
        )

    hard_count = sum(
        1
        for f in flags
        if f["severity"] == "hard"
    )

    return {
        "flags": flags,
        "passed": hard_count == 0,
        "unit_texts": unit_texts,
    }


# ============================================================
# entry 组装
# ============================================================

def assemble_entry(
    evidence, generated, validation,
):

    limitations = build_notes_and_limitations(
        evidence
    )

    notes = [
        "以下"
        f"{len(generated.units)}"
        "条排查路径对应不同的原因，"
        "是并列的独立情况，并非冲突关系；"
        "各路径的时效边界见 limitations。"
    ]

    risky = [
        s
        for s in evidence["sources"]
        if s["temporal_status"] == "temporary"
        or s["resolution"]
        in {"partial", "unresolved"}
    ]

    ratios = [
        ratio
        for _, _, ratio in validation[
            "unit_texts"
        ]
    ]

    temporal = evidence["temporal_decision"]

    units = []

    for unit, source, ratio in validation[
        "unit_texts"
    ]:

        units.append(
            {
                "unit_type": "cause",
                "title": clean_text(unit.title),
                "condition": clean_text(
                    unit.condition
                ) or None,
                "content": clean_text(
                    unit.content
                ),
                "steps": list(unit.steps or []),
                "source_issue_keys": [
                    source["issue_key"]
                ],
            }
        )

    entry = {
        "cluster_id": TARGET_CLUSTER,
        "canonical_question": evidence[
            "canonical_question"
        ],
        "knowledge_structure": "troubleshooting",
        "generation_mode": "cause_items",
        "summary_answer": clean_text(
            generated.summary_answer
        ),
        "units": units,
        "unit_count": len(units),
        "notes": notes,
        "limitations": limitations,
        "crm_module": "CRM",
        "crm_feature": clean_text(
            generated.crm_feature
        ),
        "problem_type": clean_text(
            generated.problem_type
        ),
        "knowledge_confidence": float(
            generated.knowledge_confidence
        ),
        "source_issue_keys": [
            s["issue_key"]
            for s in evidence["sources"]
        ],
        "usable_source_count": len(
            evidence["sources"]
        ),
        "cited_source_count": len(units),
        "risky_source_count": len(risky),
        "min_lexical_grounding": round(
            min(ratios), 4
        )
        if ratios
        else 0.0,
        "avg_lexical_grounding": round(
            sum(ratios) / len(ratios), 4
        )
        if ratios
        else 0.0,
        "candidate_status": "candidate_ok"
        if validation["passed"]
        else "needs_review",
        "hard_flag_count": sum(
            1
            for f in validation["flags"]
            if f["severity"] == "hard"
        ),
        "hard_flags": [
            f["flag"]
            for f in validation["flags"]
            if f["severity"] == "hard"
        ],
        "soft_flags": [],
        "structural_flags": [],
        "request_seconds": validation.get(
            "request_seconds", 0.0
        ),
        "recheck_applied": False,
        "repair_applied": False,
        "repair_fields": [],
        "repair_reason": "",
        "repair_source_status": (
            "generated_in_step_12_3"
        ),
        "validator_false_positive_reviewed": False,
        "validator_false_positive_reason": "",
        "temporal_validity": {
            "source_step": "12.2",
            "final_decision": (
                temporal.get("final_decision")
            ),
            "unblock_for_candidate": (
                temporal.get(
                    "unblock_for_candidate"
                )
            ),
            "validated_at": temporal.get(
                "validated_at"
            ),
            "annotations_injected": len(
                evidence["annotations"]
            ),
            "note": (
                "Step 12.2 判定 CURRENT_CONFIRMED; "
                "时效注记逐字注入 limitations; "
                "最终版本裁决待正式 CRM 文档"
            ),
        },
    }

    return entry


# ============================================================
# 输出
# ============================================================

def write_v4(entry):

    v3_lines = [
        line
        for line in CANDIDATE_V3.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    output_lines = v3_lines + [
        json.dumps(entry, ensure_ascii=False)
    ]

    OUTPUT_JSONL.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8",
    )

    records = [
        json.loads(line)
        for line in output_lines
    ]

    rows = []

    for item in records:

        rows.append(
            {
                "cluster_id": item.get(
                    "cluster_id"
                ),
                "canonical_question": item.get(
                    "canonical_question"
                ),
                "knowledge_structure": item.get(
                    "knowledge_structure"
                ),
                "generation_mode": item.get(
                    "generation_mode"
                ),
                "summary_answer": item.get(
                    "summary_answer"
                ),
                "unit_count": item.get(
                    "unit_count"
                ),
                "usable_source_count": item.get(
                    "usable_source_count"
                ),
                "risky_source_count": item.get(
                    "risky_source_count"
                ),
                "candidate_status": item.get(
                    "candidate_status"
                ),
                "limitations": " | ".join(
                    item.get("limitations") or []
                ),
                "merged_from_clusters":
                " | ".join(
                    item.get(
                        "merged_from_clusters"
                    ) or []
                ),
                "temporal_unblock": bool(
                    item.get("temporal_validity")
                ),
            }
        )

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        pd.DataFrame(rows).to_excel(
            writer,
            sheet_name="entries",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "metric": "generated_cluster",
                    "value": TARGET_CLUSTER,
                },
                {
                    "metric": "model",
                    "value": MODEL,
                },
                {
                    "metric": "units",
                    "value": str(
                        entry.get("unit_count")
                    ),
                },
                {
                    "metric": (
                        "limitations_injected"
                    ),
                    "value": str(
                        len(
                            entry.get(
                                "limitations"
                            ) or []
                        )
                    ),
                },
                {
                    "metric": "generated_at",
                    "value": entry[
                        "temporal_validity"
                    ].get("validated_at", ""),
                },
            ]
        ).to_excel(
            writer,
            sheet_name="generation_info",
            index=False,
        )

    return len(records)


def main():

    print("=" * 70)
    print(
        "Step 12.3 - Temporal-Unblocked Candidate:"
        f" {TARGET_CLUSTER}"
    )
    print(f"Model: {MODEL}")
    print("=" * 70)

    evidence = assemble_evidence()

    facts = (
        evidence["temporal_decision"].get(
            "facts"
        ) or {}
    )

    print()
    print(
        f"sources: {len(evidence['sources'])} "
        f"(temporal decision: "
        f"{evidence['temporal_decision'].get('final_decision')})"
    )

    for source in evidence["sources"]:

        print(
            f"  [{source['issue_date']}] "
            f"{source['temporal_label']:19} "
            f"{source['issue_key']:32} "
            f"{source['question'][:36]}"
        )

    print()
    print(
        f"时效注记注入: "
        f"{len(evidence['annotations'])} 条"
    )

    if DRY_RUN:

        print()
        print("DRY RUN 结束，未调用 LLM，未写输出。")
        return

    if not os.environ.get("OPENAI_API_KEY"):

        raise SystemExit(
            "缺少 OPENAI_API_KEY 环境变量"
        )

    client = build_client()

    generated, elapsed = judge_entry(
        client, evidence
    )

    validation = validate_generated(
        evidence, generated
    )

    validation["request_seconds"] = elapsed

    print()
    print(
        f"generated: units={len(generated.units)} "
        f"| gate_passed={validation['passed']}"
    )

    for flag in validation["flags"]:

        print(
            f"  [{flag['severity']}] "
            f"{flag['flag']}: {flag['detail']}"
        )

    entry = assemble_entry(
        evidence, generated, validation
    )

    print()
    print(f"summary: {entry['summary_answer']}")
    print()
    print("limitations (注入):")

    for item in entry["limitations"]:
        print(f"  - {item[:90]}")

    if not validation["passed"]:

        print()
        print(
            "[WARN] deterministic 门未通过，"
            "entry 仍写入 v4 但标记 needs_review;"
            " 以 39 v4 验证结果为准。"
        )

    total = write_v4(entry)

    print()
    print("=" * 70)
    print(f"完成: candidate v4 共 {total} 条 entry")
    print("=" * 70)
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Excel: {OUTPUT_XLSX}")
    print()
    print("下一步: 39 v4 验证")
    print(
        ".venv/bin/python "
        "scripts/39_validate_mixed_candidate_grounding.py "
        "--candidate output/kb_entries_mixed_candidate_v4.jsonl"
    )


if __name__ == "__main__":
    main()
