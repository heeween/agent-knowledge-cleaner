#!/usr/bin/env python3
"""
Step 12.2 - Temporal Validity 处理
==================================

对 4 个 material temporal cluster:

    QCLUSTER-0008  登录失败 (2024-06 → 2026-05)
    QCLUSTER-0074  回访录音 (2026-06, 单条)
    QCLUSTER-0139  车主画像 (2024-11 → 2026-05)
    QCLUSTER-0194  验证过期 (2024-07 → 2024-09)

基于 Step 12.1 的真实 timestamp 做
current vs historical / incident 判定。

原则 (PROJECT_HANDOFF 18):

- current vs historical 不能只看 temporal_status,
  必须用真实 source timestamps 排序
- 不根据 temporary/stable/future_plan 推断先后
- 不同原因 ≠ conflict
- 不同 workaround ≠ conflict
- temporary ≠ conflict

本步骤是初步 temporal 判定;
最终版本裁决仍需 38 份正式 CRM 文档 (后续步骤)。

结构:

1. deterministic 证据组装 (成员按真实时间排序)
2. LLM (glm-5.3-flash) 逐 cluster 判定
3. deterministic 时效门:
   - 引用 quote 必须词面 grounded
   - 决策中出现的日期必须来自证据
   - 引用 issue_key 必须是成员
   - 决策必须与时间事实兼容
     (例如 CURRENT_CONFIRMED 必须有 recent
      且 stable/resolved 证据)
   - 任何 gate 失败 → 回落 keep_blocked

输出 (不修改任何冻结输入):

    output/temporal_validity_decisions.jsonl
    output/temporal_validity_decisions.xlsx

用法:

    .venv/bin/python scripts/43_validate_temporal_validity.py
    .venv/bin/python scripts/43_validate_temporal_validity.py --dry-run
    .venv/bin/python scripts/43_validate_temporal_validity.py --only QCLUSTER-0008
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "output"

TIMESTAMPS_FILE = OUTPUT_DIR / "issue_timestamps.jsonl"

GROUNDING_FILE = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

STRUCTURE_FILE = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "temporal_validity_decisions.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "temporal_validity_decisions.xlsx"
)

MODEL = "glm-5.3-flash"

MAX_RETRIES = 4

REQUEST_TIMEOUT = 180

TEMPORAL_CLUSTER_IDS = [
    "QCLUSTER-0008",
    "QCLUSTER-0074",
    "QCLUSTER-0139",
    "QCLUSTER-0194",
]

# recent 定义: 距语料最后证据日 12 个月内
RECENT_DAYS = 365

DRY_RUN = "--dry-run" in sys.argv

ONLY_IDS = set()

for _index, _arg in enumerate(sys.argv):

    if _arg == "--only" and _index + 1 < len(
        sys.argv
    ):
        ONLY_IDS = {
            item.strip()
            for item in sys.argv[
                _index + 1
            ].split(",")
            if item.strip()
        }


# ============================================================
# 工具 (词面 grounding 与 38/39 同规则)
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


def contains_grounded_phrase(phrase, source_text):

    """
    与 38 的 contains_grounded_phrase 同规则:

    长短语 (>=6 规范化字符):
      原文命中 或 字符覆盖率 >= 0.85
    短语:
      覆盖率 == 1.0 且至少一个 2-gram 命中
    """

    phrase_n = normalize_text(phrase)

    source_n = normalize_text(source_text)

    if not phrase_n or not source_n:
        return False

    if phrase_n in source_n:
        return True

    chars = set(phrase_n)

    overlap = (
        len(chars & set(source_n)) / len(chars)
    )

    if len(phrase_n) >= 6:
        return overlap >= 0.85

    if overlap < 1.0:
        return False

    phrase_bigrams = shingles(phrase_n)

    source_bigrams = shingles(source_n)

    return bool(
        phrase_bigrams & source_bigrams
    )


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


# ============================================================
# LLM Schema
# ============================================================

class MemberAssessment(BaseModel):

    issue_key: str

    temporal_label: Literal[
        "current",
        "historical",
        "temporary_incident",
        "uncertain",
    ]

    claim_summary: str

    quote: str = Field(
        description="来自该成员 answer/solution 的逐字引用"
    )


class DimensionConflict(BaseModel):

    dimension: str

    assessment: Literal[
        "different_conditions",
        "version_change",
        "true_conflict",
    ]

    evidence_a: dict

    evidence_b: dict


class TemporalDecision(BaseModel):

    knowledge_claim: str

    member_assessments: List[MemberAssessment]

    answer_consistency: Literal[
        "consistent",
        "partial_overlap",
        "divergent",
    ]

    fact_dimension_conflicts: List[
        DimensionConflict
    ] = []

    temporal_decision: Literal[
        "CURRENT_CONFIRMED",
        "HISTORICAL_SUPERSEDED",
        "INCIDENT_TEMPORARY",
        "STALE_NO_RECENT_CONFIRMATION",
        "VERSION_SUSPECTED_NEEDS_DOCS",
    ]

    required_annotations: List[str] = []

    reason: str

    confidence: float

    @field_validator(
        "confidence", mode="before"
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


def normalize_llm_payload(data):

    """
    deterministic 归一化 LLM 返回:
    枚举大小写/空白、confidence 字符串。
    不改变语义, 只修格式偏差。
    """

    if not isinstance(data, dict):
        return data

    for field, style in (
        ("temporal_decision", "upper"),
        ("answer_consistency", "lower"),
    ):

        value = data.get(field)

        if isinstance(value, str):

            data[field] = (
                value.strip().upper()
                if style == "upper"
                else value.strip().lower()
            )

    for member in (
        data.get("member_assessments") or []
    ):

        if isinstance(member, dict) and isinstance(
            member.get("temporal_label"), str
        ):
            member["temporal_label"] = member[
                "temporal_label"
            ].strip().lower()

    for conflict in (
        data.get("fact_dimension_conflicts")
        or []
    ):

        if isinstance(conflict, dict) and (
            isinstance(
                conflict.get("assessment"), str
            )
        ):
            conflict["assessment"] = conflict[
                "assessment"
            ].strip().lower()

    return data


# ============================================================
# 输入加载与证据组装
# ============================================================

def load_timestamps():

    by_key = {}

    for row in load_jsonl_file(TIMESTAMPS_FILE):
        by_key[clean_text(row.get("issue_key"))] = (
            row
        )

    return by_key


def load_jsonl_file(path):

    records = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


def load_alignment_rows():

    df = pd.read_excel(
        GROUNDING_FILE,
        sheet_name="all_sources",
    )

    return {
        clean_text(row.get("issue_key")): row
        for row in df.to_dict(orient="records")
    }


def load_structure_rows():

    df = pd.read_excel(
        STRUCTURE_FILE,
        sheet_name="all_clusters",
    )

    return {
        clean_text(row.get("cluster_id")): row
        for row in df.to_dict(orient="records")
    }


def assemble_evidence(
    cluster_id,
    timestamps,
    alignment_rows,
    structure_rows,
    corpus_end,
):

    members = []

    for issue_key, row in timestamps.items():

        if clean_text(
            row.get("temporal_cluster")
        ) != cluster_id:
            continue

        alignment = alignment_rows.get(
            issue_key, {}
        )

        start = parse_ts(
            row.get("issue_time_min")
        )

        members.append(
            {
                "issue_key": issue_key,
                "issue_time_min": clean_text(
                    row.get("issue_time_min")
                ),
                "issue_time_max": clean_text(
                    row.get("issue_time_max")
                ),
                "_start": start,
                "question": clean_text(
                    alignment.get("question")
                )
                or clean_text(
                    row.get("question_normalized")
                ),
                "supported_answer": clean_text(
                    alignment.get("supported_answer")
                ),
                "supported_solution": clean_text(
                    alignment.get(
                        "supported_solution"
                    )
                ),
                "temporal_status": clean_text(
                    alignment.get("temporal_status")
                )
                or clean_text(
                    row.get("temporal_status")
                ),
                "resolution": clean_text(
                    alignment.get("resolution")
                )
                or clean_text(row.get("resolution")),
            }
        )

    members.sort(key=lambda m: m["_start"])

    cutoff = corpus_end - timedelta(
        days=RECENT_DAYS
    )

    for member in members:

        member["is_recent"] = bool(
            member["_start"]
            and member["_start"] >= cutoff
        )

    span_days = (
        (
            parse_ts(members[-1]["issue_time_max"])
            - parse_ts(members[0]["issue_time_min"])
        ).days
        if len(members) >= 2
        else 0
    )

    facts = {
        "member_count": len(members),
        "corpus_end": corpus_end.strftime(
            "%Y-%m-%d"
        ),
        "recent_cutoff": cutoff.strftime(
            "%Y-%m-%d"
        ),
        "has_recent": any(
            m["is_recent"] for m in members
        ),
        "all_stale": all(
            not m["is_recent"] for m in members
        ),
        "recent_stable_or_resolved_exists": any(
            m["is_recent"]
            and (
                m["temporal_status"] == "stable"
                or m["resolution"] == "resolved"
            )
            for m in members
        ),
        "span_days": span_days,
    }

    structure = structure_rows.get(
        cluster_id, {}
    )

    return {
        "cluster_id": cluster_id,
        "structure": clean_text(
            structure.get("structure")
        ),
        "canonical_question": clean_text(
            structure.get("canonical_question")
        ),
        "members": members,
        "facts": facts,
    }


# ============================================================
# Prompt
# ============================================================

def build_prompt(evidence):

    member_blocks = []

    for index, member in enumerate(
        evidence["members"], start=1
    ):

        recent = (
            "是"
            if member["is_recent"]
            else f"否 (早于 {evidence['facts']['recent_cutoff']})"
        )

        member_blocks.append(
            f"""
------------------------------------------------------------
MEMBER {index}
------------------------------------------------------------

issue_key:
{member['issue_key']}

真实消息时间:
{member['issue_time_min']} → {member['issue_time_max']}
是否 recent (按真实时间判定): {recent}

question:
{member['question']}

supported_answer:
{member['supported_answer']}

supported_solution:
{member['supported_solution'] or '(无)'}

temporal_status: {member['temporal_status']}
resolution: {member['resolution']}
""".strip()
        )

    facts = evidence["facts"]

    return f"""
请对下面这个 cluster 做 temporal validity 判定。

============================================================
CLUSTER
============================================================

cluster_id:
{evidence['cluster_id']}

canonical_question:
{evidence['canonical_question']}

gate 结构判定:
{evidence['structure']}

成员数:
{facts['member_count']}

语料最后证据日:
{facts['corpus_end']}

recent 定义:
真实时间 >= {facts['recent_cutoff']} (语料最后证据日前 {RECENT_DAYS} 天)

deterministic 时间事实 (已按真实 timestamp 计算, 不可推翻):
- has_recent_member: {facts['has_recent']}
- all_stale: {facts['all_stale']}
- 存在 recent 且 stable/resolved 的成员: {facts['recent_stable_or_resolved_exists']}
- 时间跨度天数: {facts['span_days']}

============================================================
MEMBERS (按真实时间升序)
============================================================

{chr(10).join(member_blocks)}

============================================================
判定要求
============================================================

1. 排序只依据上面的真实消息时间,
   不依据 temporal_status 字段。

2. quote 必须逐字摘自该成员的
   supported_answer / supported_solution,
   不得改写、拼接或概括。

3. 日期只能使用证据中出现的日期,
   不得推断或发明任何版本号、时间、规则。

4. 不同原因 ≠ conflict;
   不同 workaround ≠ conflict;
   temporary ≠ conflict。
   fact_dimension_conflicts 只登记:
   同一事实维度 + 同一条件 + 结论互斥的情况,
   并区分 different_conditions / version_change /
   true_conflict。

5. temporal_decision 判定标准:

   - CURRENT_CONFIRMED:
     有 recent 成员, 且答案跨时间收敛,
     知识在最新证据点仍成立

   - HISTORICAL_SUPERSEDED:
     旧证据已被新证据取代
     (如旧 Bug 解释 vs 新版本行为)

   - INCIDENT_TEMPORARY:
     当前证据是临时故障 / 待修复,
     知识是时间盒, 修复后失效

   - STALE_NO_RECENT_CONFIRMATION:
     全部证据已过 old (deterministic:
     all_stale = true), 无法确认 current

   - VERSION_SUSPECTED_NEEDS_DOCS:
     同一事实维度在不同时间出现
     不一致结论, 需要正式文档裁决

6. required_annotations 只在 CURRENT_CONFIRMED
   时填写: 每条注记必须引用具体日期 (证据中的)
   或逐字短语, 说明该结论的时效边界。

7. 判定是初步 temporal 判定,
   最终版本裁决将由正式 CRM 文档完成。
""".strip()


# ============================================================
# Deterministic 门
# ============================================================

def member_source_text(member):

    return (
        member.get("supported_answer", "")
        + "\n"
        + member.get("supported_solution", "")
    )


def validate_decision(evidence, decision):

    flags = []

    members_by_key = {
        m["issue_key"]: m
        for m in evidence["members"]
    }

    facts = evidence["facts"]

    # 1. 成员引用合法性 + quote grounding
    for assessment in (
        decision.member_assessments
    ):

        key = clean_text(assessment.issue_key)

        member = members_by_key.get(key)

        if member is None:

            flags.append(
                {
                    "flag": "CITED_KEY_NOT_MEMBER",
                    "severity": "hard",
                    "detail": key,
                }
            )

            continue

        source_text = member_source_text(
            member
        )

        quote = clean_text(assessment.quote)

        if not quote:

            flags.append(
                {
                    "flag": "QUOTE_EMPTY",
                    "severity": "hard",
                    "detail": key,
                }
            )

        elif not contains_grounded_phrase(
            quote, source_text
        ):

            flags.append(
                {
                    "flag": "QUOTE_NOT_GROUNDED",
                    "severity": "hard",
                    "detail": (
                        f"{key}: {quote[:50]}"
                    ),
                }
            )

    # 2. fact_dimension_conflicts 的 quote grounding
    for conflict in (
        decision.fact_dimension_conflicts
    ):

        for side in (
            conflict.evidence_a,
            conflict.evidence_b,
        ):

            key = clean_text(
                side.get("issue_key")
            )

            member = members_by_key.get(key)

            if member is None:

                flags.append(
                    {
                        "flag": "CITED_KEY_NOT_MEMBER",
                        "severity": "hard",
                        "detail": key,
                    }
                )

                continue

            quote = clean_text(
                side.get("quote")
            )

            if quote and not (
                contains_grounded_phrase(
                    quote,
                    member_source_text(member),
                )
            ):

                flags.append(
                    {
                        "flag": "QUOTE_NOT_GROUNDED",
                        "severity": "hard",
                        "detail": (
                            f"{key}: {quote[:50]}"
                        ),
                    }
                )

    # 3. 日期合法性: 决策文本中的日期必须在证据中
    evidence_dates = set()

    for member in evidence["members"]:

        for field in (
            "issue_time_min",
            "issue_time_max",
        ):

            text = clean_text(
                member.get(field)
            )

            if text:
                evidence_dates.add(text[:10])

    def check_dates(text, location):

        for match in re.findall(
            r"\d{4}-\d{2}(?:-\d{2})?",
            clean_text(text),
        ):

            date = (
                match
                if len(match) == 10
                else match
            )

            if date not in evidence_dates and (
                len(match) == 10
                or not any(
                    d.startswith(match)
                    for d in evidence_dates
                )
            ):

                flags.append(
                    {
                        "flag": "DATE_NOT_IN_EVIDENCE",
                        "severity": "hard",
                        "detail": (
                            f"{location}: {match}"
                        ),
                    }
                )

    check_dates(
        decision.knowledge_claim, "knowledge_claim"
    )

    for annotation in (
        decision.required_annotations
    ):

        check_dates(annotation, "annotation")

    # 4. 决策与时间事实兼容性
    cited = [
        clean_text(a.issue_key)
        for a in decision.member_assessments
    ]

    gate_failures = []

    if (
        decision.temporal_decision
        == "CURRENT_CONFIRMED"
    ):

        if not facts["has_recent"]:

            gate_failures.append(
                "CURRENT_CONFIRMED 但无 recent 成员"
            )

        if not facts[
            "recent_stable_or_resolved_exists"
        ]:

            gate_failures.append(
                "CURRENT_CONFIRMED 但 recent 成员"
                "均非 stable/resolved"
            )

        if not decision.required_annotations:

            gate_failures.append(
                "CURRENT_CONFIRMED 但未提供"
                "时效注记"
            )

    elif (
        decision.temporal_decision
        == "STALE_NO_RECENT_CONFIRMATION"
    ):

        if not facts["all_stale"]:

            gate_failures.append(
                "STALE 但存在 recent 成员"
            )

    elif (
        decision.temporal_decision
        == "INCIDENT_TEMPORARY"
    ):

        cited_members = [
            members_by_key[key]
            for key in cited
            if key in members_by_key
        ]

        targets = (
            cited_members
            if cited_members
            else evidence["members"]
        )

        if any(
            m["temporal_status"] != "temporary"
            for m in targets
        ):

            gate_failures.append(
                "INCIDENT_TEMPORARY 但引用成员"
                "含非 temporary"
            )

    elif (
        decision.temporal_decision
        == "HISTORICAL_SUPERSEDED"
    ):

        if facts["member_count"] < 2:

            gate_failures.append(
                "HISTORICAL_SUPERSEDED 但只有"
                "1 个成员"
            )

        if facts["span_days"] < RECENT_DAYS:

            gate_failures.append(
                "HISTORICAL_SUPERSEDED 但时间跨度"
                f"不足 {RECENT_DAYS} 天"
            )

    for failure in gate_failures:

        flags.append(
            {
                "flag": "DECISION_GATE_FAIL",
                "severity": "hard",
                "detail": failure,
            }
        )

    hard_count = sum(
        1
        for item in flags
        if item["severity"] == "hard"
    )

    gate_passed = hard_count == 0

    final_decision = (
        decision.temporal_decision
        if gate_passed
        else "GATE_FAILED_KEEP_BLOCKED"
    )

    unblock = (
        gate_passed
        and final_decision == "CURRENT_CONFIRMED"
    )

    return {
        "flags": flags,
        "gate_passed": gate_passed,
        "final_decision": final_decision,
        "unblock_for_candidate": unblock,
    }


# ============================================================
# LLM 调用
# ============================================================

def build_client():

    return OpenAI(
        base_url=(
            "https://open.bigmodel.cn/api/paas/v4"
        ),
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


ERROR_LOG = (
    OUTPUT_DIR
    / "temporal_validity_llm_errors.log"
)

ENUM_VALUES = {
    "temporal_decision": [
        "CURRENT_CONFIRMED",
        "HISTORICAL_SUPERSEDED",
        "INCIDENT_TEMPORARY",
        "STALE_NO_RECENT_CONFIRMATION",
        "VERSION_SUSPECTED_NEEDS_DOCS",
    ],
    "answer_consistency": [
        "consistent",
        "partial_overlap",
        "divergent",
    ],
    "temporal_label": [
        "current",
        "historical",
        "temporary_incident",
        "uncertain",
    ],
    "assessment": [
        "different_conditions",
        "version_change",
        "true_conflict",
    ],
}


def log_llm_error(cluster_id, raw, error):

    with ERROR_LOG.open(
        "a", encoding="utf-8"
    ) as f:

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


def build_repair_note(error):

    enum_hint = "; ".join(
        f"{field}: {'/'.join(values)}"
        for field, values in (
            ENUM_VALUES.items()
        )
    )

    return (
        "你上一次的输出未通过 schema 校验，错误如下:\n"
        f"{error}\n\n"
        "请重新输出完整 JSON，要求:\n"
        "1. 字段名与类型与要求完全一致;\n"
        f"2. 枚举取值必须严格使用以下之一 ({enum_hint});\n"
        "3. 不要输出 JSON 以外的任何文字。"
    )


def judge_cluster(client, evidence):

    prompt = build_prompt(evidence)

    cluster_id = evidence["cluster_id"]

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
                        "你是 CRM 知识库的 "
                        "temporal validity 审核员。"
                        "只基于提供的证据判断，"
                        "引用必须逐字，日期必须来自"
                        "证据。不同原因、不同 "
                        "workaround、temporary 都"
                        "不是 conflict。输出 JSON。"
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

            data = normalize_llm_payload(data)

            decision = (
                TemporalDecision.model_validate(
                    data
                )
            )

            if attempt > 1:

                log_llm_error(
                    cluster_id,
                    raw,
                    f"RECOVERED_ON_ATTEMPT_{attempt}",
                )

            return decision, elapsed

        except Exception as exc:

            last_error = exc

            log_llm_error(
                cluster_id,
                last_raw or "(no response)",
                repr(exc),
            )

            repair_note = build_repair_note(
                repr(exc)
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise last_error


# ============================================================
# 输出
# ============================================================

def export_excel(results):

    summary_rows = []

    decision_rows = []

    member_rows = []

    flag_rows = []

    for result in results:

        evidence = result["evidence"]

        validation = result["validation"]

        decision = result["decision"]

        facts = evidence["facts"]

        summary_rows.append(
            {
                "cluster_id": evidence[
                    "cluster_id"
                ],
                "members": facts["member_count"],
                "time_span_days": facts[
                    "span_days"
                ],
                "has_recent": facts["has_recent"],
                "llm_decision": (
                    decision.temporal_decision
                    if decision
                    else ""
                ),
                "final_decision": validation[
                    "final_decision"
                ],
                "unblock_for_candidate":
                validation[
                    "unblock_for_candidate"
                ],
                "gate_passed": validation[
                    "gate_passed"
                ],
                "hard_flags": sum(
                    1
                    for f in validation["flags"]
                    if f["severity"] == "hard"
                ),
                "confidence": (
                    decision.confidence
                    if decision
                    else 0.0
                ),
            }
        )

        decision_rows.append(
            {
                "cluster_id": evidence[
                    "cluster_id"
                ],
                "canonical_question": evidence[
                    "canonical_question"
                ],
                "structure": evidence[
                    "structure"
                ],
                "knowledge_claim": (
                    decision.knowledge_claim
                    if decision
                    else ""
                ),
                "answer_consistency": (
                    decision.answer_consistency
                    if decision
                    else ""
                ),
                "fact_dimension_conflicts": json.dumps(
                    [
                        c.model_dump()
                        for c in (
                            decision.fact_dimension_conflicts
                            if decision
                            else []
                        )
                    ],
                    ensure_ascii=False,
                ),
                "required_annotations": " | ".join(
                    decision.required_annotations
                    if decision
                    else []
                ),
                "reason": (
                    decision.reason
                    if decision
                    else ""
                ),
                "final_decision": validation[
                    "final_decision"
                ],
                "unblock_for_candidate":
                validation[
                    "unblock_for_candidate"
                ],
            }
        )

        if decision:

            for assessment in (
                decision.member_assessments
            ):

                member_rows.append(
                    {
                        "cluster_id": evidence[
                            "cluster_id"
                        ],
                        "issue_key": clean_text(
                            assessment.issue_key
                        ),
                        "temporal_label": (
                            assessment.temporal_label
                        ),
                        "claim_summary": (
                            assessment.claim_summary
                        ),
                        "quote": assessment.quote,
                        "quote_grounded": not any(
                            f["flag"]
                            == "QUOTE_NOT_GROUNDED"
                            and clean_text(
                                assessment.issue_key
                            )
                            in f["detail"]
                            for f in validation[
                                "flags"
                            ]
                        ),
                    }
                )

        for flag in validation["flags"]:

            flag_rows.append(
                {
                    "cluster_id": evidence[
                        "cluster_id"
                    ],
                    **flag,
                }
            )

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="summary", index=False
        )

        pd.DataFrame(decision_rows).to_excel(
            writer,
            sheet_name="decisions",
            index=False,
        )

        pd.DataFrame(member_rows).to_excel(
            writer,
            sheet_name="member_assessments",
            index=False,
        )

        pd.DataFrame(flag_rows).to_excel(
            writer,
            sheet_name="validation_flags",
            index=False,
        )


def main():

    print("=" * 70)
    print("Step 12.2 - Temporal Validity")
    print(f"Model: {MODEL}")
    print("=" * 70)

    timestamps = load_timestamps()

    alignment_rows = load_alignment_rows()

    structure_rows = load_structure_rows()

    # recent 判定基于全语料最后证据日,
    # 不是各 cluster 自己的最新时间
    corpus_end = max(
        parse_ts(row.get("issue_time_max"))
        for row in timestamps.values()
    )

    print(
        f"语料最后证据日: "
        f"{corpus_end.strftime('%Y-%m-%d')} "
        f"(recent cutoff: "
        f"{(corpus_end - timedelta(days=RECENT_DAYS)).strftime('%Y-%m-%d')})"
    )

    evidences = []

    for cluster_id in TEMPORAL_CLUSTER_IDS:

        if ONLY_IDS and (
            cluster_id not in ONLY_IDS
        ):
            continue

        evidence = assemble_evidence(
            cluster_id,
            timestamps,
            alignment_rows,
            structure_rows,
            corpus_end,
        )

        if not evidence["members"]:
            raise RuntimeError(
                f"{cluster_id} 无成员"
            )

        evidences.append(evidence)

    if DRY_RUN:

        for evidence in evidences:

            facts = evidence["facts"]

            print()
            print(
                f"{evidence['cluster_id']} "
                f"({facts['member_count']} members, "
                f"span {facts['span_days']}d)"
            )

            for member in evidence["members"]:

                print(
                    f"  [{member['issue_time_min']}] "
                    f"recent={member['is_recent']} "
                    f"{member['temporal_status']}/"
                    f"{member['resolution']}  "
                    f"{member['question'][:40]}"
                )

        print()
        print("DRY RUN 结束，未调用 LLM，未写输出。")
        return

    if not os.environ.get("OPENAI_API_KEY"):

        raise SystemExit(
            "缺少 OPENAI_API_KEY 环境变量"
        )

    client = build_client()

    results = []

    for evidence in evidences:

        cluster_id = evidence["cluster_id"]

        print()
        print(f"--- {cluster_id} ---")

        decision, elapsed = judge_cluster(
            client, evidence
        )

        validation = validate_decision(
            evidence, decision
        )

        results.append(
            {
                "cluster_id": cluster_id,
                "evidence": evidence,
                "decision": decision,
                "validation": validation,
                "request_seconds": elapsed,
                "model": MODEL,
                "validated_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        )

        print(
            f"  LLM: {decision.temporal_decision}"
        )

        print(
            f"  FINAL: "
            f"{validation['final_decision']} "
            f"| unblock="
            f"{validation['unblock_for_candidate']}"
        )

        for flag in validation["flags"]:

            print(
                f"  [{flag['severity']}] "
                f"{flag['flag']}: {flag['detail']}"
            )

    OUTPUT_JSONL.write_text(
        "\n".join(
            json.dumps(
                {
                    "cluster_id": result[
                        "cluster_id"
                    ],
                    "final_decision": result[
                        "validation"
                    ][
                        "final_decision"
                    ],
                    "unblock_for_candidate": result[
                        "validation"
                    ][
                        "unblock_for_candidate"
                    ],
                    "gate_passed": result[
                        "validation"
                    ][
                        "gate_passed"
                    ],
                    "decision": result[
                        "decision"
                    ].model_dump(),
                    "flags": result["validation"][
                        "flags"
                    ],
                    "facts": result["evidence"][
                        "facts"
                    ],
                    "members": [
                        {
                            key: value
                            for key, value in member.items()
                            if key != "_start"
                        }
                        for member in result[
                            "evidence"
                        ]["members"]
                    ],
                    "model": result["model"],
                    "request_seconds": result[
                        "request_seconds"
                    ],
                    "validated_at": result[
                        "validated_at"
                    ],
                },
                ensure_ascii=False,
            )
            for result in results
        )
        + "\n",
        encoding="utf-8",
    )

    export_excel(results)

    print()
    print("=" * 70)

    unblocked = [
        result["cluster_id"]
        for result in results
        if result["validation"][
            "unblock_for_candidate"
        ]
    ]

    print(
        f"完成: {len(results)} clusters | "
        f"unblock: {unblocked or '无'}"
    )

    print("=" * 70)
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Excel: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
