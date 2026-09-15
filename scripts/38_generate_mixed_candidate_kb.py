"""
Step 11.4 - Mixed Cluster Candidate KB Generation

输入（全部为已冻结 / 已验证产物）：

output/mixed_structure_classifications_v3_1.xlsx
  - all_clusters   : 只使用 generation_ready == true
  - evidence_roles : 提供 Gate 已验证的 scenario_label

output/mixed_source_qa_alignments_v2.xlsx
  - all_sources    : 只使用 usable_for_kb == true
  - 只把 supported_answer / supported_solution 交给模型

output/mixed_cluster_audits_v2.xlsx
  - all_clusters   : 提供 cluster 上下文

输出：

output/kb_entries_mixed_candidate.jsonl
output/kb_entries_mixed_candidate.xlsx

重要：

本步骤只产出 Candidate KB。

后面仍然必须有：
- grounding validation
- temporal / conflict handling
- final publishability gate

本脚本不调用任何已冻结 KB 文件，
不修改 74 条正式 KB。
"""

import os
import re
import sys
import json
import time
import threading
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "output"

STRUCTURE_FILE = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.xlsx"
)

GROUNDING_FILE = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

MIXED_AUDIT_FILE = (
    OUTPUT_DIR
    / "mixed_cluster_audits_v2.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5

MAX_RETRIES = 4

REQUEST_TIMEOUT = 120

# 词面 grounding 比例低于该值只做 soft flag
LEXICAL_GROUNDING_THRESHOLD = 0.35

DRY_RUN = "--dry-run" in sys.argv

RECHECK = "--recheck" in sys.argv

REDO_IDS = set()

for _index, _arg in enumerate(sys.argv):

    if _arg == "--redo" and _index + 1 < len(sys.argv):

        REDO_IDS = {
            item.strip()
            for item in sys.argv[_index + 1].split(",")
            if item.strip()
        }


# ============================================================
# Schema
# ============================================================

class KnowledgeUnit(BaseModel):

    unit_type: Literal[
        "answer",
        "section",
        "cause",
        "scenario",
    ]

    title: str

    condition: str | None = None

    content: str

    steps: list[str] = Field(
        default_factory=list
    )

    source_issue_keys: list[str]


class MixedCandidateEntry(BaseModel):

    canonical_question: str

    summary_answer: str

    units: list[KnowledgeUnit]

    notes: list[str] = Field(
        default_factory=list
    )

    limitations: list[str] = Field(
        default_factory=list
    )

    crm_module: str

    crm_feature: str

    problem_type: str

    knowledge_confidence: float

    source_issue_keys: list[str]


# ============================================================
# Mode 规则
# ============================================================

EXPECTED_UNIT_TYPE = {
    "single_answer": "answer",
    "howto_sections": "section",
    "cause_items": "cause",
    "scenario_sections": "scenario",
}

# 结构数量必须与 Gate 对齐的 mode
STRICT_COUNT_MODES = {
    "cause_items",
    "scenario_sections",
}

# 只允许这些 hard flag 之外的算 soft
SOFT_FLAGS = {
    "WEAK_LEXICAL_GROUNDING",
    "TEMPORAL_CAUTION_MISSING",
    "HOWTO_UNIT_COUNT_LOW",
    "SINGLE_SOURCE_ENTRY",
}


MODE_INSTRUCTIONS = {

    "single_answer": """
生成模式：single_answer

要求：

- units 只能有 1 个
- unit_type = answer
- title = 该问题的简短标题
- content = 归纳后的唯一答案
- steps 可以为空
""".strip(),

    "howto_sections": """
生成模式：howto_sections

要求：

- 每个 unit_type = section
- 每个 section 对应一条 source 提供的
  操作路径 / 处理分支 / fallback
- title = section 名称
  例如：标准操作 / 异常状态下的处理方式
- condition = 该 section 的适用条件
  只有在 source 明确写出条件时才填写，
  否则填 null
- content = 该 section 的说明
- steps = 该 section 的操作步骤
- 不允许把不同 section 合并成一段
- 不允许新增 source 中不存在的 section
""".strip(),

    "cause_items": """
生成模式：cause_items

要求：

- 每个 unit_type = cause
- title = 独立原因
- content = 该原因对应的检查方法或处理方法
- steps = 可选的排查步骤
- 原因数量必须等于 Gate 给出的
  independent_cause_count
- 即使这些原因全部来自同一条 source，
  也必须按 independent_cause_count
  拆成对应数量的 cause unit，
  并让这些 unit 引用同一个 issue_key
- 不允许合并两个独立原因
- 不允许新增 Gate 数量之外的原因
- 不允许把多余的原因写进 summary_answer / notes
  来代替 cause unit
- 不同原因不是冲突，
  不要在 notes 里描述成冲突
""".strip(),

    "scenario_sections": """
生成模式：scenario_sections

要求：

- 每个 unit_type = scenario
- title 必须直接使用 Gate 已验证的
  scenario_label，不允许改写、不允许新增
- condition 必填，不允许为 null
- condition = 该场景的适用对象或适用前提
  例如：客户账号 / 员工账号
- condition 必须来自该 scenario 对应 source 的
  supported_answer / supported_solution，
  尽量使用原文中的表述
- content = 该场景下的处理方式
- scenario 数量必须等于 Gate 给出的 scenario_count
- 不允许发明新的场景条件
""".strip(),
}


SYSTEM_PROMPT = """
你正在把已经通过 Source Q/A Grounding 和
Mixed Knowledge Structure Gate 的 CRM 聊天证据，
整理成 Candidate KB 条目。

这些证据已经确认：

- 属于同一个 question intent
- answer 与 question 对齐
- solution 只能被 answer 支持，
  不能反过来证明 answer
- 已经排除 unusable source

============================================================
最高优先级规则
============================================================

1. 只能使用输入 GROUNDED EVIDENCE 中
   supported_answer / supported_solution
   明确存在的信息。

禁止：

- 补充产品规则
- 根据常识猜测
- 发明操作路径
- 发明菜单名称
- 发明按钮名称
- 发明限制条件
- 发明版本状态
- 发明场景条件
- 发明时间信息

如果材料没有明确支持某个结论，就不要写。

2. 每个 unit 必须给出 source_issue_keys，
   只能使用输入中出现的 issue_key。

   同一条 source 可以同时支撑多个 unit。

   当一条 source 内部列出多个独立原因、
   多个操作分支或多个场景时，
   必须拆成多个 unit，
   这些 unit 引用同一个 issue_key 是允许且正确的。

3. 所有输入的 issue_key 都必须被至少一个 unit 使用，
   不允许静默丢弃 source。

   反过来，unit 数量不受 source 数量限制。

   不允许因为"只有一条 source"
   就把多个独立原因合并成一个 unit。

4. summary_answer 是对整个问题的总述，
   只能概括 units 中已经存在的信息，
   不能引入 unit 中没有的原因、步骤或结论。

   不允许把本应成为 unit 的内容
   写进 notes / limitations 来代替 unit。

5. temporary 不等于 conflict。
   不同原因不等于冲突。
   不同处理路径不等于冲突。

6. 如果 source 是 temporary / partial / unresolved，
   必须在 limitations 中说明该结论
   依赖当前系统状态，可能随版本变化。

7. 不要为了"完整"而扩写。
   宁可短，也不要超出证据。

============================================================
输出
============================================================

只输出 JSON，不要输出任何解释文字。
"""


# ============================================================
# Utils
# ============================================================

def build_client():

    return OpenAI(
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


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

    """
    content 的 2-gram 有多少比例
    出现在被引用 source 的 grounded 文本中。
    """

    content_shingles = shingles(content)

    if not content_shingles:
        return 0.0

    source_shingles = shingles(source_text)

    if not source_shingles:
        return 0.0

    hit = len(content_shingles & source_shingles)

    return hit / len(content_shingles)


def contains_grounded_phrase(phrase, source_text):

    """
    判断 phrase 是否能被 source_text 支撑。

    长短语（>= 6 个规范化字符）：
      原文命中，或字符覆盖率 >= 0.85

    短语（< 6 个规范化字符，
      例如 "客户账号" 这类场景适用对象）：
      字符覆盖率必须 == 1.0，
      并且至少有一个连续 2-gram 在原文中出现，
      避免字符随机拼凑也被判为 grounded
    """

    phrase_n = normalize_text(phrase)

    source_n = normalize_text(source_text)

    if not phrase_n or not source_n:
        return False

    if phrase_n in source_n:
        return True

    chars = set(phrase_n)

    overlap = len(chars & set(source_n)) / len(chars)

    if len(phrase_n) >= 6:

        if overlap >= 0.85:
            return True

        return False

    if overlap < 1.0:
        return False

    phrase_bigrams = shingles(phrase_n)

    source_bigrams = shingles(source_n)

    return bool(phrase_bigrams & source_bigrams)


def prune_records(cluster_ids):

    """
    --redo 时先把指定 cluster 的旧记录移出 JSONL，
    原文件完整备份为 *.bak.jsonl。
    """

    if not cluster_ids:
        return 0

    if not OUTPUT_JSONL.exists():
        return 0

    lines = []

    with OUTPUT_JSONL.open("r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                lines.append(line.rstrip("\n") + "\n")

    keep = []

    removed = []

    for line in lines:

        try:

            obj = json.loads(line)

        except json.JSONDecodeError:

            keep.append(line)

            continue

        cluster_id = clean_text(obj.get("cluster_id"))

        if cluster_id in cluster_ids:
            removed.append(line)
        else:
            keep.append(line)

    if not removed:
        return 0

    backup = OUTPUT_JSONL.with_name(
        OUTPUT_JSONL.stem + ".bak.jsonl"
    )

    backup.write_text("".join(lines), encoding="utf-8")

    OUTPUT_JSONL.write_text("".join(keep), encoding="utf-8")

    print(
        f"[REDO] 已移除 {len(removed)} 条旧记录，"
        f"原文件备份到 {backup.name}"
    )

    return len(removed)


def load_processed():

    processed = set()

    if not OUTPUT_JSONL.exists():
        return processed

    with OUTPUT_JSONL.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(line)

            except json.JSONDecodeError:
                continue

            cluster_id = clean_text(
                obj.get("cluster_id")
            )

            if cluster_id:
                processed.add(cluster_id)

    return processed


# ============================================================
# 输入加载
# ============================================================

def load_ready_clusters():

    structure_xl = pd.ExcelFile(STRUCTURE_FILE)

    clusters_df = structure_xl.parse("all_clusters")

    ready_df = clusters_df[
        clusters_df["generation_ready"] == True
    ].copy()

    evidence_df = structure_xl.parse("evidence_roles")

    evidence_by_cluster = {}

    for cluster_id, group in evidence_df.groupby(
        evidence_df["cluster_id"].astype(str)
    ):

        evidence_by_cluster[cluster_id] = (
            group.to_dict(orient="records")
        )

    audit_df = pd.read_excel(
        MIXED_AUDIT_FILE,
        sheet_name="all_clusters",
    )

    audit_by_cluster = {
        clean_text(row.get("cluster_id")): row
        for row in audit_df.to_dict(orient="records")
    }

    tasks = []

    for row in ready_df.to_dict(orient="records"):

        cluster_id = clean_text(row.get("cluster_id"))

        tasks.append(
            {
                "cluster_id": cluster_id,
                "canonical_question": clean_text(
                    row.get("canonical_question")
                ),
                "knowledge_structure": clean_text(
                    row.get("knowledge_structure")
                ),
                "generation_mode": clean_text(
                    row.get("recommended_generation_mode")
                ),
                "independent_cause_count": int(
                    row.get("independent_cause_count") or 0
                ),
                "scenario_count": int(
                    row.get("scenario_count") or 0
                ),
                "temporal_dependency": clean_text(
                    row.get("temporal_dependency")
                ),
                "evidence_roles": evidence_by_cluster.get(
                    cluster_id,
                    [],
                ),
                "audit_row": audit_by_cluster.get(
                    cluster_id,
                    {},
                ),
            }
        )

    return tasks


def load_usable_sources():

    grounding_df = pd.read_excel(
        GROUNDING_FILE,
        sheet_name="all_sources",
    )

    usable_df = grounding_df[
        grounding_df["usable_for_kb"] == True
    ].copy()

    sources_by_cluster = {}

    for cluster_id, group in usable_df.groupby(
        usable_df["cluster_id"].astype(str)
    ):

        sources_by_cluster[cluster_id] = (
            group.to_dict(orient="records")
        )

    return sources_by_cluster


def grounded_scenario_labels(task):

    """
    只取 Gate 已经验证过 grounding 的 scenario_label。
    """

    labels = []

    for item in task["evidence_roles"]:

        label = clean_text(item.get("scenario_label"))

        evidence = clean_text(
            item.get("scenario_evidence")
        )

        if not label or not evidence:
            continue

        if label not in labels:
            labels.append(label)

    return labels


def source_grounded_text(source_rows, issue_keys):

    parts = []

    for row in source_rows:

        if clean_text(row.get("issue_key")) not in issue_keys:
            continue

        parts.append(
            clean_text(row.get("supported_answer"))
        )

        parts.append(
            clean_text(row.get("supported_solution"))
        )

    return "\n".join(part for part in parts if part)


# ============================================================
# Prompt
# ============================================================

def build_prompt(task, source_rows):

    mode = task["generation_mode"]

    scenario_labels = grounded_scenario_labels(task)

    blocks = []

    for idx, row in enumerate(source_rows, start=1):

        blocks.append(
            f"""
============================================================
GROUNDED EVIDENCE {idx}
============================================================

issue_key:
{clean_text(row.get("issue_key"))}

question:
{clean_text(row.get("question"))}

supported_answer:
{clean_text(row.get("supported_answer"))}

supported_solution:
{clean_text(row.get("supported_solution"))}

resolution:
{clean_text(row.get("resolution"))}

knowledge_value:
{clean_text(row.get("knowledge_value"))}

temporal_status:
{clean_text(row.get("temporal_status"))}

source_confidence:
{clean_text(row.get("source_confidence"))}
""".strip()
        )

    roles = []

    for item in task["evidence_roles"]:

        roles.append(
            f"- {clean_text(item.get('issue_key'))}"
            f" | role={clean_text(item.get('role'))}"
            f" | scenario_label="
            f"{clean_text(item.get('scenario_label')) or 'null'}"
        )

    scenario_block = ""

    if scenario_labels:

        scenario_block = (
            "\n"
            "====================================================\n"
            "Gate 已验证的 scenario_label（必须原样使用）\n"
            "====================================================\n\n"
            + "\n".join(f"- {label}" for label in scenario_labels)
            + "\n"
        )

    audit_row = task["audit_row"]

    return f"""
请为下面这个 Mixed Cluster 生成 Candidate KB 条目。

============================================================
CLUSTER
============================================================

cluster_id:
{task['cluster_id']}

canonical_question:
{task['canonical_question']}

knowledge_structure:
{task['knowledge_structure']}

recommended_generation_mode:
{mode}

independent_cause_count:
{task['independent_cause_count']}

scenario_count:
{task['scenario_count']}

temporal_dependency:
{task['temporal_dependency']}

grounded_source_count:
{len(source_rows)}

Step 11.1 classification:
{clean_text(audit_row.get("classification"))}

Step 11.1 answer_structure_types:
{clean_text(audit_row.get("answer_structure_types"))}

============================================================
Gate evidence roles
============================================================

{chr(10).join(roles)}
{scenario_block}
============================================================
生成要求
============================================================

{MODE_INSTRUCTIONS.get(mode, "")}

============================================================
JSON 结构
============================================================

{{
  "canonical_question": "...",
  "summary_answer": "...",
  "units": [
    {{
      "unit_type": "answer | section | cause | scenario",
      "title": "...",
      "condition": null,
      "content": "...",
      "steps": ["..."],
      "source_issue_keys": ["..."]
    }}
  ],
  "notes": ["..."],
  "limitations": ["..."],
  "crm_module": "...",
  "crm_feature": "...",
  "problem_type": "...",
  "knowledge_confidence": 0.0,
  "source_issue_keys": ["..."]
}}

============================================================
GROUNDED EVIDENCE
============================================================

{chr(10).join(blocks)}
"""


# ============================================================
# Deterministic Structural Checks
# ============================================================

def check_entry(task, entry, source_rows):

    mode = task["generation_mode"]

    allowed_keys = set(
        clean_text(row.get("issue_key"))
        for row in source_rows
    )

    source_by_key = {
        clean_text(row.get("issue_key")): row
        for row in source_rows
    }

    flags = []

    expected_unit_type = EXPECTED_UNIT_TYPE.get(mode)

    # --------------------------------------------------------
    # 1. issue_key 合法性
    # --------------------------------------------------------

    used_keys = set()

    for unit in entry.units:

        for key in unit.source_issue_keys:

            key = clean_text(key)

            used_keys.add(key)

            if key not in allowed_keys:

                flags.append(
                    f"INVENTED_SOURCE_KEY: "
                    f"{unit.title} 引用了不存在的 "
                    f"issue_key {key}"
                )

    for key in entry.source_issue_keys:

        key = clean_text(key)

        if key not in allowed_keys:

            flags.append(
                f"INVENTED_SOURCE_KEY: "
                f"entry 级别引用了不存在的 issue_key {key}"
            )

    # --------------------------------------------------------
    # 2. source 覆盖完整性
    # --------------------------------------------------------

    missing_keys = sorted(allowed_keys - used_keys)

    if missing_keys:

        flags.append(
            "SOURCE_COVERAGE_INCOMPLETE: "
            "以下 grounded source 没有被任何 unit 使用 "
            + ", ".join(missing_keys)
        )

    # --------------------------------------------------------
    # 3. unit_type 与 mode 对齐
    # --------------------------------------------------------

    for unit in entry.units:

        if expected_unit_type and unit.unit_type != expected_unit_type:

            flags.append(
                f"UNIT_TYPE_MISMATCH: "
                f"mode={mode} 要求 unit_type="
                f"{expected_unit_type}，"
                f"实际 {unit.unit_type}（{unit.title}）"
            )

    # --------------------------------------------------------
    # 4. unit 数量与 Gate 对齐
    # --------------------------------------------------------

    if mode == "cause_items":

        if len(entry.units) != task["independent_cause_count"]:

            flags.append(
                f"UNIT_COUNT_MISMATCH: "
                f"independent_cause_count="
                f"{task['independent_cause_count']}，"
                f"实际 cause 数量 {len(entry.units)}"
            )

    if mode == "scenario_sections":

        titles = [
            normalize_text(unit.title)
            for unit in entry.units
        ]

        if len(set(titles)) != task["scenario_count"]:

            flags.append(
                f"UNIT_COUNT_MISMATCH: "
                f"scenario_count={task['scenario_count']}，"
                f"实际不同 scenario 数量 {len(set(titles))}"
            )

    if mode == "single_answer" and len(entry.units) != 1:

        flags.append(
            f"UNIT_COUNT_MISMATCH: "
            f"single_answer 只允许 1 个 unit，"
            f"实际 {len(entry.units)}"
        )

    if mode == "howto_sections" and len(entry.units) < 2:

        flags.append(
            f"HOWTO_UNIT_COUNT_LOW: "
            f"howto_sections 只有 {len(entry.units)} 个 section，"
            f"grounded source 有 {len(source_rows)} 条"
        )

    # --------------------------------------------------------
    # 5. Scenario label / condition grounding
    # --------------------------------------------------------

    if mode == "scenario_sections":

        gate_labels = set(
            normalize_text(label)
            for label in grounded_scenario_labels(task)
        )

        for unit in entry.units:

            if gate_labels and normalize_text(unit.title) not in gate_labels:

                flags.append(
                    f"SCENARIO_LABEL_NOT_FROM_GATE: "
                    f"{unit.title} 不在 Gate 已验证的 "
                    f"scenario_label 中"
                )

            condition = clean_text(unit.condition)

            if not condition:

                flags.append(
                    f"SCENARIO_CONDITION_MISSING: "
                    f"{unit.title} 缺少 condition"
                )

                continue

            cited_text = source_grounded_text(
                source_rows,
                set(
                    clean_text(key)
                    for key in unit.source_issue_keys
                ),
            )

            if not contains_grounded_phrase(condition, cited_text):

                flags.append(
                    f"SCENARIO_CONDITION_UNGROUNDED: "
                    f"{unit.title} 的 condition "
                    f"未在被引用 source 中命中"
                )

    # --------------------------------------------------------
    # 6. 词面 grounding 指标（soft）
    # --------------------------------------------------------

    grounding_scores = []

    for unit in entry.units:

        cited_keys = set(
            clean_text(key)
            for key in unit.source_issue_keys
        )

        cited_text = source_grounded_text(
            source_rows,
            cited_keys,
        )

        text = unit.content

        if unit.steps:
            text = text + "\n" + "\n".join(unit.steps)

        ratio = lexical_grounded_ratio(text, cited_text)

        grounding_scores.append(ratio)

        if ratio < LEXICAL_GROUNDING_THRESHOLD:

            flags.append(
                f"WEAK_LEXICAL_GROUNDING: "
                f"{unit.title} 词面 grounding 比例 "
                f"{ratio:.2f} < "
                f"{LEXICAL_GROUNDING_THRESHOLD}"
            )

    # --------------------------------------------------------
    # 7. temporal caution（soft）
    # --------------------------------------------------------

    risky_sources = [
        row
        for row in source_rows
        if clean_text(row.get("temporal_status")) == "temporary"
        or clean_text(row.get("resolution")) in
        {"partial", "unresolved"}
    ]

    if risky_sources:

        caution_text = normalize_text(
            " ".join(entry.notes + entry.limitations)
        )

        caution_words = [
            "临时",
            "当前",
            "版本",
            "优化",
            "可能",
            "待",
            "未解决",
            "部分",
        ]

        if not any(
            normalize_text(word) in caution_text
            for word in caution_words
        ):

            flags.append(
                "TEMPORAL_CAUTION_MISSING: "
                f"{len(risky_sources)} 条 source 为 "
                "temporary / partial / unresolved，"
                "但 notes / limitations 未做时效提示"
            )

    # --------------------------------------------------------
    # 8. 单一 source 支撑（soft）
    # --------------------------------------------------------

    if len(source_rows) == 1:

        flags.append(
            "SINGLE_SOURCE_ENTRY: "
            "整条 Candidate KB 只有 1 条 grounded source 支撑，"
            f"但需要产出 {len(entry.units)} 个 unit，"
            "publishability 阶段需要额外复核"
        )

    hard_flags = [
        flag for flag in flags if flag.split(":")[0] not in SOFT_FLAGS
    ]

    soft_flags = [
        flag for flag in flags if flag.split(":")[0] in SOFT_FLAGS
    ]

    status = (
        "candidate_ok"
        if not hard_flags
        else "needs_review"
    )

    return {
        "status": status,
        "hard_flags": hard_flags,
        "soft_flags": soft_flags,
        "structural_flags": flags,
        "unit_count": len(entry.units),
        "cited_source_count": len(
            set(
                clean_text(key)
                for key in
                [
                    key
                    for unit in entry.units
                    for key in unit.source_issue_keys
                ]
            )
        ),
        "usable_source_count": len(source_rows),
        "min_lexical_grounding": (
            min(grounding_scores) if grounding_scores else 0.0
        ),
        "avg_lexical_grounding": (
            sum(grounding_scores) / len(grounding_scores)
            if grounding_scores
            else 0.0
        ),
        "risky_source_count": len(risky_sources),
        "source_by_key": source_by_key,
    }


# ============================================================
# Generate
# ============================================================

def generate_cluster(task, source_rows):

    client = build_client()

    cluster_id = task["cluster_id"]

    prompt = build_prompt(task, source_rows)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            started = time.time()

            response = (
                client
                .chat
                .completions
                .create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    extra_body={"enable_thinking": False},
                )
            )

            elapsed = time.time() - started

            data = json.loads(
                response.choices[0].message.content
            )

            entry = MixedCandidateEntry.model_validate(data)

            check = check_entry(task, entry, source_rows)

            return {
                "cluster_id": cluster_id,
                "canonical_question": entry.canonical_question,
                "knowledge_structure": task["knowledge_structure"],
                "generation_mode": task["generation_mode"],
                "summary_answer": entry.summary_answer,
                "units": [
                    unit.model_dump()
                    for unit in entry.units
                ],
                "unit_count": check["unit_count"],
                "notes": entry.notes,
                "limitations": entry.limitations,
                "crm_module": entry.crm_module,
                "crm_feature": entry.crm_feature,
                "problem_type": entry.problem_type,
                "knowledge_confidence": entry.knowledge_confidence,
                "source_issue_keys": entry.source_issue_keys,
                "usable_source_count": check["usable_source_count"],
                "cited_source_count": check["cited_source_count"],
                "risky_source_count": check["risky_source_count"],
                "min_lexical_grounding": round(
                    check["min_lexical_grounding"], 4
                ),
                "avg_lexical_grounding": round(
                    check["avg_lexical_grounding"], 4
                ),
                "candidate_status": check["status"],
                "hard_flag_count": len(check["hard_flags"]),
                "hard_flags": check["hard_flags"],
                "soft_flags": check["soft_flags"],
                "structural_flags": check["structural_flags"],
                "request_seconds": elapsed,
            }

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                time.sleep(2 ** attempt)

    raise last_error


# ============================================================
# Export
# ============================================================

def join_list(value):

    if isinstance(value, list):
        return " | ".join(clean_text(item) for item in value)

    return value


def export_excel():

    records = []

    if OUTPUT_JSONL.exists():

        with OUTPUT_JSONL.open("r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                records.append(json.loads(line))

    if not records:

        print("[WARN] 没有可导出的记录")

        return

    df = pd.DataFrame(records)

    entry_cols = [
        "cluster_id",
        "canonical_question",
        "knowledge_structure",
        "generation_mode",
        "summary_answer",
        "unit_count",
        "usable_source_count",
        "cited_source_count",
        "risky_source_count",
        "min_lexical_grounding",
        "avg_lexical_grounding",
        "candidate_status",
        "hard_flag_count",
        "knowledge_confidence",
        "crm_module",
        "crm_feature",
        "problem_type",
        "notes",
        "limitations",
        "source_issue_keys",
        "hard_flags",
        "soft_flags",
        "request_seconds",
    ]

    kb_entries = df[
        [col for col in entry_cols if col in df.columns]
    ].copy()

    for col in [
        "notes",
        "limitations",
        "source_issue_keys",
        "hard_flags",
        "soft_flags",
    ]:

        if col in kb_entries.columns:

            kb_entries[col] = kb_entries[col].apply(join_list)

    unit_rows = []

    flag_rows = []

    coverage_rows = []

    for record in records:

        for index, unit in enumerate(
            record.get("units") or [], start=1
        ):

            unit_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    "canonical_question": record[
                        "canonical_question"
                    ],
                    "generation_mode": record["generation_mode"],
                    "unit_index": index,
                    "unit_type": unit.get("unit_type"),
                    "title": unit.get("title"),
                    "condition": unit.get("condition"),
                    "content": unit.get("content"),
                    "steps": join_list(unit.get("steps")),
                    "source_issue_keys": join_list(
                        unit.get("source_issue_keys")
                    ),
                }
            )

        for flag in record.get("structural_flags") or []:

            flag_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    "flag_type": flag.split(":")[0],
                    "severity": (
                        "soft"
                        if flag.split(":")[0] in SOFT_FLAGS
                        else "hard"
                    ),
                    "flag": flag,
                }
            )

        cited = set()

        for unit in record.get("units") or []:

            for key in unit.get("source_issue_keys") or []:

                cited.add(clean_text(key))

        for key in record.get("source_issue_keys") or []:

            cited.add(clean_text(key))

        coverage_rows.append(
            {
                "cluster_id": record["cluster_id"],
                "usable_source_count": record[
                    "usable_source_count"
                ],
                "cited_source_count": record[
                    "cited_source_count"
                ],
                "coverage_complete": (
                    record["cited_source_count"]
                    == record["usable_source_count"]
                ),
                "cited_issue_keys": " | ".join(sorted(cited)),
            }
        )

    units_df = pd.DataFrame(unit_rows)

    scenario_units_df = units_df[
        units_df["unit_type"] == "scenario"
    ] if len(units_df) else units_df

    summary = pd.DataFrame(
        [
            {"metric": "entry_count", "value": len(df)},
            {
                "metric": "candidate_ok_count",
                "value": int(
                    (df["candidate_status"] == "candidate_ok").sum()
                ),
            },
            {
                "metric": "needs_review_count",
                "value": int(
                    (df["candidate_status"] == "needs_review").sum()
                ),
            },
            {
                "metric": "unit_count",
                "value": int(df["unit_count"].sum()),
            },
            {
                "metric": "usable_source_count",
                "value": int(df["usable_source_count"].sum()),
            },
            {
                "metric": "cited_source_count",
                "value": int(df["cited_source_count"].sum()),
            },
            {
                "metric": "avg_lexical_grounding",
                "value": round(
                    float(df["avg_lexical_grounding"].mean()), 4
                ),
            },
            {
                "metric": "hard_flag_total",
                "value": int(df["hard_flag_count"].sum()),
            },
            {"metric": "model", "value": MODEL},
        ]
    )

    mode_stats = (
        df["generation_mode"]
        .value_counts()
        .rename_axis("generation_mode")
        .reset_index(name="count")
    )

    flag_stats = (
        pd.DataFrame(flag_rows)["flag_type"]
        .value_counts()
        .rename_axis("flag_type")
        .reset_index(name="count")
        if flag_rows
        else pd.DataFrame(columns=["flag_type", "count"])
    )

    needs_review_df = kb_entries[
        kb_entries["candidate_status"] == "needs_review"
    ]

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        summary.to_excel(
            writer, index=False, sheet_name="summary"
        )

        mode_stats.to_excel(
            writer, index=False, sheet_name="mode_stats"
        )

        flag_stats.to_excel(
            writer, index=False, sheet_name="flag_stats"
        )

        kb_entries.to_excel(
            writer, index=False, sheet_name="kb_entries"
        )

        units_df.to_excel(
            writer, index=False, sheet_name="units"
        )

        scenario_units_df.to_excel(
            writer, index=False, sheet_name="scenario_units"
        )

        pd.DataFrame(coverage_rows).to_excel(
            writer, index=False, sheet_name="source_coverage"
        )

        pd.DataFrame(flag_rows).to_excel(
            writer, index=False, sheet_name="structural_flags"
        )

        needs_review_df.to_excel(
            writer, index=False, sheet_name="needs_review"
        )


# ============================================================
# Main
# ============================================================

def recheck_existing():

    """
    --recheck：

    不调用 LLM，
    只把已存在的 Candidate KB 记录
    用当前 deterministic 规则重新判定一次，
    然后重写 JSONL / Excel。
    """

    if not OUTPUT_JSONL.exists():
        raise FileNotFoundError(
            f"缺少 {OUTPUT_JSONL}，无法 recheck"
        )

    tasks = {
        task["cluster_id"]: task
        for task in load_ready_clusters()
    }

    sources_by_cluster = load_usable_sources()

    records = []

    with OUTPUT_JSONL.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    backup = OUTPUT_JSONL.with_name(
        OUTPUT_JSONL.stem + ".recheck.bak.jsonl"
    )

    backup.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )

    changed = []

    for record in records:

        cluster_id = clean_text(record.get("cluster_id"))

        task = tasks.get(cluster_id)

        source_rows = sources_by_cluster.get(cluster_id, [])

        if not task or not source_rows:

            record["candidate_status"] = "needs_review"

            record["hard_flags"] = (
                record.get("hard_flags") or []
            ) + [
                "RECHECK_NO_GATE_OR_SOURCE: "
                "找不到对应的 Gate 记录或 usable source"
            ]

            record["hard_flag_count"] = len(record["hard_flags"])

            record["recheck_applied"] = True

            changed.append(cluster_id)

            continue

        entry = MixedCandidateEntry.model_validate(
            {
                "canonical_question": record[
                    "canonical_question"
                ],
                "summary_answer": record["summary_answer"],
                "units": record.get("units") or [],
                "notes": record.get("notes") or [],
                "limitations": record.get("limitations") or [],
                "crm_module": record["crm_module"],
                "crm_feature": record["crm_feature"],
                "problem_type": record["problem_type"],
                "knowledge_confidence": record[
                    "knowledge_confidence"
                ],
                "source_issue_keys": record[
                    "source_issue_keys"
                ],
            }
        )

        check = check_entry(task, entry, source_rows)

        before = record.get("candidate_status")

        record["candidate_status"] = check["status"]

        record["hard_flags"] = check["hard_flags"]

        record["soft_flags"] = check["soft_flags"]

        record["structural_flags"] = check["structural_flags"]

        record["hard_flag_count"] = len(check["hard_flags"])

        record["unit_count"] = check["unit_count"]

        record["usable_source_count"] = check[
            "usable_source_count"
        ]

        record["cited_source_count"] = check[
            "cited_source_count"
        ]

        record["risky_source_count"] = check[
            "risky_source_count"
        ]

        record["min_lexical_grounding"] = round(
            check["min_lexical_grounding"], 4
        )

        record["avg_lexical_grounding"] = round(
            check["avg_lexical_grounding"], 4
        )

        record["recheck_applied"] = True

        if before != check["status"]:

            changed.append(cluster_id)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as fout:

        for record in records:

            fout.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    export_excel()

    print()
    print("-" * 70)
    print("Recheck 结果")
    print("-" * 70)

    for record in records:

        print(
            f"{record['cluster_id']:15} "
            f"| {record['generation_mode']:18} "
            f"| units={record['unit_count']} "
            f"| {record['candidate_status']:13} "
            f"| hard={record['hard_flag_count']}"
        )

    print()

    if changed:

        print(f"状态发生变化: {sorted(changed)}")

    else:

        print("状态没有变化")

    print()
    print(f"备份: {backup}")

    print(f"JSONL: {OUTPUT_JSONL}")

    print(f"Excel: {OUTPUT_XLSX}")


def main():

    print("=" * 70)
    print("Step 11.4 - Mixed Cluster Candidate KB Generation")
    print("=" * 70)

    print(f"Model: {MODEL}")

    print(f"Structure gate: {STRUCTURE_FILE.name}")

    tasks = load_ready_clusters()

    sources_by_cluster = load_usable_sources()

    print(f"generation_ready clusters: {len(tasks)}")

    usable_total = sum(
        len(sources_by_cluster.get(task["cluster_id"], []))
        for task in tasks
    )

    print(f"usable grounded sources: {usable_total}")

    if RECHECK:

        print()
        print("RECHECK 模式 - 不调用 LLM")

        recheck_existing()

        return

    if DRY_RUN:

        print()
        print("-" * 70)
        print("DRY RUN - 不调用 LLM，不写任何输出")
        print("-" * 70)

        for task in tasks:

            cluster_id = task["cluster_id"]

            source_rows = sources_by_cluster.get(cluster_id, [])

            labels = grounded_scenario_labels(task)

            print(
                f"{cluster_id} "
                f"| {task['knowledge_structure']:18} "
                f"| {task['generation_mode']:18} "
                f"| sources={len(source_rows)} "
                f"| causes={task['independent_cause_count']} "
                f"| scenarios={task['scenario_count']}"
            )

            print(f"    Q: {task['canonical_question']}")

            if labels:

                print(
                    "    gate scenario_labels: "
                    + " / ".join(labels)
                )

            for row in source_rows:

                print(
                    f"      - {clean_text(row.get('issue_key'))} "
                    f"| temporal="
                    f"{clean_text(row.get('temporal_status'))} "
                    f"| resolution="
                    f"{clean_text(row.get('resolution'))}"
                )

        print()
        print("DRY RUN 完成")

        return

    if REDO_IDS:

        unknown = REDO_IDS - set(
            task["cluster_id"] for task in tasks
        )

        if unknown:

            print(
                f"[WARN] --redo 中的 cluster 不在 "
                f"generation_ready 列表: {sorted(unknown)}"
            )

        prune_records(
            REDO_IDS & set(
                task["cluster_id"] for task in tasks
            )
        )

    processed = load_processed()

    pending = []

    for task in tasks:

        cluster_id = task["cluster_id"]

        if cluster_id in processed:

            print(f"[SKIP] {cluster_id}: 已存在")

            continue

        source_rows = sources_by_cluster.get(cluster_id, [])

        if not source_rows:

            print(f"[SKIP] {cluster_id}: NO_VALID_SOURCE")

            continue

        pending.append((task, source_rows))

    print(f"本次待生成: {len(pending)}")

    total = len(pending)

    completed = 0

    started_at = time.time()

    lock = threading.Lock()

    if pending:

        with OUTPUT_JSONL.open("a", encoding="utf-8") as fout:

            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                future_map = {
                    executor.submit(
                        generate_cluster,
                        task,
                        source_rows,
                    ): task["cluster_id"]
                    for task, source_rows in pending
                }

                for future in as_completed(future_map):

                    cluster_id = future_map[future]

                    try:

                        record = future.result()

                    except Exception as exc:

                        print(f"[FAILED] {cluster_id}: {exc}")

                        continue

                    with lock:

                        fout.write(
                            json.dumps(record, ensure_ascii=False)
                            + "\n"
                        )

                        fout.flush()

                        completed += 1

                    elapsed = time.time() - started_at

                    avg = elapsed / completed

                    eta = avg * (total - completed)

                    print(
                        f"[{completed}/{total}] "
                        f"| {cluster_id} "
                        f"| {record['generation_mode']} "
                        f"| units={record['unit_count']} "
                        f"| {record['candidate_status']} "
                        f"| hard_flags="
                        f"{record['hard_flag_count']} "
                        f"| ETA {eta / 60:.1f} min"
                    )

    export_excel()

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)

    print(f"JSONL: {OUTPUT_JSONL}")

    print(f"Excel: {OUTPUT_XLSX}")

    print()
    print("注意：这只是 Candidate KB。")

    print("后续仍需 grounding validation / temporal 处理 /")

    print("final publishability gate 才能进入正式 KB。")


if __name__ == "__main__":
    main()
