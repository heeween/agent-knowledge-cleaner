"""
Step 11.5 - Mixed Candidate KB Grounding Validation

输入（全部为已冻结产物）：

output/kb_entries_mixed_candidate.jsonl
  Step 11.4 冻结的 Candidate KB（10 entries / 25 units）

output/mixed_source_qa_alignments_v2.xlsx
  all_sources，只使用 usable_for_kb == true
  只把 supported_answer / supported_solution 交给模型

output/mixed_structure_classifications_v3_1.xlsx
  all_clusters / evidence_roles
  提供 Gate 的结构、mode、数量与 scenario_label

输出：

output/kb_entry_grounding_validations_mixed.jsonl
output/kb_entry_grounding_validations_mixed.xlsx

本步骤只做验证，不改写 Candidate KB，
不触碰 74 条正式 KB。

验证维度：

1. unit 级语义 grounding（LLM）
2. summary_answer 级 grounding（LLM）
   —— 用于抓 unit 内容干净、
      但 summary 引入 管理员 / 版本 / 规则 这类
      无来源表述的情况
3. invented element 检测（LLM + severity）
4. scenario condition grounding（LLM）
5. temporal caution（deterministic）
6. cross-entry near duplicate（deterministic）
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

CANDIDATE_FILE = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate.jsonl"
)

GROUNDING_FILE = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

STRUCTURE_FILE = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.xlsx"
)

EMBEDDING_FILE = (
    OUTPUT_DIR
    / "question_embeddings.jsonl"
)

PAIR_FILE = (
    OUTPUT_DIR
    / "pair_classifications.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entry_grounding_validations_mixed.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entry_grounding_validations_mixed.xlsx"
)

MODEL = "glm-5.3-flash"

MAX_WORKERS = 5

MAX_RETRIES = 4

REQUEST_TIMEOUT = 180

# entry centroid cosine 阈值（基于 frozen text-embedding-v4）
NEAR_DUPLICATE_COSINE = 0.88

RELATED_COSINE = 0.80

DRY_RUN = "--dry-run" in sys.argv

PARSE_CHECK = "--parse-check" in sys.argv

ONLY_IDS = set()

for _index, _arg in enumerate(sys.argv):

    if _arg == "--only" and _index + 1 < len(sys.argv):

        ONLY_IDS = {
            item.strip()
            for item in sys.argv[_index + 1].split(",")
            if item.strip()
        }

for _index, _arg in enumerate(sys.argv):

    if _arg == "--candidate" and _index + 1 < len(sys.argv):

        CANDIDATE_FILE = (
            Path(sys.argv[_index + 1])
            .expanduser()
            .resolve()
        )

        # Step 11.6 起支持版本化 candidate:
        # _v2 / _v3 ... 输出文件带同样后缀，
        # 避免覆盖旧版本 validation 输出。
        _version = re.search(
            r"_v(\d+)\.jsonl$",
            CANDIDATE_FILE.name,
        )

        if _version:

            _suffix = f"_v{_version.group(1)}"

            OUTPUT_JSONL = (
                OUTPUT_DIR
                / (
                    "kb_entry_grounding_validations_mixed"
                    f"{_suffix}.jsonl"
                )
            )

            OUTPUT_XLSX = (
                OUTPUT_DIR
                / (
                    "kb_entry_grounding_validations_mixed"
                    f"{_suffix}.xlsx"
                )
            )


# ============================================================
# Schema
# ============================================================

class InventedElement(BaseModel):

    element_type: Literal[
        "actor",
        "menu_path",
        "button_name",
        "product_rule",
        "condition",
        "number",
        "time",
        "version_status",
        "procedure_step",
        "other",
    ]

    text: str

    location: Literal[
        "summary_answer",
        "unit",
        "notes",
        "limitations",
    ]

    unit_title: str | None = None

    severity: Literal["material", "minor"]

    reason: str = ""


class UnitGrounding(BaseModel):

    unit_index: int

    unit_title: str = ""

    grounding: Literal[
        "grounded",
        "partially_grounded",
        "unsupported",
    ]

    condition_grounded: bool | None = None

    invented_elements: list[InventedElement] = Field(
        default_factory=list
    )

    reason: str = ""

    confidence: float = 0.0


class EntryGroundingValidation(BaseModel):

    summary_grounding: Literal[
        "grounded",
        "partially_grounded",
        "unsupported",
    ]

    summary_reason: str = ""

    units: list[UnitGrounding]

    notes_grounded: bool

    temporal_caution_needed: bool

    temporal_caution_present: bool

    conflict_wrongly_claimed: bool

    invented_elements: list[InventedElement] = Field(
        default_factory=list
    )

    reason: str = ""

    confidence: float = 0.0


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在审核一批 CRM Candidate KB 条目，
判断它们是否严格被 grounded source 支撑。

这些 source 已经通过 Source Q/A Grounding Gate，
输入中只提供 supported_answer 与 supported_solution。

============================================================
审核原则
============================================================

1. 只有能在被引用 source 的
   supported_answer / supported_solution
   中找到依据的内容，才算 grounded。

2. 允许语言规范化、允许归纳、允许合并同义表述。

3. 不允许出现 source 中不存在的：

- 执行者 / 角色（例如"管理员""运营""财务"）
- 菜单路径 / 页面名称 / 按钮名称
- 产品规则 / 限制条件 / 适用范围
- 数字 / 时间 / 期限 / 版本状态
- 额外操作步骤

只要出现上述任意一种，
必须记录为 invented_elements。

4. severity 判断：

material：

- 会让读者做出错误操作
- 引入了 source 没有的执行者、权限、规则、
  数字、版本状态
- 把"可能"写成"一定"

minor：

- 只是措辞更具体，但不改变操作含义
- 不影响读者执行

5. summary_answer 必须单独审核。

即使每个 unit 都干净，
summary_answer 也可能引入 unit 中没有的信息，
这种情况必须记录，
location = summary_answer。

6. 不同原因不是冲突。
   不同处理路径不是冲突。
   temporary 不是冲突。

如果条目把多个独立原因描述成互斥冲突，
conflict_wrongly_claimed = true。

7. temporal caution：

如果被引用 source 中存在
temporary / partial / unresolved，
则 temporal_caution_needed = true。

只有当 notes / limitations
明确提示"依赖当前系统状态、可能随版本变化"
之类内容时，
temporal_caution_present 才为 true。

8. 不要因为条目写得短就判 unsupported。
   短但完全来自 source，就是 grounded。

============================================================
输出 JSON 结构（必须严格遵守）
============================================================

{
  "summary_grounding": "grounded | partially_grounded | unsupported",
  "summary_reason": "为什么这样判断",
  "units": [
    {
      "unit_index": 1,
      "unit_title": "与输入 UNIT 的 title 一致",
      "grounding": "grounded | partially_grounded | unsupported",
      "condition_grounded": null,
      "invented_elements": [],
      "reason": "为什么这样判断",
      "confidence": 0.0
    }
  ],
  "notes_grounded": true,
  "temporal_caution_needed": false,
  "temporal_caution_present": false,
  "conflict_wrongly_claimed": false,
  "invented_elements": [],
  "reason": "整条 entry 的总结判断",
  "confidence": 0.0
}

invented_elements 中每一项的结构：

{
  "element_type": "actor | menu_path | button_name | product_rule | condition | number | time | version_status | procedure_step | other",
  "text": "无来源的原文片段",
  "location": "summary_answer | unit | notes | limitations",
  "unit_title": "当 location = unit 时填写，否则 null",
  "severity": "material | minor",
  "reason": "为什么判定为无来源"
}

字段规则：

- units 必须覆盖输入中的每一个 unit，
  unit_index 从 1 开始并与输入顺序一致
- condition_grounded 只在 unit_type = scenario 时填
  true / false，其余情况填 null
- unit 内部的 invented_elements 放在该 unit 里，
  entry 级（summary_answer / notes / limitations）的
  放在顶层 invented_elements
- 没有发现问题时，invented_elements 填空数组
- 只输出 JSON，不要输出任何解释文字
"""


def build_prompt(record, source_rows, gate_row):

    source_blocks = []

    for idx, row in enumerate(source_rows, start=1):

        source_blocks.append(
            f"""
------------------------------------------------------------
SOURCE {idx}
------------------------------------------------------------

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

temporal_status:
{clean_text(row.get("temporal_status"))}

knowledge_value:
{clean_text(row.get("knowledge_value"))}
""".strip()
        )

    unit_blocks = []

    for idx, unit in enumerate(record.get("units") or [], start=1):

        unit_blocks.append(
            f"""
------------------------------------------------------------
UNIT {idx}
------------------------------------------------------------

unit_type:
{clean_text(unit.get("unit_type"))}

title:
{clean_text(unit.get("title"))}

condition:
{clean_text(unit.get("condition")) or "null"}

content:
{clean_text(unit.get("content"))}

steps:
{json.dumps(unit.get("steps") or [], ensure_ascii=False)}

source_issue_keys:
{json.dumps(unit.get("source_issue_keys") or [], ensure_ascii=False)}
""".strip()
        )

    scenario_labels = []

    for item in gate_row.get("evidence_roles") or []:

        label = clean_text(item.get("scenario_label"))

        if label and label not in scenario_labels:
            scenario_labels.append(label)

    return f"""
请审核下面这条 Candidate KB 条目。

============================================================
ENTRY
============================================================

cluster_id:
{clean_text(record.get("cluster_id"))}

canonical_question:
{clean_text(record.get("canonical_question"))}

knowledge_structure:
{clean_text(record.get("knowledge_structure"))}

generation_mode:
{clean_text(record.get("generation_mode"))}

gate_independent_cause_count:
{gate_row.get("independent_cause_count")}

gate_scenario_count:
{gate_row.get("scenario_count")}

gate_scenario_labels:
{json.dumps(scenario_labels, ensure_ascii=False)}

summary_answer:
{clean_text(record.get("summary_answer"))}

notes:
{json.dumps(record.get("notes") or [], ensure_ascii=False)}

limitations:
{json.dumps(record.get("limitations") or [], ensure_ascii=False)}

============================================================
UNITS
============================================================

{chr(10).join(unit_blocks)}

============================================================
GROUNDED SOURCES
============================================================

{chr(10).join(source_blocks)}
"""


# ============================================================
# Utils
# ============================================================

def build_client():

    return OpenAI(
        base_url='https://open.bigmodel.cn/api/paas/v4',
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


CAUTION_WORDS = [
    "临时",
    "当前",
    "版本",
    "优化",
    "可能",
    "待",
    "未解决",
    "部分",
]


def has_temporal_caution(record):

    text = normalize_text(
        " ".join(
            (record.get("notes") or [])
            + (record.get("limitations") or [])
        )
    )

    return any(
        normalize_text(word) in text
        for word in CAUTION_WORDS
    )


def risky_sources(source_rows):

    return [
        row
        for row in source_rows
        if clean_text(row.get("temporal_status")) == "temporary"
        or clean_text(row.get("resolution")) in
        {"partial", "unresolved"}
    ]


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

            cluster_id = clean_text(obj.get("cluster_id"))

            if cluster_id:
                processed.add(cluster_id)

    return processed


# ============================================================
# 输入加载
# ============================================================

def load_candidates():

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"缺少 Step 11.4 输出: {CANDIDATE_FILE}"
        )

    records = []

    with CANDIDATE_FILE.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    if not records:
        raise RuntimeError("Candidate KB JSONL 为空")

    return records


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


def load_gate_rows():

    structure_xl = pd.ExcelFile(STRUCTURE_FILE)

    clusters_df = structure_xl.parse("all_clusters")

    evidence_df = structure_xl.parse("evidence_roles")

    evidence_by_cluster = {}

    for cluster_id, group in evidence_df.groupby(
        evidence_df["cluster_id"].astype(str)
    ):

        evidence_by_cluster[cluster_id] = (
            group.to_dict(orient="records")
        )

    gate_rows = {}

    for row in clusters_df.to_dict(orient="records"):

        cluster_id = clean_text(row.get("cluster_id"))

        row["evidence_roles"] = evidence_by_cluster.get(
            cluster_id,
            [],
        )

        gate_rows[cluster_id] = row

    return gate_rows


# ============================================================
# Deterministic Source Override
# ============================================================

def combined_source_text(source_rows, issue_keys):

    wanted = {
        clean_text(item) for item in issue_keys
    }

    parts = []

    for row in source_rows:

        if clean_text(row.get("issue_key")) not in wanted:
            continue

        parts.append(
            clean_text(row.get("supported_answer"))
        )

        parts.append(
            clean_text(row.get("supported_solution"))
        )

    return "\n".join(part for part in parts if part)


def exact_actor_in_source(item, source_text):

    if item.element_type != "actor":
        return False

    if item.location != "unit":
        return False

    phrase = clean_text(item.text)

    return bool(
        phrase
        and phrase in source_text
    )


def apply_exact_actor_overrides(
    validation,
    record,
    source_rows,
):

    """
    Step 11.5 的 LLM 可能把 source 中明确出现的角色
    误判为 invented actor。

    本规则只覆盖一种确定情况：

    - element_type = actor
    - location = unit
    - item.text 原样出现在该 unit 引用的
      supported_answer / supported_solution

    summary 级 actor 不覆盖，仍交给 LLM verdict。
    """

    overrides = []

    units = list(record.get("units") or [])

    all_source_text = "\n".join(
        clean_text(row.get("supported_answer"))
        + "\n"
        + clean_text(row.get("supported_solution"))
        for row in source_rows
    )

    for unit in validation.units:

        source_unit = (
            units[unit.unit_index - 1]
            if 1 <= unit.unit_index <= len(units)
            else {}
        )

        source_text = combined_source_text(
            source_rows,
            source_unit.get("source_issue_keys") or [],
        )

        kept = []

        for item in unit.invented_elements:

            if exact_actor_in_source(item, source_text):

                overrides.append(
                    {
                        "scope": "unit",
                        "unit_index": unit.unit_index,
                        "unit_title": unit.unit_title,
                        "element_type": item.element_type,
                        "text": item.text,
                        "reason": (
                            "exact actor text found in cited source"
                        ),
                    }
                )

                continue

            kept.append(item)

        unit.invented_elements = kept

        if (
            overrides
            and unit.grounding == "partially_grounded"
        ):

            unit.grounding = "grounded"

            unit.reason = (
                "Deterministic override: "
                "the exact actor phrase occurs in the cited "
                "supported_answer/supported_solution. "
                + clean_text(unit.reason)
            )

    kept_entry_items = []

    for item in validation.invented_elements:

        if (
            item.element_type == "actor"
            and item.location == "unit"
            and clean_text(item.text) in all_source_text
        ):

            overrides.append(
                {
                    "scope": "entry",
                    "unit_index": None,
                    "unit_title": item.unit_title,
                    "element_type": item.element_type,
                    "text": item.text,
                    "reason": (
                        "duplicate unit actor flag suppressed; "
                        "exact text found in source"
                    ),
                }
            )

            continue

        kept_entry_items.append(item)

    validation.invented_elements = kept_entry_items

    return validation, overrides


# ============================================================
# Deterministic Checks
# ============================================================

def deterministic_flags(record, source_rows, gate_row):

    flags = []

    allowed_keys = set(
        clean_text(row.get("issue_key"))
        for row in source_rows
    )

    used_keys = set()

    for unit in record.get("units") or []:

        for key in unit.get("source_issue_keys") or []:

            key = clean_text(key)

            used_keys.add(key)

            if key not in allowed_keys:

                flags.append(
                    {
                        "flag": "INVENTED_SOURCE_KEY",
                        "severity": "hard",
                        "detail": f"{key} 不在 usable source 中",
                    }
                )

    missing = sorted(allowed_keys - used_keys)

    if missing:

        flags.append(
            {
                "flag": "SOURCE_COVERAGE_INCOMPLETE",
                "severity": "hard",
                "detail": "未被引用的 source: " + ", ".join(missing),
            }
        )

    mode = clean_text(record.get("generation_mode"))

    unit_count = len(record.get("units") or [])

    if mode == "cause_items":

        expected = int(gate_row.get("independent_cause_count") or 0)

        # Step 12.3 起支持 temporal-unblock entry:
        # v3.1 gate 对 troubleshooting 结构不按
        # cause 计数 (independent_cause_count=0),
        # Step 12.2 CURRENT_CONFIRMED 后按
        # source 数生成 unit, 豁免严格计数,
        # 以 info flag 留痕。
        temporal_unblock = (
            (record.get("temporal_validity") or {}).get(
                "final_decision"
            )
            == "CURRENT_CONFIRMED"
        )

        if unit_count != expected and not temporal_unblock:

            flags.append(
                {
                    "flag": "UNIT_COUNT_MISMATCH",
                    "severity": "hard",
                    "detail": (
                        f"gate independent_cause_count={expected}，"
                        f"实际 {unit_count}"
                    ),
                },
            )

        elif temporal_unblock and expected != unit_count:

            flags.append(
                {
                    "flag": "TEMPORAL_UNBLOCK_COUNT_OVERRIDE",
                    "severity": "info",
                    "detail": (
                        f"Step 12.2 CURRENT_CONFIRMED: "
                        f"gate independent_cause_count={expected} "
                        f"(troubleshooting 未计数), "
                        f"units={unit_count}"
                    ),
                },
            )

    if mode == "scenario_sections":

        expected = int(gate_row.get("scenario_count") or 0)

        titles = set(
            normalize_text(unit.get("title"))
            for unit in record.get("units") or []
        )

        if len(titles) != expected:

            flags.append(
                {
                    "flag": "UNIT_COUNT_MISMATCH",
                    "severity": "hard",
                    "detail": (
                        f"gate scenario_count={expected}，"
                        f"实际 {len(titles)}"
                    ),
                }
            )

        gate_labels = set(
            normalize_text(
                clean_text(item.get("scenario_label"))
            )
            for item in gate_row.get("evidence_roles") or []
            if clean_text(item.get("scenario_label"))
        )

        for unit in record.get("units") or []:

            title = normalize_text(unit.get("title"))

            if gate_labels and title not in gate_labels:

                flags.append(
                    {
                        "flag": "SCENARIO_LABEL_NOT_FROM_GATE",
                        "severity": "hard",
                        "detail": clean_text(unit.get("title")),
                    }
                )

    if risky_sources(source_rows) and not has_temporal_caution(record):

        flags.append(
            {
                "flag": "TEMPORAL_CAUTION_MISSING",
                "severity": "medium",
                "detail": (
                    f"{len(risky_sources(source_rows))} 条 source 为 "
                    "temporary / partial / unresolved，"
                    "但 notes / limitations 未做时效提示"
                ),
            }
        )

    return flags


def load_entry_centroids(records):

    """
    用 frozen text-embedding-v4 向量
    计算每条 entry 的 question centroid。
    """

    needed = set()

    for record in records:

        for unit in record.get("units") or []:

            for key in unit.get("source_issue_keys") or []:

                needed.add(clean_text(key))

    if not needed or not EMBEDDING_FILE.exists():
        return {}, {}

    embeddings = {}

    normalized = {}

    with EMBEDDING_FILE.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(line)

            except json.JSONDecodeError:
                continue

            key = clean_text(obj.get("issue_key"))

            if key not in needed:
                continue

            embeddings[key] = obj.get("embedding") or []

            normalized[key] = clean_text(
                obj.get("question_normalized")
            )

            if len(embeddings) == len(needed):
                break

    centroids = {}

    for record in records:

        cluster_id = clean_text(record.get("cluster_id"))

        vectors = []

        for unit in record.get("units") or []:

            for key in unit.get("source_issue_keys") or []:

                vector = embeddings.get(clean_text(key))

                if vector:
                    vectors.append(vector)

        if not vectors:
            continue

        size = len(vectors[0])

        centroids[cluster_id] = [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(size)
        ]

    return centroids, normalized


def cosine(left, right):

    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))

    norm_left = sum(a * a for a in left) ** 0.5

    norm_right = sum(b * b for b in right) ** 0.5

    if not norm_left or not norm_right:
        return 0.0

    return dot / (norm_left * norm_right)


def load_cross_entry_pairs(records):

    """
    从 frozen Step 8.4 pair_classifications
    找出跨 entry 的 issue pair。
    """

    owner = {}

    for record in records:

        cluster_id = clean_text(record.get("cluster_id"))

        for unit in record.get("units") or []:

            for key in unit.get("source_issue_keys") or []:

                owner.setdefault(clean_text(key), set()).add(
                    cluster_id
                )

    if not PAIR_FILE.exists():
        return []

    hits = []

    with PAIR_FILE.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(line)

            except json.JSONDecodeError:
                continue

            left = owner.get(clean_text(obj.get("issue_key_a")))

            right = owner.get(clean_text(obj.get("issue_key_b")))

            if not left or not right:
                continue

            if left & right:
                continue

            hits.append(
                {
                    "cluster_a": sorted(left)[0],
                    "cluster_b": sorted(right)[0],
                    "similarity": obj.get("similarity"),
                    "relation": clean_text(obj.get("relation")),
                    "question_a": clean_text(obj.get("question_a")),
                    "question_b": clean_text(obj.get("question_b")),
                }
            )

    return hits


def cross_entry_flags(records):

    """
    Cross-entry 重复检测。

    信号 1（最强）：
      frozen pair_classifications 中
      跨 entry 的 same_intent pair

    信号 2：
      entry centroid cosine >= 0.88

    信号 3（仅提示）：
      entry centroid cosine >= 0.80
    """

    flags = {}

    def add(cluster_id, flag):

        flags.setdefault(cluster_id, []).append(flag)

    for hit in load_cross_entry_pairs(records):

        if hit["relation"] != "same_intent":
            continue

        detail = (
            f"frozen Step 8.4 判定 same_intent "
            f"(similarity={hit['similarity']:.4f}): "
            f"{hit['question_a']} / {hit['question_b']}"
        )

        add(
            hit["cluster_a"],
            {
                "flag": "CROSS_ENTRY_SAME_INTENT",
                "severity": "hard",
                "detail": (
                    f"与 {hit['cluster_b']} 存在 same_intent source，"
                    + detail
                ),
            },
        )

        add(
            hit["cluster_b"],
            {
                "flag": "CROSS_ENTRY_SAME_INTENT",
                "severity": "hard",
                "detail": (
                    f"与 {hit['cluster_a']} 存在 same_intent source，"
                    + detail
                ),
            },
        )

    centroids, _ = load_entry_centroids(records)

    cluster_ids = sorted(centroids.keys())

    for index in range(len(cluster_ids)):

        for other in range(index + 1, len(cluster_ids)):

            left = cluster_ids[index]

            right = cluster_ids[other]

            score = cosine(centroids[left], centroids[right])

            if score >= NEAR_DUPLICATE_COSINE:

                severity = "medium"

                flag = "CROSS_ENTRY_NEAR_DUPLICATE"

            elif score >= RELATED_COSINE:

                severity = "info"

                flag = "CROSS_ENTRY_RELATED"

            else:
                continue

            add(
                left,
                {
                    "flag": flag,
                    "severity": severity,
                    "detail": (
                        f"与 {right} 的 question centroid cosine "
                        f"{score:.4f}"
                    ),
                },
            )

            add(
                right,
                {
                    "flag": flag,
                    "severity": severity,
                    "detail": (
                        f"与 {left} 的 question centroid cosine "
                        f"{score:.4f}"
                    ),
                },
            )

    return flags


# ============================================================
# Verdict
# ============================================================

def build_verdict(validation, deterministic):

    flags = list(deterministic)

    if validation is not None:

        if validation.summary_grounding == "unsupported":

            flags.append(
                {
                    "flag": "SUMMARY_UNSUPPORTED",
                    "severity": "hard",
                    "detail": validation.summary_reason,
                }
            )

        for unit in validation.units:

            if unit.grounding == "unsupported":

                flags.append(
                    {
                        "flag": "UNIT_UNSUPPORTED",
                        "severity": "hard",
                        "detail": (
                            f"unit {unit.unit_index} "
                            f"{unit.unit_title}: {unit.reason}"
                        ),
                    }
                )

            if unit.condition_grounded is False:

                flags.append(
                    {
                        "flag": "SCENARIO_CONDITION_UNGROUNDED",
                        "severity": "hard",
                        "detail": (
                            f"unit {unit.unit_index} "
                            f"{unit.unit_title}"
                        ),
                    }
                )

        for item in validation.invented_elements:

            severity = (
                "hard" if item.severity == "material" else "medium"
            )

            flags.append(
                {
                    "flag": f"INVENTED_{item.element_type.upper()}",
                    "severity": severity,
                    "detail": (
                        f"[{item.location}] {item.text} "
                        f"- {item.reason}"
                    ),
                }
            )

        for unit in validation.units:

            for item in unit.invented_elements:

                severity = (
                    "hard"
                    if item.severity == "material"
                    else "medium"
                )

                flags.append(
                    {
                        "flag": f"INVENTED_{item.element_type.upper()}",
                        "severity": severity,
                        "detail": (
                            f"[unit {unit.unit_index} "
                            f"{unit.unit_title}] {item.text} "
                            f"- {item.reason}"
                        ),
                    }
                )

        if validation.conflict_wrongly_claimed:

            flags.append(
                {
                    "flag": "CONFLICT_WRONGLY_CLAIMED",
                    "severity": "hard",
                    "detail": "把独立原因/不同路径描述成互斥冲突",
                }
            )

        partial_units = [
            unit.unit_title
            for unit in validation.units
            if unit.grounding == "partially_grounded"
        ]

        if partial_units:

            flags.append(
                {
                    "flag": "UNIT_PARTIALLY_GROUNDED",
                    "severity": "medium",
                    "detail": ", ".join(partial_units),
                }
            )

        if (
            validation.summary_grounding == "partially_grounded"
        ):

            flags.append(
                {
                    "flag": "SUMMARY_PARTIALLY_GROUNDED",
                    "severity": "medium",
                    "detail": validation.summary_reason,
                }
            )

        if not validation.notes_grounded:

            flags.append(
                {
                    "flag": "NOTES_UNGROUNDED",
                    "severity": "medium",
                    "detail": "notes / limitations 含无来源表述",
                }
            )

    hard = [item for item in flags if item["severity"] == "hard"]

    medium = [item for item in flags if item["severity"] == "medium"]

    info = [item for item in flags if item["severity"] == "info"]

    if hard:
        status = "needs_fix"
    elif medium:
        status = "manual_review"
    else:
        status = "grounding_pass"

    return status, flags, hard, medium, info


# ============================================================
# Validate
# ============================================================

def require_api_key():

    if not os.environ.get("OPENAI_API_KEY"):

        raise SystemExit(
            "[FATAL] 环境变量 OPENAI_API_KEY 未设置。\n"
            "        本步骤需要调用 "
            f"{MODEL}，\n"
            "        请先 export OPENAI_API_KEY=... 再运行。\n"
            "        只想检查 deterministic 规则请用 --dry-run。"
        )


def validate_entry(record, source_rows, gate_row, deterministic):

    client = build_client()

    cluster_id = clean_text(record.get("cluster_id"))

    prompt = build_prompt(record, source_rows, gate_row)

    unit_count = len(record.get("units") or [])

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
                    extra_body={"reasoning_effort": "low"},
                )
            )

            elapsed = time.time() - started

            data = json.loads(
                response.choices[0].message.content
            )

            validation = (
                EntryGroundingValidation.model_validate(data)
            )

            validation, deterministic_overrides = (
                apply_exact_actor_overrides(
                    validation,
                    record,
                    source_rows,
                )
            )

            returned_indexes = sorted(
                unit.unit_index for unit in validation.units
            )

            if returned_indexes != list(range(1, unit_count + 1)):

                deterministic = deterministic + [
                    {
                        "flag": "UNIT_INDEX_MISMATCH",
                        "severity": "hard",
                        "detail": (
                            f"期望 1..{unit_count}，"
                            f"实际 {returned_indexes}"
                        ),
                    }
                ]

            status, flags, hard, medium, info = build_verdict(
                validation,
                deterministic,
            )

            return {
                "cluster_id": cluster_id,
                "canonical_question": record.get(
                    "canonical_question"
                ),
                "knowledge_structure": record.get(
                    "knowledge_structure"
                ),
                "generation_mode": record.get("generation_mode"),
                "unit_count": unit_count,
                "usable_source_count": len(source_rows),
                "risky_source_count": len(
                    risky_sources(source_rows)
                ),
                "summary_grounding": validation.summary_grounding,
                "summary_reason": validation.summary_reason,
                "unit_grounding": [
                    unit.model_dump()
                    for unit in validation.units
                ],
                "notes_grounded": validation.notes_grounded,
                "temporal_caution_needed":
                    validation.temporal_caution_needed,
                "temporal_caution_present":
                    validation.temporal_caution_present,
                "conflict_wrongly_claimed":
                    validation.conflict_wrongly_claimed,
                "invented_elements": [
                    item.model_dump()
                    for item in validation.invented_elements
                ],
                "grounding_status": status,
                "hard_flag_count": len(hard),
                "medium_flag_count": len(medium),
                "info_flag_count": len(info),
                "candidate_soft_flags": record.get("soft_flags") or [],
                "validation_flags": flags,
                "validator_reason": validation.reason,
                "validator_confidence": validation.confidence,
                "deterministic_overrides":
                    deterministic_overrides,
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
        "unit_count",
        "usable_source_count",
        "risky_source_count",
        "summary_grounding",
        "notes_grounded",
        "temporal_caution_needed",
        "temporal_caution_present",
        "conflict_wrongly_claimed",
        "grounding_status",
        "hard_flag_count",
        "medium_flag_count",
        "validator_confidence",
        "summary_reason",
        "validator_reason",
        "request_seconds",
    ]

    entry_df = df[
        [col for col in entry_cols if col in df.columns]
    ].copy()

    unit_rows = []

    flag_rows = []

    invented_rows = []

    for record in records:

        for unit in record.get("unit_grounding") or []:

            unit_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    "unit_index": unit.get("unit_index"),
                    "unit_title": unit.get("unit_title"),
                    "grounding": unit.get("grounding"),
                    "condition_grounded": unit.get(
                        "condition_grounded"
                    ),
                    "confidence": unit.get("confidence"),
                    "invented_count": len(
                        unit.get("invented_elements") or []
                    ),
                    "reason": unit.get("reason"),
                }
            )

            for item in unit.get("invented_elements") or []:

                invented_rows.append(
                    {
                        "cluster_id": record["cluster_id"],
                        "scope": "unit",
                        "unit_index": unit.get("unit_index"),
                        "unit_title": unit.get("unit_title"),
                        **item,
                    }
                )

        for item in record.get("invented_elements") or []:

            invented_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    "scope": "entry",
                    "unit_index": None,
                    "unit_title": None,
                    **item,
                }
            )

        for flag in record.get("validation_flags") or []:

            flag_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    **flag,
                }
            )

    units_df = pd.DataFrame(unit_rows)

    flags_df = pd.DataFrame(flag_rows)

    invented_df = pd.DataFrame(invented_rows)

    status_stats = (
        df["grounding_status"]
        .value_counts()
        .rename_axis("grounding_status")
        .reset_index(name="count")
    )

    flag_stats = (
        flags_df["flag"]
        .value_counts()
        .rename_axis("flag")
        .reset_index(name="count")
        if len(flags_df)
        else pd.DataFrame(columns=["flag", "count"])
    )

    unit_stats = (
        units_df["grounding"]
        .value_counts()
        .rename_axis("grounding")
        .reset_index(name="count")
        if len(units_df)
        else pd.DataFrame(columns=["grounding", "count"])
    )

    summary = pd.DataFrame(
        [
            {"metric": "entry_count", "value": len(df)},
            {
                "metric": "grounding_pass_count",
                "value": int(
                    (df["grounding_status"] == "grounding_pass").sum()
                ),
            },
            {
                "metric": "manual_review_count",
                "value": int(
                    (df["grounding_status"] == "manual_review").sum()
                ),
            },
            {
                "metric": "needs_fix_count",
                "value": int(
                    (df["grounding_status"] == "needs_fix").sum()
                ),
            },
            {
                "metric": "unit_count",
                "value": int(df["unit_count"].sum()),
            },
            {
                "metric": "invented_element_count",
                "value": len(invented_df),
            },
            {
                "metric": "material_invented_count",
                "value": int(
                    (invented_df["severity"] == "material").sum()
                )
                if len(invented_df)
                else 0,
            },
            {
                "metric": "hard_flag_total",
                "value": int(df["hard_flag_count"].sum()),
            },
            {
                "metric": "medium_flag_total",
                "value": int(df["medium_flag_count"].sum()),
            },
            {
                "metric": "cross_entry_same_intent_flags",
                "value": int(
                    flags_df["flag"].eq(
                        "CROSS_ENTRY_SAME_INTENT"
                    ).sum()
                )
                if len(flags_df)
                else 0,
            },
            {
                "metric": "cross_entry_near_duplicate_flags",
                "value": int(
                    flags_df["flag"].eq(
                        "CROSS_ENTRY_NEAR_DUPLICATE"
                    ).sum()
                )
                if len(flags_df)
                else 0,
            },
            {"metric": "model", "value": MODEL},
        ]
    )

    needs_fix_df = entry_df[
        entry_df["grounding_status"] == "needs_fix"
    ]

    review_df = entry_df[
        entry_df["grounding_status"] == "manual_review"
    ]

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        summary.to_excel(
            writer, index=False, sheet_name="summary"
        )

        status_stats.to_excel(
            writer, index=False, sheet_name="status_stats"
        )

        unit_stats.to_excel(
            writer, index=False, sheet_name="unit_stats"
        )

        flag_stats.to_excel(
            writer, index=False, sheet_name="flag_stats"
        )

        entry_df.to_excel(
            writer, index=False, sheet_name="entry_validations"
        )

        units_df.to_excel(
            writer, index=False, sheet_name="unit_validations"
        )

        invented_df.to_excel(
            writer, index=False, sheet_name="invented_elements"
        )

        flags_df.to_excel(
            writer, index=False, sheet_name="validation_flags"
        )

        needs_fix_df.to_excel(
            writer, index=False, sheet_name="needs_fix"
        )

        review_df.to_excel(
            writer, index=False, sheet_name="manual_review"
        )


# ============================================================
# Main
# ============================================================

def run_parse_check(tasks):

    """
    不调用 LLM；构造最小合法对象验证 schema 和 prompt。
    """

    if not tasks:
        raise RuntimeError("没有可检查的任务")

    record, source_rows, gate_row, _ = tasks[0]

    prompt = build_prompt(record, source_rows, gate_row)

    units = [
        UnitGrounding(
            unit_index=index,
            unit_title=clean_text(unit.get("title")),
            grounding="grounded",
            condition_grounded=None,
            invented_elements=[],
            reason="parse check",
            confidence=1.0,
        )
        for index, unit in enumerate(
            record.get("units") or [],
            start=1,
        )
    ]

    sample = EntryGroundingValidation(
        summary_grounding="grounded",
        summary_reason="parse check",
        units=units,
        notes_grounded=True,
        temporal_caution_needed=False,
        temporal_caution_present=True,
        conflict_wrongly_claimed=False,
        invented_elements=[],
        reason="parse check",
        confidence=1.0,
    )

    if len(sample.units) != len(record.get("units") or []):
        raise RuntimeError("parse check unit 数量不一致")

    print()
    print("[PARSE CHECK] schema OK")

    print(
        f"sample cluster: {clean_text(record.get('cluster_id'))} "
        f"| units: {len(sample.units)} "
        f"| prompt chars: {len(prompt)}"
    )

    print(
        "prompt contains output schema:",
        '"summary_grounding"' in SYSTEM_PROMPT,
    )


def resolve_source_cluster_ids(record):

    """
    Step 11.6 起支持 merged entry:

    merged_from_clusters 记录被合并的 cluster，
    usable source 按 primary + merged 求并集。
    """

    cluster_id = clean_text(
        record.get("cluster_id")
    )

    merged_ids = [
        clean_text(item)
        for item in record.get(
            "merged_from_clusters"
        ) or []
        if clean_text(item)
    ]

    return [cluster_id] + merged_ids


def resolve_source_rows(
    sources_by_cluster,
    cluster_ids,
):

    rows = []

    seen = set()

    for cluster_id in cluster_ids:

        for row in sources_by_cluster.get(
            cluster_id, []
        ):

            key = clean_text(
                row.get("issue_key")
            )

            if key in seen:
                continue

            seen.add(key)

            rows.append(row)

    return rows


def resolve_gate_row(gate_rows, cluster_ids):

    """
    merged entry 的 gate 计数取各 cluster 之和:

    cause_items -> independent_cause_count 求和
    scenario_sections -> scenario_count 求和,
    evidence_roles 拼接。

    单 entry 行为与原逻辑完全一致。
    """

    primary = gate_rows.get(
        cluster_ids[0], {}
    )

    if len(cluster_ids) == 1:
        return primary

    merged = dict(primary)

    roles = list(
        primary.get("evidence_roles") or []
    )

    cause_sum = int(
        primary.get("independent_cause_count")
        or 0
    )

    scenario_sum = int(
        primary.get("scenario_count") or 0
    )

    for cluster_id in cluster_ids[1:]:

        other = gate_rows.get(cluster_id, {})

        roles = roles + list(
            other.get("evidence_roles") or []
        )

        cause_sum += int(
            other.get("independent_cause_count")
            or 0
        )

        scenario_sum += int(
            other.get("scenario_count") or 0
        )

    merged["evidence_roles"] = roles

    merged["independent_cause_count"] = cause_sum

    merged["scenario_count"] = scenario_sum

    return merged


def main():

    print("=" * 70)
    print("Step 11.5 - Mixed Candidate KB Grounding Validation")
    print("=" * 70)

    print(f"Model: {MODEL}")

    records = load_candidates()

    sources_by_cluster = load_usable_sources()

    gate_rows = load_gate_rows()

    print(f"Candidate entries: {len(records)}")

    if not DRY_RUN and not PARSE_CHECK:

        require_api_key()

    cross_flags = cross_entry_flags(records)

    same_intent_entries = sorted(
        cluster_id
        for cluster_id, items in cross_flags.items()
        if any(
            item["flag"] == "CROSS_ENTRY_SAME_INTENT"
            for item in items
        )
    )

    near_entries = sorted(
        cluster_id
        for cluster_id, items in cross_flags.items()
        if any(
            item["flag"] == "CROSS_ENTRY_NEAR_DUPLICATE"
            for item in items
        )
    )

    print(
        f"cross-entry same_intent: "
        f"{same_intent_entries or 'none'}"
    )

    print(
        f"cross-entry near_duplicate(cosine>="
        f"{NEAR_DUPLICATE_COSINE}): {near_entries or 'none'}"
    )

    tasks = []

    for record in records:

        cluster_id = clean_text(record.get("cluster_id"))

        if ONLY_IDS and cluster_id not in ONLY_IDS:
            continue

        source_cluster_ids = (
            resolve_source_cluster_ids(record)
        )

        source_rows = resolve_source_rows(
            sources_by_cluster,
            source_cluster_ids,
        )

        gate_row = resolve_gate_row(
            gate_rows,
            source_cluster_ids,
        )

        if not source_rows:

            print(f"[SKIP] {cluster_id}: NO_VALID_SOURCE")

            continue

        if not gate_row:

            print(f"[SKIP] {cluster_id}: NO_GATE_RECORD")

            continue

        deterministic = deterministic_flags(
            record,
            source_rows,
            gate_row,
        ) + cross_flags.get(cluster_id, [])

        tasks.append((record, source_rows, gate_row, deterministic))

    if DRY_RUN:

        print()
        print("-" * 70)
        print("DRY RUN - 不调用 LLM，不写任何输出")
        print("-" * 70)

        for record, source_rows, gate_row, deterministic in tasks:

            print(
                f"{clean_text(record.get('cluster_id')):15} "
                f"| {clean_text(record.get('generation_mode')):18} "
                f"| units={len(record.get('units') or [])} "
                f"| sources={len(source_rows)} "
                f"| risky={len(risky_sources(source_rows))}"
            )

            print(
                f"    Q: {clean_text(record.get('canonical_question'))}"
            )

            if deterministic:

                for flag in deterministic:

                    print(
                        f"    [{flag['severity']}] "
                        f"{flag['flag']}: {flag['detail']}"
                    )

            else:

                print("    deterministic flags: none")

        print()
        print(f"待验证 entry: {len(tasks)}")

        print("DRY RUN 完成")

        return

    if PARSE_CHECK:

        run_parse_check(tasks)

        return

    processed = load_processed()

    pending = []

    for task in tasks:

        cluster_id = clean_text(task[0].get("cluster_id"))

        if cluster_id in processed:

            print(f"[SKIP] {cluster_id}: 已存在")

            continue

        pending.append(task)

    print(f"本次待验证: {len(pending)}")

    total = len(pending)

    completed = 0

    started_at = time.time()

    lock = threading.Lock()

    failures = []

    if pending:

        with OUTPUT_JSONL.open("a", encoding="utf-8") as fout:

            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                future_map = {
                    executor.submit(
                        validate_entry,
                        record,
                        source_rows,
                        gate_row,
                        deterministic,
                    ): clean_text(record.get("cluster_id"))
                    for record, source_rows, gate_row, deterministic
                    in pending
                }

                for future in as_completed(future_map):

                    cluster_id = future_map[future]

                    try:

                        result = future.result()

                    except Exception as exc:

                        print(f"[FAILED] {cluster_id}: {exc}")

                        with lock:

                            failures.append(
                                (cluster_id, repr(exc))
                            )

                        continue

                    with lock:

                        fout.write(
                            json.dumps(result, ensure_ascii=False)
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
                        f"| {result['grounding_status']} "
                        f"| hard={result['hard_flag_count']} "
                        f"| medium={result['medium_flag_count']} "
                        f"| summary="
                        f"{result['summary_grounding']} "
                        f"| ETA {eta / 60:.1f} min"
                    )

    if failures:

        error_log = OUTPUT_JSONL.with_suffix(".errors.log")

        error_log.write_text(
            "".join(
                f"{cluster_id}\t{message}\n"
                for cluster_id, message in failures
            ),
            encoding="utf-8",
        )

        print()
        print("!" * 70)
        print(f"[ERROR] {len(failures)} 个 entry 验证失败")
        print("!" * 70)

        for cluster_id, message in failures:

            print(f"{cluster_id}: {message}")

        print()
        print(f"错误详情: {error_log}")

        if completed == 0:

            print()
            print(
                "[FATAL] 没有任何 entry 验证成功，"
                "不生成 Excel。"
            )

            raise SystemExit(1)

    export_excel()

    print()
    print("=" * 70)
    print(
        f"完成: 成功 {completed} / {total}"
    )
    print("=" * 70)

    print(f"JSONL: {OUTPUT_JSONL}")

    print(f"Excel: {OUTPUT_XLSX}")

    print()
    print("注意：本步骤只是 grounding validation。")

    print("后续仍需 temporal / conflict 处理与")

    print("final publishability gate。")


if __name__ == "__main__":
    main()
