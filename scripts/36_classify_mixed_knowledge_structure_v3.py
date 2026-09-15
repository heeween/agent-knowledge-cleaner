import os
import re
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

MIXED_AUDIT_FILE = (
    OUTPUT_DIR
    / "mixed_cluster_audits_v2.xlsx"
)

GROUNDING_FILE = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3.xlsx"
)

MODEL = "qwen3.8-max"


MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class EvidenceRole(BaseModel):

    issue_key: str

    role: Literal[
        "core_answer",
        "cause",
        "procedure",
        "condition",
        "scenario",
        "limitation",
        "diagnostic",
        "status",
        "temporary_workaround",
        "supporting_detail",
        "weak_evidence",
    ]

    scenario_label: str | None = None

    scenario_evidence: str | None = None

    reason: str


class ConflictFact(BaseModel):

    fact_dimension: str

    statement_a: str

    statement_b: str

    issue_keys_a: list[str] = Field(
        default_factory=list
    )

    issue_keys_b: list[str] = Field(
        default_factory=list
    )

    same_condition: bool

    mutually_exclusive: bool

    evidence_reason: str


class StructureClassification(BaseModel):

    knowledge_structure: Literal[
        "direct_synthesis",
        "structured_howto",
        "multiple_causes",
        "scenario_branches",
        "troubleshooting",
        "conflict_temporal",
        "insufficient_evidence",
    ]

    confidence: float

    canonical_question: str

    evidence_roles: list[EvidenceRole]

    independent_cause_count: int

    scenario_count: int

    has_true_conflict: bool

    conflict_facts: list[ConflictFact] = Field(
        default_factory=list
    )

    temporal_dependency: Literal[
        "none",
        "minor",
        "material",
        "unknown",
    ]

    current_validity_resolved: bool

    generation_ready: bool

    recommended_generation_mode: Literal[
        "single_answer",
        "howto_sections",
        "cause_items",
        "scenario_sections",
        "troubleshooting_steps",
        "do_not_generate",
    ]

    reason: str

    risk_flags: list[str]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
你正在执行 CRM RAG Mixed Knowledge Structure Gate v3。

这些 Cluster 已经过：

1. Question Intent Cluster Validation
2. Mixed Cluster Intent Audit
3. Source Q/A Grounding Gate

输入中的每条 evidence 已经通过：

usable_for_kb = true

本轮仍然绝对不要生成最终 KB。

你只负责判断：

这些 grounded evidence
应该采用什么知识结构。

============================================================
最高优先级
============================================================

必须严格区分：

- 不同原因
- 不同处理方法
- 不同条件场景
- 真正事实冲突
- 时效风险

这五者不能混为一谈。

尤其禁止：

答案不同
→ true conflict

临时状态不同
→ true conflict

解决方法不同
→ true conflict

============================================================
一、Different Causes 永远不自动等于 Conflict
============================================================

例如：

问题：
“为什么已经回访了还显示待回访？”

Evidence A：
可能是车辆/客户匹配问题。

Evidence B：
可能是系统状态没有更新。

这是：

同一个问题
+
两个独立原因

应该：

knowledge_structure = multiple_causes
has_true_conflict = false


即使：

原因 A 来自用户侧
原因 B 来自系统侧

仍然不是 conflict。


============================================================
二、Different Workarounds 永远不自动等于 Conflict
============================================================

例如：

问题：
“车辆里程如何修改？”

Evidence A：
直接点击修改。

Evidence B：
如果页面状态异常，刷新后再修改。

这两个答案完全可以同时成立。

B 很可能是异常情况下的 fallback。

应该优先：

structured_howto

或：

troubleshooting

不能因为：

“直接修改”
vs
“刷新后修改”

就判 true conflict。


又例如：

刷新
重新登录
换浏览器

都是不同 workaround。

不是冲突。


============================================================
三、Temporary 不是 Conflict
============================================================

temporary / unresolved / partial

只能说明：

存在时效或稳定性风险。

不能作为：

has_true_conflict = true

的依据。


例如：

企微车主画像空白：

A：
腾讯侧加载延迟。

B：
企业微信可见范围或系统 Bug。

两条都可能 temporary。

但仍然只是：

不同原因。

不是 true conflict。


============================================================
四、什么才是真正 Conflict
============================================================

只有同时满足下面全部条件：

1. 相同用户问题
2. 相同适用条件
3. 回答同一个事实维度
4. 两个结论互斥
5. 不能用场景、角色、版本、条件差异解释

才允许：

has_true_conflict = true


例如：

问题：
“默认密码是什么？”

A：
默认密码是 123456。

B：
默认密码是手机号。

如果 Source 没有明确说明：

不同角色
不同版本
不同租户
不同产品

则这是：

相同事实维度：
默认密码

互斥结论：

123456
vs
手机号

→ true conflict


又例如：

A：
系统支持修改手机号。

B：
系统明确不支持修改手机号。

相同条件下：

→ true conflict


============================================================
五、True Conflict 必须输出 conflict_facts
============================================================

如果：

has_true_conflict = true

必须至少输出 1 个 conflict_fact。


每个 conflict_fact 必须说明：

fact_dimension：
到底冲突的是哪个事实变量

statement_a：
结论 A

statement_b：
结论 B

issue_keys_a：
支持 A 的 source issue_key

issue_keys_b：
支持 B 的 source issue_key

same_condition：
是否同条件

mutually_exclusive：
是否互斥

evidence_reason：
为什么确实构成冲突


如果无法清楚填写：

fact_dimension

则不得判 true conflict。


错误：

fact_dimension =
“原因不同”

这不是事实变量冲突。


正确：

fact_dimension =
“CRM 默认密码规则”


============================================================
六、Conflict Fact Same-Fact Test
============================================================

判 true conflict 前必须问：

A 和 B 是否回答：

完全相同的事实变量？


例：

“为什么登录失败？”

密码错误
vs
登录地址错误

这是两个原因。

不是相同事实变量。

→ false


例：

“默认密码是什么？”

123456
vs
手机号

是同一个事实变量。

→ true


例：

“异常怎么办？”

刷新
vs
重登

这是两个方法。

不是同一事实结论。

→ false


============================================================
七、Scenario 必须被 Source 明确 Ground
============================================================

scenario_branches 只能在：

Source 明确给出不同适用条件

时使用。


例如：

Evidence A：
管理员账号需要从 A 入口登录。

Evidence B：
普通员工账号从 B 入口登录。

明确条件：

管理员账号
普通员工账号

→ scenario_branches


但是：

Evidence A：
需要短信签名备案。

Evidence B：
需要购买短信套餐。

不能自行创造：

“备案场景”
和
“套餐场景”。

因为 Evidence 只是两个可能步骤/要求，
并没有说它们是互斥适用条件。


============================================================
八、Scenario Evidence 强制要求
============================================================

如果某条 EvidenceRole：

role = scenario

则必须填写：

scenario_label
scenario_evidence


scenario_evidence 必须直接来自：

supported_answer
或
supported_solution

中的明确条件表达。


禁止：

根据模型理解自行创造条件。


例如输入没有：

“如果是管理员”
“如果手机号已存在”
“对于旧版本”

就不能自行制造这些条件。


============================================================
九、Scenario Branch 判定测试
============================================================

在判：

scenario_branches

前必须通过：

Scenario Grounding Test。


要求：

1. 至少 2 个明确 Scenario
2. 每个 Scenario 都有 source-grounded 条件
3. 条件不是根据不同答案反推出来的
4. Scenario 之间确实导致不同处理路径


如果只是：

步骤 A
步骤 B

没有适用条件，

则不应：

scenario_branches


应考虑：

structured_howto


============================================================
十、knowledge_structure
============================================================


------------------------------------------------------------
1. direct_synthesis
------------------------------------------------------------

多个 evidence 回答的是同一个核心答案。

区别只是：

- 措辞
- 细节
- 完整度

不需要：

原因列表
条件分支
排查结构

→ direct_synthesis


------------------------------------------------------------
2. structured_howto
------------------------------------------------------------

主要问题是：

如何做
怎么操作
流程是什么
在哪里操作

Evidence 可组成：

- 前置条件
- 步骤
- 注意事项
- fallback
- 补充操作

即使不同 Source 提供不同步骤，

只要它们可以属于同一流程，

仍然：

structured_howto


例如：

A：
点击修改。

B：
状态异常时先刷新再修改。

→ structured_howto


------------------------------------------------------------
3. multiple_causes
------------------------------------------------------------

问题主要问：

为什么
什么原因
为什么异常

并且存在 >=2 个独立原因。


如果能自然写成：

“可能原因包括 A、B、C”

则优先：

multiple_causes


不同原因对应不同处理方法，

仍然可以：

multiple_causes。


------------------------------------------------------------
4. scenario_branches
------------------------------------------------------------

只有：

不同适用条件
+
不同答案

时使用。


结构：

条件 A
→ 方法 A

条件 B
→ 方法 B


每个条件都必须有 Source Evidence。


------------------------------------------------------------
5. troubleshooting
------------------------------------------------------------

问题是：

异常怎么办

Evidence 主要是：

- 排查项
- 诊断
- workaround
- fallback

如果 Source 有明确检查顺序：

先 A
再 B
最后 C

可判：

troubleshooting


如果 Source 没有顺序，

禁止自动发明顺序。


如果核心是多个独立原因，

优先：

multiple_causes。


------------------------------------------------------------
6. conflict_temporal
------------------------------------------------------------

只用于：

A. 真正 same-fact conflict

或

B. 明显存在版本/规则变化，
且当前有效状态无法确定


不能只因为：

temporary
unresolved
不同原因
不同 workaround

判 conflict_temporal。


------------------------------------------------------------
7. insufficient_evidence
------------------------------------------------------------

虽然 Q/A Grounding 通过，

但仍然不足以形成稳定知识结构。


例如：

- 只有单次 temporary Bug
- 只有模糊 workaround
- 当前有效状态无法确认
- 核心规则未知
- 答案虽然相关，但无法安全泛化
- Source 数量或内容不足


============================================================
十一、Temporal Dependency
============================================================

none：

核心知识明显稳定。


minor：

部分 Source 是 temporary，
但核心知识仍可复用。


material：

知识是否正确高度依赖：

- 当前版本
- Bug 是否已经修复
- 当前 URL
- 当前默认密码
- 当前菜单入口
- 当前产品政策
- 临时后台状态


如果：

temporal_dependency = material

并且：

current_validity_resolved = false

则必须：

generation_ready = false

recommended_generation_mode =
do_not_generate


这不要求存在 true conflict。


============================================================
十二、current_validity_resolved
============================================================

表示：

现有 evidence 是否足以确认：

当前哪个结论仍然有效。


例如：

Source 中明确说明：

“旧版入口已废弃，
现在统一使用新入口。”

则：

current_validity_resolved = true


如果只是看到：

旧答案 A
新答案 B

但没有可靠时序依据，

则：

false


不要自行猜当前版本。


============================================================
十三、Evidence Roles
============================================================

每个 issue_key 必须且只能输出一次。

允许：

core_answer
cause
procedure
condition
scenario
limitation
diagnostic
status
temporary_workaround
supporting_detail
weak_evidence


如果回答主要是：

“因为 X”

即使 Source temporary，

角色仍然可以：

cause


不要因为 temporary
机械标成 temporary_workaround。


============================================================
十四、independent_cause_count
============================================================

只统计真正独立原因。

同义表达只算 1。


如果：

knowledge_structure = multiple_causes

必须：

independent_cause_count >= 2


============================================================
十五、scenario_count
============================================================

只统计：

有明确 Source Grounded Condition

的 scenario。


如果：

knowledge_structure = scenario_branches

必须：

scenario_count >= 2


============================================================
十六、generation_ready
============================================================

generation_ready = true

只表示：

可以进入候选 KB 生成。

不是正式发布。


可为 true：

direct_synthesis
structured_howto
multiple_causes
scenario_branches
troubleshooting


必须 false：

conflict_temporal
insufficient_evidence


以及：

temporal_dependency = material
且
current_validity_resolved = false


============================================================
十七、特别案例提醒
============================================================

Case A：

“已完成回访还显示待回访”

车辆匹配问题
+
状态更新问题

→ multiple_causes
→ has_true_conflict=false


Case B：

“企微车主画像空白”

加载延迟
+
可见范围/Bug

→ 多原因或排查
→ 不得仅因 temporary 判 conflict


Case C：

“车辆里程怎么修改”

直接修改
+
刷新后再修改

→ structured_howto / troubleshooting
→ 不得判 true conflict


Case D：

“默认密码是什么”

123456
vs
手机号
vs
6个1

若无明确条件解释：

→ true conflict


Case E：

“短信如何开通”

签名备案
+
购买套餐

如果 Source 没有明确说明它们分别属于不同条件：

→ 不允许 scenario_branches

更可能：

structured_howto
或
insufficient_evidence


============================================================
十八、严格禁止
============================================================

禁止：

- 生成最终 KB Answer
- 补 CRM 常识
- 补 URL
- 补菜单
- 补时间
- 补角色
- 补条件
- 补 Scenario
- 推断最新版
- 把 Different Cause 当 Conflict
- 把 Different Workaround 当 Conflict
- 把 Temporary 当 Conflict
- 把不同步骤自动解释成 Scenario


============================================================
十九、输出
============================================================

只输出合法 JSON。

格式：

{
  "knowledge_structure": "direct_synthesis | structured_howto | multiple_causes | scenario_branches | troubleshooting | conflict_temporal | insufficient_evidence",
  "confidence": 0.0,
  "canonical_question": "规范问题",
  "evidence_roles": [
    {
      "issue_key": "输入 issue_key",
      "role": "core_answer | cause | procedure | condition | scenario | limitation | diagnostic | status | temporary_workaround | supporting_detail | weak_evidence",
      "scenario_label": null,
      "scenario_evidence": null,
      "reason": "简短说明"
    }
  ],
  "independent_cause_count": 0,
  "scenario_count": 0,
  "has_true_conflict": false,
  "conflict_facts": [],
  "temporal_dependency": "none | minor | material | unknown",
  "current_validity_resolved": true,
  "generation_ready": true,
  "recommended_generation_mode": "single_answer | howto_sections | cause_items | scenario_sections | troubleshooting_steps | do_not_generate",
  "reason": "结构判断依据",
  "risk_flags": []
}
"""


# ============================================================
# Client
# ============================================================

def build_client():

    return OpenAI(
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


# ============================================================
# Utils
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    return str(value).strip()


def normalize_text(text):

    text = clean_text(text)

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    text = re.sub(
        r"[，。！？；：、,.!?;:\-_()（）\[\]【】]",
        "",
        text,
    )

    return text.lower()


def contains_grounded_phrase(
    phrase,
    source_text,
):

    phrase_n = normalize_text(
        phrase
    )

    source_n = normalize_text(
        source_text
    )

    if not phrase_n:
        return False

    if not source_n:
        return False

    # 完整短语命中
    if phrase_n in source_n:
        return True

    # 对较长 scenario evidence 做宽松 token overlap
    if len(phrase_n) >= 6:

        chars = set(
            phrase_n
        )

        if not chars:
            return False

        overlap = (
            len(
                chars
                & set(source_n)
            )
            / len(chars)
        )

        if overlap >= 0.85:
            return True

    return False


def load_processed():

    processed = set()

    if not OUTPUT_JSONL.exists():
        return processed

    with OUTPUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(
                    line
                )

                cluster_id = obj.get(
                    "cluster_id"
                )

                if cluster_id:

                    processed.add(
                        cluster_id
                    )

            except Exception:
                pass

    return processed


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    cluster_row,
    source_rows,
):

    blocks = []

    for idx, row in enumerate(
        source_rows,
        start=1,
    ):

        blocks.append(
            f"""
============================================================
GROUNDED EVIDENCE {idx}
============================================================

issue_key:
{clean_text(row.get("issue_key"))}

question:
{clean_text(row.get("question"))}

question_normalized:
{clean_text(row.get("question_normalized"))}

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

alignment:
{clean_text(row.get("alignment"))}

alignment_confidence:
{clean_text(row.get("alignment_confidence"))}
""".strip()
        )

    return f"""
请对下面 Mixed Cluster 做 Knowledge Structure Classification v3。

============================================================
CLUSTER
============================================================

cluster_id:
{clean_text(cluster_row.get("cluster_id"))}

canonical_question:
{clean_text(cluster_row.get("canonical_question"))}

core_question:
{clean_text(cluster_row.get("core_question"))}

Step 11.1 classification:
{clean_text(cluster_row.get("classification"))}

Step 11.1 answer_structure_types:
{clean_text(cluster_row.get("answer_structure_types"))}

Step 11.1 has_conflict:
{clean_text(cluster_row.get("has_conflict"))}

Step 11.1 has_temporal_risk:
{clean_text(cluster_row.get("has_temporal_risk"))}

grounded_source_count:
{len(source_rows)}


{chr(10).join(blocks)}
"""


# ============================================================
# Deterministic Validation
# ============================================================

def validate_conflict_facts(
    result,
    source_map,
):

    errors = []

    if not result.has_true_conflict:

        if result.conflict_facts:

            errors.append(
                "has_true_conflict=false "
                "但 conflict_facts 非空"
            )

        return errors

    if not result.conflict_facts:

        errors.append(
            "has_true_conflict=true "
            "但没有 conflict_facts"
        )

        return errors

    all_keys = set(
        source_map.keys()
    )

    valid_fact_count = 0

    for idx, fact in enumerate(
        result.conflict_facts,
        start=1,
    ):

        if not clean_text(
            fact.fact_dimension
        ):

            errors.append(
                f"conflict_fact#{idx} "
                f"缺少 fact_dimension"
            )

        keys_a = set(
            fact.issue_keys_a
        )

        keys_b = set(
            fact.issue_keys_b
        )

        if not keys_a:

            errors.append(
                f"conflict_fact#{idx} "
                f"issue_keys_a 为空"
            )

        if not keys_b:

            errors.append(
                f"conflict_fact#{idx} "
                f"issue_keys_b 为空"
            )

        if not keys_a.issubset(
            all_keys
        ):

            errors.append(
                f"conflict_fact#{idx} "
                f"issue_keys_a 含未知 issue"
            )

        if not keys_b.issubset(
            all_keys
        ):

            errors.append(
                f"conflict_fact#{idx} "
                f"issue_keys_b 含未知 issue"
            )

        if keys_a & keys_b:

            errors.append(
                f"conflict_fact#{idx} "
                f"A/B issue 重叠"
            )

        if not fact.same_condition:

            errors.append(
                f"conflict_fact#{idx} "
                f"same_condition=false，"
                f"不能作为 true conflict"
            )

        if not fact.mutually_exclusive:

            errors.append(
                f"conflict_fact#{idx} "
                f"mutually_exclusive=false，"
                f"不能作为 true conflict"
            )

        # 必须至少能在各自 source 中找到 statement
        a_supported = False

        for key in keys_a:

            row = source_map.get(
                key,
                {}
            )

            text = (
                clean_text(
                    row.get(
                        "supported_answer"
                    )
                )
                + "\n"
                + clean_text(
                    row.get(
                        "supported_solution"
                    )
                )
            )

            if contains_grounded_phrase(
                fact.statement_a,
                text,
            ):

                a_supported = True
                break

        b_supported = False

        for key in keys_b:

            row = source_map.get(
                key,
                {}
            )

            text = (
                clean_text(
                    row.get(
                        "supported_answer"
                    )
                )
                + "\n"
                + clean_text(
                    row.get(
                        "supported_solution"
                    )
                )
            )

            if contains_grounded_phrase(
                fact.statement_b,
                text,
            ):

                b_supported = True
                break

        if not a_supported:

            errors.append(
                f"conflict_fact#{idx} "
                f"statement_a 未在对应 grounded "
                f"source 中找到依据"
            )

        if not b_supported:

            errors.append(
                f"conflict_fact#{idx} "
                f"statement_b 未在对应 grounded "
                f"source 中找到依据"
            )

        if (
            fact.same_condition
            and fact.mutually_exclusive
            and a_supported
            and b_supported
        ):

            valid_fact_count += 1

    if valid_fact_count == 0:

        errors.append(
            "没有任何通过 deterministic "
            "grounding 的 conflict_fact"
        )

    return errors


def validate_scenarios(
    result,
    source_map,
):

    errors = []

    # role 不是唯一依据：
    # 模型可能把 scenario evidence
    # 标成 procedure / condition，
    # 但只要给出 scenario_label
    # 或 scenario_evidence，
    # 就必须做 Source Grounding 校验
    scenario_items = [
        item
        for item
        in result.evidence_roles
        if item.role == "scenario"
        or clean_text(item.scenario_label)
        or clean_text(item.scenario_evidence)
    ]

    if (
        result.knowledge_structure
        != "scenario_branches"
    ):

        return errors

    if result.scenario_count < 2:

        errors.append(
            "scenario_branches "
            "但 scenario_count < 2"
        )

    labels = set()

    grounded_labels = set()

    for item in scenario_items:

        label = clean_text(
            item.scenario_label
        )

        evidence = clean_text(
            item.scenario_evidence
        )

        if not label:

            errors.append(
                f"{item.issue_key}: "
                f"scenario 缺少 scenario_label"
            )

            continue

        labels.add(
            label
        )

        if not evidence:

            errors.append(
                f"{item.issue_key}: "
                f"scenario 缺少 scenario_evidence"
            )

            continue

        source = source_map.get(
            item.issue_key,
            {}
        )

        source_text = (
            clean_text(
                source.get(
                    "supported_answer"
                )
            )
            + "\n"
            + clean_text(
                source.get(
                    "supported_solution"
                )
            )
        )

        if contains_grounded_phrase(
            evidence,
            source_text,
        ):

            grounded_labels.add(
                label
            )

        else:

            errors.append(
                f"{item.issue_key}: "
                f"scenario_evidence "
                f"未在 grounded source 中命中"
            )

    if len(labels) < 2:

        errors.append(
            "scenario_branches "
            "没有至少 2 个不同 scenario_label"
        )

    if len(grounded_labels) < 2:

        errors.append(
            "scenario_branches "
            "没有至少 2 个 Source-grounded scenario"
        )

    return errors


def validate_cause_conflict_consistency(
    result,
):

    errors = []

    cause_roles = [
        item
        for item
        in result.evidence_roles
        if item.role == "cause"
    ]

    # 如果模型自己说有 >=2 独立原因，
    # 又大量 evidence 都是 cause，
    # 但还想 true conflict，
    # 必须有严格 conflict_fact 支撑。
    if (
        result.independent_cause_count >= 2
        and
        len(cause_roles) >= 2
        and
        result.has_true_conflict
        and
        not result.conflict_facts
    ):

        errors.append(
            "识别出多个独立原因，"
            "同时设置 true conflict，"
            "但没有 conflict_fact"
        )

    return errors


def apply_hard_rules(
    result,
    source_rows,
):

    source_map = {
        clean_text(
            row.get(
                "issue_key"
            )
        ):
        row
        for row in source_rows
    }

    hard_rule_flags = []

    # --------------------------------------------------------
    # 1. issue_key 完整性
    # --------------------------------------------------------

    allowed_keys = set(
        source_map.keys()
    )

    output_keys = [
        item.issue_key
        for item
        in result.evidence_roles
    ]

    if set(output_keys) != allowed_keys:

        raise RuntimeError(
            "evidence_roles 未完整覆盖 source issue_key。"
            f" expected={sorted(allowed_keys)}"
            f" actual={sorted(set(output_keys))}"
        )

    if len(output_keys) != len(
        set(output_keys)
    ):

        raise RuntimeError(
            "evidence_roles 中存在重复 issue_key"
        )

    # --------------------------------------------------------
    # 2. multiple causes 基础规则
    # --------------------------------------------------------

    if (
        result.knowledge_structure
        == "multiple_causes"
        and
        result.independent_cause_count < 2
    ):

        result.generation_ready = False

        result.recommended_generation_mode = (
            "do_not_generate"
        )

        hard_rule_flags.append(
            "MULTICAUSE_COUNT_LT_2"
        )

    # --------------------------------------------------------
    # 3. Scenario grounding
    # --------------------------------------------------------

    scenario_errors = validate_scenarios(
        result,
        source_map,
    )

    if scenario_errors:

        hard_rule_flags.extend(
            [
                "SCENARIO_GROUNDING_FAIL: "
                + err
                for err in scenario_errors
            ]
        )

        # 不允许虚构 scenario 继续生成
        if (
            result.knowledge_structure
            == "scenario_branches"
        ):

            result.generation_ready = False

            result.recommended_generation_mode = (
                "do_not_generate"
            )

    # --------------------------------------------------------
    # 4. Conflict Grounding
    # --------------------------------------------------------

    conflict_errors = (
        validate_conflict_facts(
            result,
            source_map,
        )
    )

    if conflict_errors:

        hard_rule_flags.extend(
            [
                "CONFLICT_GROUNDING_FAIL: "
                + err
                for err in conflict_errors
            ]
        )

        if result.has_true_conflict:

            # 模型声称冲突但无法 Ground，
            # 不允许保留 true conflict
            result.has_true_conflict = False

            result.conflict_facts = []

            # 如果原结构纯粹依赖 conflict，
            # 不自动猜新的结构，
            # 先降为 insufficient_evidence
            if (
                result.knowledge_structure
                == "conflict_temporal"
                and
                result.temporal_dependency
                != "material"
            ):

                result.knowledge_structure = (
                    "insufficient_evidence"
                )

            result.generation_ready = False

            result.recommended_generation_mode = (
                "do_not_generate"
            )

    # --------------------------------------------------------
    # 5. Cause vs conflict 自洽检查
    # --------------------------------------------------------

    consistency_errors = (
        validate_cause_conflict_consistency(
            result
        )
    )

    if consistency_errors:

        hard_rule_flags.extend(
            [
                "CAUSE_CONFLICT_INCONSISTENCY: "
                + err
                for err in consistency_errors
            ]
        )

        result.generation_ready = False

        result.recommended_generation_mode = (
            "do_not_generate"
        )

    # --------------------------------------------------------
    # 6. conflict_temporal 必须有依据
    # --------------------------------------------------------

    if (
        result.knowledge_structure
        == "conflict_temporal"
        and
        not result.has_true_conflict
        and
        result.temporal_dependency
        != "material"
    ):

        result.knowledge_structure = (
            "insufficient_evidence"
        )

        result.generation_ready = False

        result.recommended_generation_mode = (
            "do_not_generate"
        )

        hard_rule_flags.append(
            "CONFLICT_TEMPORAL_WITHOUT_CONFLICT_OR_MATERIAL_TEMPORAL"
        )

    # --------------------------------------------------------
    # 7. insufficient_evidence 永远不生成
    # --------------------------------------------------------

    if (
        result.knowledge_structure
        == "insufficient_evidence"
    ):

        result.generation_ready = False

        result.recommended_generation_mode = (
            "do_not_generate"
        )

    # --------------------------------------------------------
    # 8. conflict_temporal 永远不生成
    # --------------------------------------------------------

    if (
        result.knowledge_structure
        == "conflict_temporal"
    ):

        result.generation_ready = False

        result.recommended_generation_mode = (
            "do_not_generate"
        )

    # --------------------------------------------------------
    # 9. material temporal hard block
    # --------------------------------------------------------

    if (
        result.temporal_dependency
        == "material"
        and
        not result.current_validity_resolved
    ):

        result.generation_ready = False

        result.recommended_generation_mode = (
            "do_not_generate"
        )

        hard_rule_flags.append(
            "MATERIAL_TEMPORAL_UNRESOLVED"
        )

    # --------------------------------------------------------
    # 10. Generation Mode 与结构对齐
    # --------------------------------------------------------

    expected_modes = {
        "direct_synthesis":
            "single_answer",

        "structured_howto":
            "howto_sections",

        "multiple_causes":
            "cause_items",

        "scenario_branches":
            "scenario_sections",

        "troubleshooting":
            "troubleshooting_steps",
    }

    if result.generation_ready:

        expected_mode = expected_modes.get(
            result.knowledge_structure
        )

        if expected_mode:

            result.recommended_generation_mode = (
                expected_mode
            )

    # --------------------------------------------------------
    # 11. Risk flags
    # --------------------------------------------------------

    for flag in hard_rule_flags:

        if flag not in result.risk_flags:

            result.risk_flags.append(
                flag
            )

    return result, hard_rule_flags


# ============================================================
# LLM classify
# ============================================================

def classify_cluster(
    cluster_row,
    source_rows,
):

    client = build_client()

    cluster_id = clean_text(
        cluster_row.get(
            "cluster_id"
        )
    )

    prompt = build_prompt(
        cluster_row,
        source_rows,
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

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
                            "role":
                                "system",
                            "content":
                                SYSTEM_PROMPT,
                        },
                        {
                            "role":
                                "user",
                            "content":
                                prompt,
                        },
                    ],
                    response_format={
                        "type":
                            "json_object"
                    },
                    temperature=0,
                    extra_body={
                        "enable_thinking":
                            False
                    },
                )
            )

            elapsed = (
                time.time()
                - started
            )

            data = json.loads(
                response
                .choices[0]
                .message
                .content
            )

            result = (
                StructureClassification
                .model_validate(
                    data
                )
            )

            # =================================================
            # Hard Rules
            # =================================================

            result, hard_rule_flags = (
                apply_hard_rules(
                    result,
                    source_rows,
                )
            )

            return {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    result.canonical_question,

                "source_count":
                    len(source_rows),

                "knowledge_structure":
                    result.knowledge_structure,

                "structure_confidence":
                    result.confidence,

                "independent_cause_count":
                    result.independent_cause_count,

                "scenario_count":
                    result.scenario_count,

                "has_true_conflict":
                    result.has_true_conflict,

                "conflict_fact_count":
                    len(
                        result.conflict_facts
                    ),

                "conflict_facts":
                    [
                        item.model_dump()
                        for item
                        in result.conflict_facts
                    ],

                "temporal_dependency":
                    result.temporal_dependency,

                "current_validity_resolved":
                    result.current_validity_resolved,

                "generation_ready":
                    result.generation_ready,

                "recommended_generation_mode":
                    result.recommended_generation_mode,

                "reason":
                    result.reason,

                "risk_flags":
                    result.risk_flags,

                "hard_rule_flag_count":
                    len(
                        hard_rule_flags
                    ),

                "hard_rule_flags":
                    hard_rule_flags,

                "evidence_roles":
                    [
                        item.model_dump()
                        for item
                        in result.evidence_roles
                    ],

                "request_seconds":
                    elapsed,
            }

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                time.sleep(
                    2 ** attempt
                )

    raise last_error


# ============================================================
# Excel Export
# ============================================================

def export_excel():

    objects = []

    if not OUTPUT_JSONL.exists():
        return

    with OUTPUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                objects.append(
                    json.loads(
                        line
                    )
                )

            except Exception:
                pass

    cluster_rows = []
    role_rows = []
    conflict_rows = []
    hard_rule_rows = []

    for obj in objects:

        cluster_rows.append(
            {
                "cluster_id":
                    obj[
                        "cluster_id"
                    ],

                "canonical_question":
                    obj[
                        "canonical_question"
                    ],

                "source_count":
                    obj[
                        "source_count"
                    ],

                "knowledge_structure":
                    obj[
                        "knowledge_structure"
                    ],

                "structure_confidence":
                    obj[
                        "structure_confidence"
                    ],

                "independent_cause_count":
                    obj[
                        "independent_cause_count"
                    ],

                "scenario_count":
                    obj[
                        "scenario_count"
                    ],

                "has_true_conflict":
                    obj[
                        "has_true_conflict"
                    ],

                "conflict_fact_count":
                    obj[
                        "conflict_fact_count"
                    ],

                "temporal_dependency":
                    obj[
                        "temporal_dependency"
                    ],

                "current_validity_resolved":
                    obj[
                        "current_validity_resolved"
                    ],

                "generation_ready":
                    obj[
                        "generation_ready"
                    ],

                "recommended_generation_mode":
                    obj[
                        "recommended_generation_mode"
                    ],

                "hard_rule_flag_count":
                    obj[
                        "hard_rule_flag_count"
                    ],

                "reason":
                    obj[
                        "reason"
                    ],

                "risk_flags":
                    " | ".join(
                        obj.get(
                            "risk_flags",
                            []
                        )
                    ),

                "hard_rule_flags":
                    " | ".join(
                        obj.get(
                            "hard_rule_flags",
                            []
                        )
                    ),
            }
        )

        for item in obj.get(
            "evidence_roles",
            []
        ):

            role_rows.append(
                {
                    "cluster_id":
                        obj[
                            "cluster_id"
                        ],

                    "issue_key":
                        item[
                            "issue_key"
                        ],

                    "role":
                        item[
                            "role"
                        ],

                    "scenario_label":
                        item.get(
                            "scenario_label"
                        ),

                    "scenario_evidence":
                        item.get(
                            "scenario_evidence"
                        ),

                    "reason":
                        item[
                            "reason"
                        ],
                }
            )

        for idx, item in enumerate(
            obj.get(
                "conflict_facts",
                []
            ),
            start=1,
        ):

            conflict_rows.append(
                {
                    "cluster_id":
                        obj[
                            "cluster_id"
                        ],

                    "conflict_no":
                        idx,

                    "fact_dimension":
                        item[
                            "fact_dimension"
                        ],

                    "statement_a":
                        item[
                            "statement_a"
                        ],

                    "statement_b":
                        item[
                            "statement_b"
                        ],

                    "issue_keys_a":
                        " | ".join(
                            item.get(
                                "issue_keys_a",
                                []
                            )
                        ),

                    "issue_keys_b":
                        " | ".join(
                            item.get(
                                "issue_keys_b",
                                []
                            )
                        ),

                    "same_condition":
                        item[
                            "same_condition"
                        ],

                    "mutually_exclusive":
                        item[
                            "mutually_exclusive"
                        ],

                    "evidence_reason":
                        item[
                            "evidence_reason"
                        ],
                }
            )

        for flag in obj.get(
            "hard_rule_flags",
            []
        ):

            hard_rule_rows.append(
                {
                    "cluster_id":
                        obj[
                            "cluster_id"
                        ],

                    "hard_rule_flag":
                        flag,
                }
            )

    clusters_df = pd.DataFrame(
        cluster_rows
    )

    roles_df = pd.DataFrame(
        role_rows
    )

    conflicts_df = pd.DataFrame(
        conflict_rows
    )

    hard_rules_df = pd.DataFrame(
        hard_rule_rows
    )

    structure_stats = (
        clusters_df[
            "knowledge_structure"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "knowledge_structure"
        )
        .reset_index(
            name="count"
        )
    )

    temporal_stats = (
        clusters_df[
            "temporal_dependency"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "temporal_dependency"
        )
        .reset_index(
            name="count"
        )
    )

    generation_ready_df = (
        clusters_df[
            clusters_df[
                "generation_ready"
            ]
            == True
        ]
        .copy()
    )

    blocked_df = (
        clusters_df[
            clusters_df[
                "generation_ready"
            ]
            != True
        ]
        .copy()
    )

    hard_rule_affected_df = (
        clusters_df[
            clusters_df[
                "hard_rule_flag_count"
            ]
            > 0
        ]
        .copy()
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "cluster_count",
                "value":
                    len(
                        clusters_df
                    ),
            },
            {
                "metric":
                    "generation_ready_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "generation_ready"
                            ]
                            == True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "blocked_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "generation_ready"
                            ]
                            != True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "true_conflict_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "has_true_conflict"
                            ]
                            == True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "material_temporal_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "temporal_dependency"
                            ]
                            == "material"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "hard_rule_affected_cluster_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "hard_rule_flag_count"
                            ]
                            > 0
                        ).sum()
                    ),
            },
            {
                "metric":
                    "conflict_fact_count",
                "value":
                    int(
                        clusters_df[
                            "conflict_fact_count"
                        ].sum()
                    ),
            },
        ]
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        structure_stats.to_excel(
            writer,
            sheet_name="structure_stats",
            index=False,
        )

        temporal_stats.to_excel(
            writer,
            sheet_name="temporal_stats",
            index=False,
        )

        clusters_df.to_excel(
            writer,
            sheet_name="all_clusters",
            index=False,
        )

        roles_df.to_excel(
            writer,
            sheet_name="evidence_roles",
            index=False,
        )

        conflicts_df.to_excel(
            writer,
            sheet_name="conflict_facts",
            index=False,
        )

        hard_rules_df.to_excel(
            writer,
            sheet_name="hard_rule_flags",
            index=False,
        )

        generation_ready_df.to_excel(
            writer,
            sheet_name="generation_ready",
            index=False,
        )

        blocked_df.to_excel(
            writer,
            sheet_name="blocked",
            index=False,
        )

        hard_rule_affected_df.to_excel(
            writer,
            sheet_name="hard_rule_affected",
            index=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 11.3 v3 - "
        "Mixed Knowledge Structure Gate"
    )
    print("=" * 70)

    print(
        f"Model: {MODEL}"
    )

    mixed_df = pd.read_excel(
        MIXED_AUDIT_FILE,
        sheet_name="all_clusters",
    )

    grounding_df = pd.read_excel(
        GROUNDING_FILE,
        sheet_name="all_sources",
    )

    # --------------------------------------------------------
    # 只使用 Grounding Gate 通过 evidence
    # --------------------------------------------------------

    usable_df = (
        grounding_df[
            grounding_df[
                "usable_for_kb"
            ]
            == True
        ]
        .copy()
    )

    print(
        f"Mixed cluster count: "
        f"{len(mixed_df)}"
    )

    print(
        f"Usable grounded sources: "
        f"{len(usable_df)}"
    )

    processed = load_processed()

    tasks = []

    for _, cluster_row in (
        mixed_df.iterrows()
    ):

        cluster_id = clean_text(
            cluster_row.get(
                "cluster_id"
            )
        )

        if cluster_id in processed:
            continue

        source_rows = (
            usable_df[
                usable_df[
                    "cluster_id"
                ]
                .astype(str)
                == cluster_id
            ]
            .to_dict(
                orient="records"
            )
        )

        if not source_rows:

            print(
                f"[SKIP] {cluster_id}: "
                f"NO_VALID_SOURCE"
            )

            continue

        tasks.append(
            (
                cluster_row.to_dict(),
                source_rows,
            )
        )

    print(
        f"本次待审核: "
        f"{len(tasks)}"
    )

    total = len(tasks)
    completed = 0

    started_at = time.time()
    lock = threading.Lock()

    if tasks:

        with OUTPUT_JSONL.open(
            "a",
            encoding="utf-8",
        ) as fout:

            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                future_map = {

                    executor.submit(
                        classify_cluster,
                        cluster_row,
                        source_rows,
                    ):
                    clean_text(
                        cluster_row.get(
                            "cluster_id"
                        )
                    )

                    for (
                        cluster_row,
                        source_rows,
                    )
                    in tasks
                }

                for future in as_completed(
                    future_map
                ):

                    cluster_id = (
                        future_map[
                            future
                        ]
                    )

                    try:

                        result = (
                            future.result()
                        )

                    except Exception as exc:

                        print(
                            f"[FAILED] "
                            f"{cluster_id}: "
                            f"{exc}"
                        )

                        continue

                    with lock:

                        fout.write(
                            json.dumps(
                                result,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        fout.flush()

                        completed += 1

                    elapsed = (
                        time.time()
                        - started_at
                    )

                    avg = (
                        elapsed
                        / completed
                    )

                    eta = (
                        avg
                        * (
                            total
                            - completed
                        )
                    )

                    print(
                        f"[{completed}/{total}] "
                        f"| "
                        f"{cluster_id} "
                        f"| "
                        f"{result['knowledge_structure']} "
                        f"| ready="
                        f"{result['generation_ready']} "
                        f"| hard_flags="
                        f"{result['hard_rule_flag_count']} "
                        f"| ETA "
                        f"{eta / 60:.1f} min"
                    )

    export_excel()

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)

    print(
        f"Excel: "
        f"{OUTPUT_XLSX}"
    )


if __name__ == "__main__":
    main()