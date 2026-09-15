#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 7 - 使用 LLM 从 candidate conversation blocks 中抽取独立 CRM issue

输入：
    output/issue_candidates.jsonl

输出：
    output/extracted_issues.jsonl
    output/extracted_issues.xlsx

当前阶段：
    默认只处理前 50 个 candidate blocks，用于验证质量。

重要：
    一个 candidate block 可以抽取：
        0 个 issue
        1 个 issue
        多个 issue

不要假设：
    1 candidate block == 1 knowledge item
"""

import json
import os
from pathlib import Path
from typing import List, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

import time
import random
import threading

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
# ============================================================
# 并发设置
# ============================================================
MAX_WORKERS = 5

MAX_RETRIES = 4

REQUEST_TIMEOUT = 120

# ============================================================
# 路径
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT_DIR
    / "output"
    / "issue_candidates.jsonl"
)

OUTPUT_JSONL = (
    ROOT_DIR
    / "output"
    / "extracted_issues.jsonl"
)

OUTPUT_XLSX = (
    ROOT_DIR
    / "output"
    / "extracted_issues.xlsx"
)


# ============================================================
# 当前测试参数
# ============================================================

MODEL = "qwen3.5-flash"

# Step 7 第一轮只跑 50 个
LIMIT = None

# 如果之前中断，是否跳过已经成功处理的 candidate
RESUME = True


# ============================================================
# Pydantic Schema
# ============================================================

class ExtractedIssue(BaseModel):

    question: str = Field(
        description=(
            "该问题的规范、独立、可检索的问题表述。"
            "不要保留客户姓名、门店名等无关来源信息。"
        )
    )

    question_normalized: str = Field(
        description=(
            "规范化后的独立问题句，用于后续 embedding 和相似问题聚类。"
            "必须写成一个可以脱离聊天上下文独立理解的问题。"
            "必须使用问句形式。"
            "去除客户姓名、门店名、车辆、手机号、人员姓名等来源噪音，"
            "但不能改变原问题含义。"
        )
    )

    description: str = Field(
        description=(
            "根据聊天上下文概括问题背景和现象。"
            "不得编造聊天中没有的信息。"
        )
    )

    answer: str = Field(
        description=(
            "客服/技术人员在聊天中实际给出的回答。"
            "仅根据原始消息总结。"
        )
    )

    solution: str = Field(
        description=(
            "从回答中提炼出的解决方式、操作步骤或规则。"
            "如果对话中没有明确方案，应说明未给出明确解决方案。"
        )
    )

    problem_type: Literal[
        "how_to",
        "bug",
        "configuration",
        "information_query",
        "feature_request",
        "data_issue",
        "permission_issue",
        "account_issue",
        "unknown",
    ]

    crm_module: str = Field(
        description=(
            "CRM 所属业务模块。"
            "只根据对话判断；无法确定写 unknown。"
        )
    )

    crm_module: Literal[
        "客户管理",
        "商机管理",
        "车辆管理",
        "工单/维修",
        "预约",
        "回访",
        "企业微信",
        "短信",
        "数据同步",
        "标签",
        "权限",
        "账号",
        "报表",
        "系统配置",
        "移动端",
        "PC端",
        "插件",
        "其他",
        "unknown",
    ] = Field(
        description=(
            "CRM 一级业务模块。"
            "必须从固定枚举中选择，禁止自行创造新的一级模块名称。"
            "如果无法可靠判断，选择 unknown。"
        )
    )
    crm_feature: str = Field(
        description=(
            "CRM 具体功能点，例如：流失客户、短信发送、车辆档案、"
            "保养提醒、客户跟进、回访工单等。"
            "只根据聊天内容判断；无法确定时写 unknown。"
        )
    )

    resolution: Literal[
        "resolved",
        "unresolved",
        "partial",
        "feature_request",
        "bug_confirmed",
        "unknown",
    ]

    temporal_status: Literal[
        "stable",
        "temporary",
        "future_plan",
        "historical",
        "unknown",
    ] = Field(
        description=(
            "该回答的时间稳定性。"
            "stable 表示长期适用的产品规则、操作方式或配置方法；"
            "temporary 表示处理中、临时方案、当前状态、预计上线时间等；"
            "future_plan 表示未来计划、待开发、计划上线功能；"
            "historical 表示只描述过去发生过的状态；"
            "无法判断时选择 unknown。"
        )
    )

    knowledge_value: Literal[
        "high",
        "medium",
        "low",
    ] = Field(
        description=(
            "该 issue 对最终 CRM RAG 知识库的直接价值。"
            "有明确可复用答案通常为 high；"
            "部分解决或需要结合后续信息为 medium；"
            "只有问题、需求、排查中通常为 low。"
        )
    )

    needs_followup: bool = Field(
        description=(
            "当前聊天信息是否不足，需要结合后续聊天、正式文档"
            "或其他来源才能形成可靠知识。"
        )
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "该 issue 作为可复用 CRM 知识的可靠程度，"
            "而不是判断模型是否正确理解聊天的置信度。"
            "完整且有明确问题和明确回答时可以较高；"
            "没有明确答案、只有排查中、只有需求描述、"
            "上下文不足时必须明显降低。"
        )
    )

    source_message_indexes: List[int] = Field(
        description=(
            "只填写真正支持这个 issue 的 "
            "source_message_index。"
            "不能把整个 candidate 的消息全部无差别复制。"
        )
    )



class ExtractionResult(BaseModel):

    useful_for_knowledge_base: bool = Field(
        description=(
            "该 candidate block 是否包含值得进入 CRM "
            "知识抽取流程的业务问题。"
        )
    )

    issues: List[ExtractedIssue]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在处理 CRM 客服聊天记录。

你的任务不是总结整个聊天，而是从聊天中识别一个或多个
彼此独立的 CRM 业务问题，并抽取结构化知识单元。

非常重要：

1. 一个 conversation block 可能包含 0、1 或多个独立问题。

2. 如果聊天中明显切换了问题，即使没有很长时间间隔，
   也必须拆成多个 issue。

例如：

客户：手机端怎么删除待跟进？
客服：手机端暂时不支持。

客服：另外短信功能你们店还没有开通。

这里是两个问题，不允许混为一个。

3. 普通寒暄、感谢、拉群、发送名片、单纯培训通知、
   账号开通确认等，如果没有形成可复用 CRM 知识，
   可以不抽取。

4. question 必须写成独立问题。
   不要写：
   “客户这里为什么不对”
   应该写：
   “车辆保养记录中的预计保养里程为什么出现异常？”

5. question_normalized 更强调去除具体客户、门店、车辆、
   手机号、人员姓名等来源信息。

6. answer 和 solution 只能根据聊天里的实际信息整理。
   不得使用你自己的产品知识补充。

7. 如果客服只是说：
   “我看看”
   “技术处理中”
   “后面修复”
   而没有真正回答问题，
   resolution 应为 unresolved 或 partial。

8. 如果回答明确表示：
   当前产品不支持，并计划以后增加，
   可以判断为 feature_request 或 partial，
   不要虚构实现方式。

9. 一个 issue 的 source_message_indexes
   必须只包含真正支持该问题和答案的消息。

10. CRM 模块不能确定时写 unknown。
    不要为了填字段而猜测。
11. question_normalized 必须始终写成问句。

正确：
“回访工单为什么缺少SA姓名？”

不要写：
“回访工单缺少SA姓名，导致无法筛选。”

question_normalized 的目标是后续直接进行 embedding 和问题聚类。


12. crm_module 必须严格使用给定枚举。

禁止自己创造类似：

“客户运营与消息触达”
“账号与订阅管理”
“CRM产品规划”

这样的新一级模块。

如果现有一级模块不能可靠覆盖，选择：
“其他” 或 “unknown”。

更具体的信息应该放到 crm_feature。


13. confidence 表示：
“这条内容作为可复用 CRM 知识的可靠程度”。

建议尺度：

0.90 - 1.00
问题明确，回答明确，操作/规则明确，证据充分。

0.75 - 0.89
问题和回答基本明确，但仍有部分上下文缺失。

0.55 - 0.74
只有部分回答、临时方案、处理中、待进一步确认。

0.30 - 0.54
只有问题或需求，没有有效答案。

0.00 - 0.29
信息非常模糊，不适合作为知识。

例如：

客户问：
“明年有什么产品规划？”

聊天中无人回答。

则即使你非常确定“聊天中没有答案”，
该知识条目的 confidence 仍然应该较低，
通常不应高于 0.50。


14. 对 unresolved 内容要区分“问题记录”和“知识”。

可以保留 unresolved issue，
因为后续可能用于：
- 发现产品问题
- 与后续回答进行合并
- 识别旧问题

但是不能把“无人回答”包装成一个高置信度知识答案。


15. 拆 issue 时按“知识主题”拆，不要过度拆分。

如果多个现象实际上属于同一产品规则，可以作为一个 issue。

例如：

“按时间已到保养周期但按里程未到”
和
“未保养离店后是否继续提醒”

如果聊天明确在讨论同一个“保养到期提醒判断逻辑”，
可以保留为一个 issue。

但“保养提醒”和“短信开通”必须拆开。

16. 判断回答的时间稳定性。

stable：
长期有效的产品规则、功能说明、操作步骤、权限规则。

例如：
“战败原因需要管理员权限才能配置。”

temporary：
当前处理中、临时处理方式、某一天预计上线、
“今天修复”“明天上线”“目前先这样处理”。

例如：
“预计今天修复，明天上线。”

future_plan：
明确表示未来计划增加、开发或优化的功能。

例如：
“这个功能后续版本会增加。”

historical：
只是描述过去发生过的状态，
对当前产品已经不一定有效。

不要把 temporary 或 future_plan
当作长期稳定知识直接进入最终 RAG。

17. needs_followup 判断标准：

以下情况通常应为 true：

- resolution = unresolved
- 正在排查
- 等待技术处理
- 预计某天修复或上线
- 当前仅有临时方案
- future_plan
- 回答依赖后续版本
- 回答可能很快过期

只有已经明确、稳定、可复用的产品规则或操作方法，
才通常设置为 false。

18. resolution 判断时，必须判断“原始问题本身是否真正解决”。

不要因为客服给出了一个回答、解释或替代方案，
就自动判断为 resolved。

例如：

客户问：
“系统能不能自动把日报发送到第三方创建的群？”

客服回答：
“目前不支持，可以先人工转发。”

这种情况原始需求并没有真正解决。

应优先判断为：
resolution = unresolved 或 partial

如果后续仍需要产品、技术或版本处理：
needs_followup = true

只有原始问题本身已经通过明确操作、
配置方法、产品规则或实际修复得到解决，
才能判断：
resolution = resolved。


19. 必须区分“正式解决方案”和“临时替代方案”。

以下情况通常属于 workaround / 临时方案：

- 手工处理
- 人工转发
- 暂时换一种操作
- 先绕过问题
- 等技术处理
- 等版本上线
- 后续优化
- 当前先这样处理

如果 solution 只是以上临时替代方式，
不能仅因为存在 solution 就判断为 resolved。

通常应考虑：
resolution = partial 或 unresolved

如果方案只是暂时有效：
temporal_status = temporary

如果后续仍需要处理：
needs_followup = true

knowledge_value 也不应仅因为存在临时方案就判断为 high。


20. knowledge_value = high 必须满足较严格条件。

high 表示：
当前聊天已经形成明确、可靠、可复用的 CRM 知识，
可以较安全地作为后续 RAG / Agent 的知识来源。

以下情况通常不应直接判断为 high：

- 原始问题仍未解决
- 只有临时 workaround
- 正在排查
- 等待技术处理
- 等待版本上线
- 当前产品明确不支持
- 只是提出需求
- 回答明显依赖当前时间或当前版本
- 回答仍可能发生变化

这些情况通常应判断为：
knowledge_value = medium 或 low。


21. 拆 issue 时，既要避免过度拆分，也不要把独立知识强行合并。

如果同一个 conversation block 中同时包含多个
可以独立理解、独立检索、独立回答的问题，
应拆成多个 issue。

尤其是以下类型如果聊天分别给出了明确答案，
通常可以独立成 issue：

- 为什么出现某个现象
- 系统采用什么判断或计算规则
- 某个操作会产生什么结果
- 某个功能在哪里查看
- 某个功能如何操作

例如：

“为什么车辆没有保养提醒？”

和：

“SA 在工单录入最新里程后，
系统是否会重新计算保养提醒？”

如果聊天分别明确回答了这两个问题，
应允许拆成两个独立 issue。

但如果多个现象明显是在解释同一条产品规则，
仍然可以合并，
不要机械地一问拆一个 issue。

目标：
得到以后可以用于 CRM RAG / Agent 的高质量知识，
而不是简单聊天摘要。
"""


# ============================================================
# Client
# ============================================================

def build_client():

    # if not os.environ.get("OPENAI_API_KEY"):
    #     raise RuntimeError(
    #         "没有检测到 OPENAI_API_KEY。\n"
    #         "请先执行：\n"
    #         'export OPENAI_API_KEY="你的API_KEY"'
    #     )

    return OpenAI(
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


# ============================================================
# 加载 candidate
# ============================================================

def load_candidates():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到：{INPUT_FILE}"
        )

    rows = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


# ============================================================
# 已处理记录
# ============================================================

def load_processed_candidate_ids():

    ids = set()

    if (
        not RESUME
        or not OUTPUT_JSONL.exists()
    ):
        return ids

    with OUTPUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

                candidate_id = (
                    item.get(
                        "source_candidate_id"
                    )
                )

                if candidate_id:
                    ids.add(candidate_id)

            except Exception:
                pass

    return ids


# ============================================================
# Candidate 转 Prompt
# ============================================================

def build_candidate_text(candidate):

    lines = []

    lines.append(
        f"candidate_id: "
        f"{candidate.get('issue_id')}"
    )

    lines.append(
        f"source_filename: "
        f"{candidate.get('source_filename')}"
    )

    lines.append("")

    lines.append("聊天消息：")

    for msg in candidate.get(
        "source_messages",
        []
    ):

        lines.append(
            "[source_message_index="
            f"{msg.get('source_message_index')}] "
            f"[{msg.get('timestamp')}] "
            f"{msg.get('speaker')} "
            f"({msg.get('speaker_role')}): "
            f"{msg.get('content')}"
        )

    return "\n".join(lines)


# ============================================================
# 调用 LLM
# ============================================================

def extract_candidate(
    client,
    candidate,
):

    candidate_text = (
        build_candidate_text(candidate)
    )

    schema = (
        ExtractionResult
        .model_json_schema()
    )

    schema_text = json.dumps(
        schema,
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = f"""
请严格根据以下 JSON Schema 返回结果。

只能返回合法 JSON。
不要返回 Markdown。
不要使用 ```json 代码块。
不要输出解释文字。

JSON Schema：

{schema_text}

下面是需要分析的 CRM 聊天记录：

{candidate_text}
"""

    response = (
        client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            temperature=0.1,

            # 如果 CloseAI/Qwen 当前线路支持，
            # 可以强制 JSON Object 输出
            response_format={
                "type": "json_object"
            },

            extra_body={
                "enable_thinking": False
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:
        raise RuntimeError(
            "模型返回内容为空"
        )

    # 防止部分模型仍然套 markdown code fence
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.I,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

    try:
        parsed = json.loads(content)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "模型返回的不是合法 JSON：\n"
            f"{content[:1500]}"
        ) from exc

    try:
        result = (
            ExtractionResult
            .model_validate(parsed)
        )

    except Exception as exc:

        raise RuntimeError(
            "JSON 与 ExtractionResult "
            "Schema 不匹配：\n"
            f"{content[:1500]}"
        ) from exc

    return result

def extract_candidate_with_retry(
    client,
    candidate,
):

    candidate_id = candidate.get(
        "issue_id"
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            start_time = time.time()

            result = extract_candidate(
                client,
                candidate,
            )

            elapsed = (
                time.time() - start_time
            )

            return {
                "success": True,
                "candidate": candidate,
                "result": result,
                "elapsed": elapsed,
                "attempt": attempt,
                "error": None,
            }

        except Exception as exc:

            if attempt >= MAX_RETRIES:

                return {
                    "success": False,
                    "candidate": candidate,
                    "result": None,
                    "elapsed": None,
                    "attempt": attempt,
                    "error": str(exc),
                }

            sleep_seconds = (
                2 ** attempt
                + random.uniform(0, 1)
            )

            print(
                f"[RETRY] {candidate_id} "
                f"第 {attempt} 次失败："
                f"{exc}"
            )

            print(
                f"        {sleep_seconds:.1f}s "
                f"后重试"
            )

            time.sleep(
                sleep_seconds
            )
# ============================================================
# 保存单条 candidate 结果
# ============================================================

def append_candidate_result(
    candidate,
    result,
):

    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_id = candidate["issue_id"]

    rows = []

    # candidate 没有知识价值时，
    # 也写一条记录，方便 resume
    if not result.issues:

        rows.append({
            "source_candidate_id":
                candidate_id,

            "document_id":
                candidate.get(
                    "document_id"
                ),

            "source_filename":
                candidate.get(
                    "source_filename"
                ),

            "useful_for_knowledge_base":
                result.useful_for_knowledge_base,

            "issue_local_index": None,

            "issue_id": None,

            "question": None,

            "question_normalized": None,

            "description": None,

            "answer": None,

            "solution": None,

            "problem_type": None,

            "crm_module": None,

            "crm_feature": None,

            "resolution": None,

            "temporal_status": None,

            "confidence": None,

            "knowledge_value": None,

            "needs_followup": None,

            "source_message_indexes": [],
        })

    else:

        for local_index, issue in enumerate(
            result.issues,
            start=1,
        ):

            issue_id = (
                f"{candidate_id}"
                f"-{local_index:02d}"
            )

            rows.append({
                "source_candidate_id":
                    candidate_id,

                "document_id":
                    candidate.get(
                        "document_id"
                    ),

                "source_filename":
                    candidate.get(
                        "source_filename"
                    ),

                "useful_for_knowledge_base":
                    result.useful_for_knowledge_base,

                "issue_local_index":
                    local_index,

                "issue_id":
                    issue_id,

                "question":
                    issue.question,

                "question_normalized":
                    issue.question_normalized,

                "description":
                    issue.description,

                "answer":
                    issue.answer,

                "solution":
                    issue.solution,

                "temporal_status":
                    issue.temporal_status,

                "problem_type":
                    issue.problem_type,

                "crm_module":
                    issue.crm_module,

                "crm_feature":
                    issue.crm_feature,

                "resolution":
                    issue.resolution,

                "confidence":
                    issue.confidence,

                "source_message_indexes":
                    issue.source_message_indexes,
                "knowledge_value":
                    issue.knowledge_value,

                "needs_followup":
                    issue.needs_followup,
            })

    with OUTPUT_JSONL.open(
        "a",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# JSONL -> Excel
# ============================================================

def build_excel():

    if not OUTPUT_JSONL.exists():
        return

    rows = []

    with OUTPUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            item[
                "source_message_indexes"
            ] = ",".join(
                str(x)
                for x in item.get(
                    "source_message_indexes",
                    []
                )
            )

            rows.append(item)

    if not rows:
        return

    df = pd.DataFrame(rows)

    # 真正抽出了 issue 的记录
    issue_df = df[
        df["issue_id"].notna()
    ].copy()

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    candidate_count = (
        df["source_candidate_id"]
        .nunique()
    )

    extracted_issue_count = len(
        issue_df
    )

    multi_issue_candidates = 0

    if len(issue_df):

        per_candidate = (
            issue_df.groupby(
                "source_candidate_id"
            )
            .size()
        )

        multi_issue_candidates = int(
            (per_candidate > 1).sum()
        )

    summary_rows = [
        {
            "metric":
                "processed_candidate_count",
            "value":
                candidate_count,
        },
        {
            "metric":
                "extracted_issue_count",
            "value":
                extracted_issue_count,
        },
        {
            "metric":
                "multi_issue_candidate_count",
            "value":
                multi_issue_candidates,
        },
    ]

    if len(issue_df):

        summary_rows.append({
            "metric":
                "avg_confidence",

            "value":
                round(
                    issue_df[
                        "confidence"
                    ].mean(),
                    4,
                ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:

        issue_df.to_excel(
            writer,
            sheet_name="extracted_issues",
            index=False,
        )

        # 前 100 条人工审核
        issue_df.head(100).to_excel(
            writer,
            sheet_name="review_100",
            index=False,
        )

        df[
            df["issue_id"].isna()
        ].to_excel(
            writer,
            sheet_name="discarded_candidates",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        workbook = writer.book

        for sheet_name in workbook.sheetnames:

            ws = workbook[sheet_name]

            ws.freeze_panes = "A2"

            if ws.max_row > 1:
                ws.auto_filter.ref = (
                    ws.dimensions
                )

        for sheet_name in [
            "extracted_issues",
            "review_100",
        ]:

            ws = workbook[sheet_name]

            widths = {
                "A": 24,
                "B": 18,
                "C": 35,
                "D": 18,
                "E": 12,
                "F": 24,
                "G": 55,
                "H": 55,
                "I": 70,
                "J": 70,
                "K": 70,
                "L": 20,
                "M": 22,
                "N": 28,
                "O": 20,
                "P": 14,
                "Q": 35,
            }

            for col, width in (
                widths.items()
            ):
                ws.column_dimensions[
                    col
                ].width = width


# ============================================================
# main
# ============================================================

def main():

    print("=" * 72)
    print(
        "Step 7 - LLM CRM Knowledge Extraction"
    )
    print("=" * 72)

    client = build_client()

    candidates = load_candidates()

    print(
        f"Candidate 总数："
        f"{len(candidates):,}"
    )

    processed = (
        load_processed_candidate_ids()
    )

    if processed:
        print(
            f"已有成功处理 Candidate："
            f"{len(processed):,}"
        )

    selected = []

    if LIMIT is None:
        remaining_limit = None
    else:
        remaining_limit = max(
            0,
            LIMIT - len(processed)
        )

    for candidate in candidates:

        candidate_id = candidate.get(
            "issue_id"
        )

        if candidate_id in processed:
            continue

        selected.append(candidate)

        if (
            remaining_limit is not None
            and len(selected) >= remaining_limit
        ):
            break

    print()
    print("=" * 60)
    print("Candidate 处理计划")
    print("=" * 60)

    print(f"全部 candidate: {len(candidates)}")
    print(f"已处理 candidate: {len(processed)}")
    print(f"本次待处理: {len(selected)}")

    if LIMIT is None:
        print("LIMIT: ALL")
    else:
        print(f"LIMIT: {LIMIT}")

    print(
        f"本次计划处理："
        f"{len(selected):,}"
    )

    if not selected:
        print("没有需要处理的数据。")
        build_excel()
        return

    success = 0
    failed = 0

    completed = 0

    total = len(selected)

    start_all = time.time()

    write_lock = threading.Lock()


    def process_one(candidate):

        # 每个线程单独创建 client，
        # 避免第三方代理在线程并发时出现共享连接问题
        thread_client = build_client()

        return extract_candidate_with_retry(
            thread_client,
            candidate,
        )


    print(
        f"\n开始并发处理："
        f"workers={MAX_WORKERS}"
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(
                process_one,
                candidate,
            ): candidate

            for candidate in selected
        }

        for future in as_completed(
            future_map
        ):

            completed += 1

            candidate = (
                future_map[future]
            )

            candidate_id = (
                candidate.get("issue_id")
            )

            try:

                data = future.result()

            except Exception as exc:

                failed += 1

                print(
                    f"\n[{completed}/{total}] "
                    f"{candidate_id}"
                )

                print(
                    f"  [ERROR] "
                    f"线程异常：{exc}"
                )

                continue

            if not data["success"]:

                failed += 1

                print(
                    f"\n[{completed}/{total}] "
                    f"{candidate_id}"
                )

                print(
                    f"  [ERROR] "
                    f"{data['error']}"
                )

                continue

            result = data["result"]

            # 写 JSONL 时加锁
            with write_lock:

                append_candidate_result(
                    candidate,
                    result,
                )

            success += 1

            elapsed_all = (
                time.time()
                - start_all
            )

            avg_seconds = (
                elapsed_all
                / completed
            )

            remaining = (
                total - completed
            )

            eta_seconds = (
                remaining
                * avg_seconds
            )

            print(
                f"\n[{completed}/{total}] {candidate_id}"
            )

            print(
                f"  useful={result.useful_for_knowledge_base}  "
                f"issues={len(result.issues)}  "
                f"request={data['elapsed']:.1f}s  "
                f"retry={data['attempt'] - 1}  "
                f"ETA≈{eta_seconds / 60:.1f} min"
            )

            for issue_index, issue in enumerate(
                result.issues,
                start=1,
            ):
                print(
                    f"  {issue_index}. {issue.question}"
                )

    build_excel()

    print("\n" + "=" * 72)

    print(
        f"成功：{success}"
    )

    print(
        f"失败：{failed}"
    )

    print(
        f"\nJSONL：{OUTPUT_JSONL}"
    )

    print(
        f"Excel：{OUTPUT_XLSX}"
    )

    print(
        "\n当前只跑测试批次。"
        "请先人工审核 review_100，"
        "不要直接把 LIMIT 改成全部 3606。"
    )


if __name__ == "__main__":
    main()