import os
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
from pydantic import BaseModel


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

MIXED_AUDIT_FILE = (
    OUTPUT_DIR
    / "mixed_cluster_audits_v2.xlsx"
)

CLUSTER_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class AlignmentResult(BaseModel):

    alignment: Literal[
        "aligned",
        "partially_aligned",
        "misaligned",
        "no_answer",
        "uncertain",
    ]

    confidence: float

    question_answer_consistent: bool

    solution_supported_by_answer: bool

    usable_for_kb: bool

    supported_answer: str | None = None

    supported_solution: str | None = None

    reason: str

    risk_flags: list[str]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
你正在执行 CRM RAG 知识库的 Source Q/A Alignment Gate。

输入是一条已经从客服聊天中抽取出的 issue。

本轮只判断：

question / question_normalized

是否真正被：

answer

回答。

solution 只能作为辅助检查对象。

非常重要：

solution 本身也是上游 LLM 抽取结果，
它不能用于证明 answer 正确。

============================================================
最高优先级原则
============================================================

Grounding 顺序必须是：

question
    ↓
answer
    ↓
solution

而不是：

question
    ↓
answer + solution


也就是说：

第一步必须独立判断：

answer 是否真正回答 question。

只有 Answer 本身成立以后，
才能继续判断 Solution 是否被 Answer 支持。


============================================================
禁止 Solution 救 Answer
============================================================

如果 question 问：

“如何开通短信功能？”

answer 只有：

“你们店还没有开通，可以开通一下。”

solution 却写：

“联系管理员开通。”

那么：

Answer 本身没有回答“如何开通”。

不能因为 Solution 补出了步骤，
就把它判 aligned。

应该：

alignment = no_answer

或在存在少量有效信息时：

partially_aligned

但 Solution 中新增的“联系管理员”
不能作为 supported_solution。


============================================================
另一个例子
============================================================

question：

“如何创建 CRM 账号？”

answer：

“提供员工手机号，客服创建账号。
账号默认是手机号，初始密码为6个1。”

solution：

“提供手机号；
访问 https://xxx；
必须使用谷歌浏览器；
首次登录修改密码。”

Answer 可以回答创建账号问题。

因此 Q/A 可以：

aligned

但是 Answer 没有提到：

- https://xxx
- 谷歌浏览器

所以：

solution_supported_by_answer = false

supported_solution 只能保留
Answer 明确支持的部分，

不能保留 URL 和浏览器要求。


============================================================
aligned
============================================================

只有 Answer 本身明确回答了 Question，
才可以判：

aligned


例如：

question：

“为什么已回访客户还在列表？”

answer：

“因为这是同一客户的另一个工单，
系统按工单维度展示。”

→ aligned


============================================================
partially_aligned
============================================================

Answer 中有明确有效信息回答 Question，

但：

- 回答不完整
- 混入其他内容
- 只覆盖问题的一部分
- 部分内容无法确定
- Solution 比 Answer 有扩张

可以：

partially_aligned


注意：

Solution 扩张本身不一定让 Q/A 变成 partially_aligned。

如果 Answer 已完整回答问题，
仍可以 alignment=aligned，

但必须：

solution_supported_by_answer=false

并在 supported_solution 中只保留
Answer 能支持的部分。


============================================================
no_answer
============================================================

Question 明确提出一个问题，

但 Answer 只有：

- 可以开通
- 看一下
- 稍等
- 已反馈
- 正在处理
- 需要确认
- @某人
- 后续联系
- 已经重置但仍失败

没有形成真正能回答该 Question 的知识。

→ no_answer


特别注意：

问题问：

“如何做 X？”

Answer 只有：

“可以做 X。”

这不等于回答了“如何做”。

通常应：

no_answer


问题问：

“为什么 X？”

Answer 只有：

“我们正在处理。”

也不是答案。

→ no_answer


============================================================
misaligned
============================================================

Answer 有实际内容，
但回答的是另一个问题。

例如：

question：

“如何修改里程？”

answer：

“短信套餐已经购买成功。”

→ misaligned


区别：

no_answer：
没有形成有效答案。

misaligned：
形成了答案，但回答的是别的问题。


============================================================
uncertain
============================================================

只有确实无法判断时使用。

不要滥用。


============================================================
question_answer_consistent
============================================================

只根据：

Question
vs
Answer

判断。

禁止使用 Solution 来把它改成 true。


============================================================
solution_supported_by_answer
============================================================

必须执行严格 entailment。

只有 Solution 中的事实和操作：

能够直接从 Answer 中找到依据，

才可以：

true


如果 Solution 出现 Answer 没有提到的：

- 菜单入口
- URL
- 浏览器要求
- 联系对象
- 时间
- 密码规则
- 操作步骤
- 配置项
- 产品规则

则：

solution_supported_by_answer = false


即使这些内容可能在现实中正确，

只要 Answer 没有支持，

本轮就是 false。


============================================================
supported_answer
============================================================

只能来自 Answer 原文。

允许：

- 删除无关内容
- 压缩措辞
- 保留核心事实

禁止：

- 使用 Solution 补 Answer
- 增加新事实
- 推断产品规则
- 推断操作步骤


aligned：

保留 Answer 中真正回答问题的内容。


partially_aligned：

只保留其中有效部分。


misaligned
no_answer
uncertain：

supported_answer = null


============================================================
supported_solution
============================================================

supported_solution 必须同时满足：

1. 属于当前 Question
2. 能由 Answer 直接支持


如果 Solution 部分支持、部分扩张：

不要因为部分扩张把全部丢掉。

只保留支持部分。


例如：

Answer：

“提供手机号，由客服创建账号。
初始密码是6个1。”

Solution：

“1. 提供手机号
2. 客服创建账号
3. 使用谷歌浏览器
4. 初始密码6个1”

supported_solution 应只保留：

“提供手机号，由客服创建账号；
初始密码为6个1。”

不能保留：

“使用谷歌浏览器”。


============================================================
usable_for_kb
============================================================

这里表示：

该 Source 是否可以作为后续 KB 的证据。

不是最终 Publishability。


通常：

aligned
+
有 supported_answer

→ true


partially_aligned
+
仍有明确 supported_answer

→ true


no_answer
misaligned
uncertain

→ false


============================================================
Resolution / Temporal
============================================================

不要因为以下字段自动否定 source：

resolution = unresolved
resolution = partial
temporal_status = temporary
knowledge_value = low


例如：

问题：

“为什么录音没有显示？”

Answer：

“当前后台语音上传失败。”

即使：

temporary + unresolved

Q/A 仍然可以 aligned。


发布时效风险以后再处理。


============================================================
How-to 问题特别规则
============================================================

如果 Question 是：

“如何……？”
“怎么……？”
“流程是什么？”
“在哪里操作？”

Answer 必须至少提供：

- 操作方法
- 操作入口
- 必要条件
- 联系谁处理
- 明确的流程事实

中的一种。


只有：

“可以”
“支持”
“还没开”
“需要开一下”

不构成 How-to Answer。


============================================================
Why 问题特别规则
============================================================

如果 Question 是：

“为什么……？”
“是什么原因？”

Answer 必须至少提供：

- 原因
- 机制
- 判断逻辑
- 明确故障点

中的一种。


只有：

“刷新试试”
“重启试试”

属于 workaround，

不自动等于回答了“为什么”。


但是：

如果用户问题可以自然理解为：

“出现这个问题怎么办？”

则具体 troubleshooting 操作
可以视为有效回答。

请结合 question 原文判断，
不要机械按照“为什么”三个字。


============================================================
严格禁止
============================================================

禁止：

- 用 Solution 补 Answer
- 用常识补 Source
- 根据 CRM 经验推测
- 根据其他 Issue 推测
- 自动补菜单路径
- 自动补 URL
- 自动补时间
- 自动补密码
- 自动补联系人
- 自动补产品规则


============================================================
输出
============================================================

只输出合法 JSON。

格式：

{
  "alignment": "aligned | partially_aligned | misaligned | no_answer | uncertain",
  "confidence": 0.0,
  "question_answer_consistent": true,
  "solution_supported_by_answer": true,
  "usable_for_kb": true,
  "supported_answer": "只能来自 Answer 本身",
  "supported_solution": "只能保留 Answer 能直接支持的 Solution 内容",
  "reason": "简短说明",
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
# 工具
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

                issue_key = obj.get(
                    "issue_key"
                )

                if issue_key:

                    processed.add(
                        issue_key
                    )

            except Exception:
                pass

    return processed


# ============================================================
# Prompt
# ============================================================

def build_prompt(row):

    return f"""
请审核下面这条 CRM source issue 的 Q/A alignment。

============================================================
SOURCE ISSUE
============================================================

cluster_id:
{clean_text(row.get("cluster_id"))}

issue_key:
{clean_text(row.get("issue_key"))}

question:
{clean_text(row.get("question"))}

question_normalized:
{clean_text(row.get("question_normalized"))}

answer:
{clean_text(row.get("answer"))}

solution:
{clean_text(row.get("solution"))}

resolution:
{clean_text(row.get("resolution"))}

knowledge_value:
{clean_text(row.get("knowledge_value"))}

temporal_status:
{clean_text(row.get("temporal_status"))}

confidence:
{clean_text(row.get("confidence"))}


============================================================
TASK
============================================================

只判断：

question / question_normalized

是否真正被：

answer / solution

回答。

不要判断最终 Publishability。
"""


# ============================================================
# Audit
# ============================================================

def audit_issue(row):

    client = build_client()

    issue_key = clean_text(
        row.get("issue_key")
    )

    prompt = build_prompt(
        row
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
                AlignmentResult
                .model_validate(
                    data
                )
            )

            # =================================================
            # Hard rules
            # =================================================

            if result.alignment in {
                "misaligned",
                "no_answer",
                "uncertain",
            }:

                result.usable_for_kb = (
                    False
                )

            if (
                result.alignment
                == "aligned"
                and
                not result.question_answer_consistent
            ):

                result.alignment = (
                    "partially_aligned"
                )

            if (
                result.alignment
                == "partially_aligned"
                and
                not (
                    result.supported_answer
                    or
                    result.supported_solution
                )
            ):

                result.usable_for_kb = (
                    False
                )

            return {
                "cluster_id":
                    clean_text(
                        row.get(
                            "cluster_id"
                        )
                    ),

                "issue_key":
                    issue_key,

                "question":
                    clean_text(
                        row.get(
                            "question"
                        )
                    ),

                "question_normalized":
                    clean_text(
                        row.get(
                            "question_normalized"
                        )
                    ),

                "answer":
                    clean_text(
                        row.get(
                            "answer"
                        )
                    ),

                "solution":
                    clean_text(
                        row.get(
                            "solution"
                        )
                    ),

                "resolution":
                    clean_text(
                        row.get(
                            "resolution"
                        )
                    ),

                "knowledge_value":
                    clean_text(
                        row.get(
                            "knowledge_value"
                        )
                    ),

                "temporal_status":
                    clean_text(
                        row.get(
                            "temporal_status"
                        )
                    ),

                "source_confidence":
                    clean_text(
                        row.get(
                            "confidence"
                        )
                    ),

                "alignment":
                    result.alignment,

                "alignment_confidence":
                    result.confidence,

                "question_answer_consistent":
                    result.question_answer_consistent,

                "solution_supported_by_answer":
                    result.solution_supported_by_answer,

                "usable_for_kb":
                    result.usable_for_kb,

                "supported_answer":
                    result.supported_answer,

                "supported_solution":
                    result.supported_solution,

                "reason":
                    result.reason,

                "risk_flags":
                    result.risk_flags,

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

    rows = []

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

                rows.append(
                    json.loads(
                        line
                    )
                )

            except Exception:
                pass

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return

    df["risk_flags"] = (
        df["risk_flags"]
        .apply(
            lambda x:
                " | ".join(x)
                if isinstance(x, list)
                else clean_text(x)
        )
    )

    alignment_stats = (
        df[
            "alignment"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "alignment"
        )
        .reset_index(
            name="count"
        )
    )

    usable_stats = (
        df[
            "usable_for_kb"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "usable_for_kb"
        )
        .reset_index(
            name="count"
        )
    )

    unusable_df = (
        df[
            df[
                "usable_for_kb"
            ]
            != True
        ]
        .copy()
    )

    partial_df = (
        df[
            df[
                "alignment"
            ]
            == "partially_aligned"
        ]
        .copy()
    )

    cluster_stats = (
        df
        .groupby(
            "cluster_id",
            dropna=False,
        )
        .agg(
            source_count=(
                "issue_key",
                "count",
            ),

            usable_source_count=(
                "usable_for_kb",
                lambda x:
                    int(
                        (
                            x == True
                        ).sum()
                    ),
            ),

            unusable_source_count=(
                "usable_for_kb",
                lambda x:
                    int(
                        (
                            x != True
                        ).sum()
                    ),
            ),
        )
        .reset_index()
    )

    cluster_stats[
        "source_status"
    ] = cluster_stats.apply(
        lambda row:
            (
                "CLEAN"
                if (
                    row[
                        "unusable_source_count"
                    ]
                    == 0
                )
                else
                (
                    "PARTIAL_SOURCE"
                    if (
                        row[
                            "usable_source_count"
                        ]
                        > 0
                    )
                    else
                    "NO_VALID_SOURCE"
                )
            ),
        axis=1,
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "audited_source_count",
                "value":
                    len(df),
            },
            {
                "metric":
                    "usable_source_count",
                "value":
                    int(
                        (
                            df[
                                "usable_for_kb"
                            ]
                            == True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "unusable_source_count",
                "value":
                    int(
                        (
                            df[
                                "usable_for_kb"
                            ]
                            != True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "cluster_count",
                "value":
                    int(
                        df[
                            "cluster_id"
                        ].nunique()
                    ),
            },
            {
                "metric":
                    "affected_cluster_count",
                "value":
                    int(
                        (
                            cluster_stats[
                                "unusable_source_count"
                            ]
                            > 0
                        ).sum()
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

        alignment_stats.to_excel(
            writer,
            sheet_name="alignment_stats",
            index=False,
        )

        usable_stats.to_excel(
            writer,
            sheet_name="usable_stats",
            index=False,
        )

        cluster_stats.to_excel(
            writer,
            sheet_name="cluster_stats",
            index=False,
        )

        df.to_excel(
            writer,
            sheet_name="all_sources",
            index=False,
        )

        partial_df.to_excel(
            writer,
            sheet_name="partial",
            index=False,
        )

        unusable_df.to_excel(
            writer,
            sheet_name="unusable",
            index=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 11.2 - Mixed source Q/A alignment"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Step 11.1 v2 的 member decisions
    # --------------------------------------------------------

    decisions_df = pd.read_excel(
        MIXED_AUDIT_FILE,
        sheet_name="member_decisions",
    )

    # 只保留 keep
    keep_df = (
        decisions_df[
            decisions_df[
                "disposition"
            ]
            == "keep"
        ]
        .copy()
    )

    print(
        f"Keep members: "
        f"{len(keep_df)}"
    )

    # --------------------------------------------------------
    # 原始 cluster members
    # --------------------------------------------------------

    source_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    target_keys = set(
        keep_df[
            "issue_key"
        ]
        .astype(str)
        .tolist()
    )

    targets = (
        source_df[
            source_df[
                "issue_key"
            ]
            .astype(str)
            .isin(
                target_keys
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # 安全检查：
    # 43 个 keep 必须全部能在 source workbook 找到
    # --------------------------------------------------------

    found_keys = set(
        targets[
            "issue_key"
        ]
        .astype(str)
        .tolist()
    )

    missing_keys = (
        target_keys
        - found_keys
    )

    if missing_keys:

        raise RuntimeError(
            "以下 keep issue 在 "
            "question_clusters.xlsx 中找不到："
            + str(
                sorted(
                    missing_keys
                )
            )
        )

    print(
        f"Matched source members: "
        f"{len(targets)}"
    )

    processed = load_processed()

    rows = (
        targets
        .to_dict(
            orient="records"
        )
    )

    tasks = [
        row

        for row in rows

        if clean_text(
            row.get(
                "issue_key"
            )
        )
        not in processed
    ]

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
                        audit_issue,
                        row,
                    ):
                    clean_text(
                        row.get(
                            "issue_key"
                        )
                    )

                    for row in tasks
                }

                for future in as_completed(
                    future_map
                ):

                    issue_key = (
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
                            f"{issue_key}: "
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