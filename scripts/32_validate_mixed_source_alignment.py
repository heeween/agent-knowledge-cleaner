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
    / "mixed_source_qa_alignments.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments.xlsx"
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

你的任务不是判断这条知识是否最终应该发布。

你的任务只判断：

question / question_normalized

与

answer / solution

在语义上是否真正属于同一个问题。

这是一个 Source Grounding 审核。


============================================================
为什么需要这一步
============================================================

聊天切分和 LLM 抽取可能发生边界污染。

例如：

question：
“为什么年检提醒没有显示？”

answer：
“账号和密码可以借用同事的。”

虽然 question 和 answer 都来自同一段聊天，

但 answer 显然没有回答 question。

这种 source 不能进入 KB。


============================================================
aligned
============================================================

question 与 answer 明确对应。

answer 实质回答了用户的问题。

例如：

question：
“为什么已经回访了还在待回访列表？”

answer：
“因为系统没有识别最新回访状态。”

→ aligned


不同原因、不同解决方法都没问题。

重点只看：

answer 是否真正回答 question。


============================================================
partially_aligned
============================================================

answer 中至少有一部分明确回答 question，

但还夹杂：

- 其他问题内容
- 上下文噪声
- 未完成信息
- 无关处理过程
- 过多案例信息

只要能够可靠提取出：

supported_answer

或

supported_solution

仍可：

usable_for_kb = true


例如：

question：
“为什么没有录音？”

answer：
“先让技术看一下。后台录音服务昨天失败了，
另外账号密码我稍后发。”

真正支持问题的是：

“后台录音服务失败。”

→ partially_aligned


============================================================
misaligned
============================================================

answer 实际回答的是另一个问题。

例如：

question：
“如何修改车辆里程？”

answer：
“已经帮您报价成功。”

→ misaligned

usable_for_kb = false


============================================================
no_answer
============================================================

聊天只有：

- 看一下
- 稍等
- 已反馈
- 处理中
- @某人
- 需要技术确认

但没有形成实际答案或解决知识。

→ no_answer

usable_for_kb = false


============================================================
uncertain
============================================================

只有确实无法判断 Q/A 是否对应时才使用。

不要滥用 uncertain。


============================================================
特别注意
============================================================

以下情况不会自动导致 misaligned：

1. resolution = unresolved
2. temporal_status = temporary
3. knowledge_value = low
4. answer 是“目前不支持”
5. answer 只给出原因，没有完整解决方案
6. answer 只给出操作步骤，没有解释原因

这些属于知识质量或发布风险，

不是 Q/A alignment 问题。


============================================================
supported_answer
============================================================

如果 alignment = aligned：

supported_answer 可以直接保留输入 answer 的核心内容。

如果 alignment = partially_aligned：

只提取真正回答 question 的部分。

禁止：

- 补充输入中不存在的事实
- 推测产品规则
- 扩展答案范围
- 修改原始结论


如果 alignment 为：

misaligned
no_answer
uncertain

通常：

supported_answer = null


============================================================
supported_solution
============================================================

规则与 supported_answer 相同。

只能保留 source 中真实存在、
且明确服务于当前 question 的操作内容。

不能生成新的步骤。


============================================================
solution_supported_by_answer
============================================================

判断 solution 是否能由 answer / source 内容支撑。

如果 solution 明显比 answer 扩张：

false

例如：

answer：
“刷新后试一下。”

solution：
“进入设置 > 高级配置 > 清缓存后重启系统。”

source 没有这些信息。

→ false


============================================================
usable_for_kb
============================================================

原则：

aligned
+
有有效 answer

通常：

true


partially_aligned
+
可以安全提取支持内容

可以：

true


misaligned
no_answer
uncertain

必须：

false


注意：

usable_for_kb 这里只表示：

“这个 source 可以作为后续 KB 的证据”。

不是：

“这条知识已经可以正式发布”。


============================================================
Temporal / Resolution
============================================================

不要因为：

temporary
unresolved
partial

把 source 判为 unusable。

这些风险后面 Publishability Gate 再处理。

本轮只检查：

Q ↔ A 是否对齐。


============================================================
输出
============================================================

只输出合法 JSON。

不要输出 Markdown。

格式：

{
  "alignment": "aligned | partially_aligned | misaligned | no_answer | uncertain",
  "confidence": 0.0,
  "question_answer_consistent": true,
  "solution_supported_by_answer": true,
  "usable_for_kb": true,
  "supported_answer": "仅保留真正被 source 支持的答案",
  "supported_solution": "仅保留真正被 source 支持的方案",
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