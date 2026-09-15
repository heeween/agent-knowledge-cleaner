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

REGENERATED_FILE = (
    OUTPUT_DIR
    / "kb_entries_regenerated.xlsx"
)

CLUSTER_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)

ALIGNMENT_FILE = (
    OUTPUT_DIR
    / "issue_qa_alignments.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "regenerated_publishability.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "regenerated_publishability.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class PublishabilityResult(BaseModel):

    verdict: Literal[
        "publish",
        "publish_with_caution",
        "case_only",
        "insufficient_evidence",
        "scope_mismatch",
        "manual_review",
    ]

    confidence: float

    answer_supported: bool

    reusable: bool

    scope_aligned: bool

    stable_enough: bool

    evidence_quality: Literal[
        "high",
        "medium",
        "low",
    ]

    reason: str

    risk_flags: list[str]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在审核一条准备进入正式 CRM RAG 知识库的知识。

注意：

这条知识已经经过 source QA 对齐检查。
所以当前问题不是简单判断“答案是否来自来源”。

你的任务是判断：

“这条知识是否适合长期、可复用地发布到 CRM RAG？”


============================================================
verdict
============================================================

publish

满足：

- answer 有明确来源支持
- 问题范围和答案范围一致
- 知识可复用
- 不是单次客户事故状态
- 不是未确认推测
- 没有明显时效风险
- 可以作为正式知识回答类似用户问题


============================================================

publish_with_caution

核心知识可复用，
但存在明确限定条件。

例如：

- 只适用于某个功能
- 只支持某个操作场景
- 答案只是问题的一种处理方式

前提是这些限定条件能够清楚写入最终知识。


============================================================

case_only

来源描述的是一次具体事故或项目状态，
不能推广为稳定知识。

例如：

- 开发刚刚修复
- 已手动补发
- 等周一导数据
- 当前门店尚未开通
- 临时技术处理
- 某个客户当前的数据异常

即使这个案例答案完全真实，
也不能成为通用 RAG 产品规则。


============================================================

insufficient_evidence

现有来源不足以形成可靠知识。

例如：

- 只是推测
- “可能是”
- “我看看”
- 未解决
- 只知道需要技术排查
- solution 本身写着“基于客户请求推断”

如果问题问：

“为什么没有显示？”

来源只说：

“需要进一步排查原因”

则属于 insufficient_evidence。


============================================================

scope_mismatch

问题范围明显比来源答案更宽，
最终知识会造成错误泛化。

例如：

问题：

“如何在 CRM 中跟进商机？”

来源只说：

“把某一个商机类型设置为增项未修。”

如果没有证据证明所有商机都应该这样跟进，
则不能将这一特定操作推广为通用“商机跟进方法”。

应判：

scope_mismatch


另一个例子：

问题：

“CRM 登录失败如何重置密码？”

来源实际只是：

“该用户没有账号，所以后台新建账号，
新账号密码为手机号。”

新建账号处理，
不等于通用密码重置流程。

如果最终问题范围过宽，
应判 scope_mismatch。


============================================================

manual_review

材料复杂，
无法可靠归入以上类型。


============================================================
answer_supported
============================================================

只判断最终生成 answer
是否能被本次实际 source 支持。

有依据 → true。


============================================================
reusable
============================================================

是否适合回答未来其他客户的同类问题。

以下通常 false：

- 单次故障结果
- 某门店当前状态
- 某个具体日期计划
- 等待技术处理
- 某次人工补发
- 某次临时 workaround


============================================================
scope_aligned
============================================================

canonical_question 的范围
是否和来源真正支持的知识范围一致。

非常重要：

不能因为 source 的 question_normalized
写得很宽，
就自动认为答案也支持这么宽的范围。

必须检查实际 answer / solution。


============================================================
stable_enough
============================================================

不是要求永久不变。

而是至少现有证据没有明确表示：

- temporary
- future_plan
- 正在开发
- 正在排查
- 等升级
- 临时方案
- 未解决

如果这些状态构成知识核心，
通常 false。


============================================================
evidence_quality
============================================================

high：

- resolved
- high knowledge_value
- stable
- 问答明确
- 可操作

medium：

存在少量限制，
但仍有明确知识。

low：

- unresolved
- temporary
- low knowledge_value
- 推断性内容
- 单次案例
- 缺少最终结论


============================================================
risk_flags
============================================================

可以使用：

temporary
unresolved
low_knowledge_value
single_case
inferred_solution
question_too_broad
answer_only_partial
requires_technical_confirmation
historical_or_version_risk
other

没有则 []。


============================================================
特别规则
============================================================

1. temporary + unresolved

通常不能直接 publish。


2. “系统出现异常，技术已修复并补发”

这是案例处理结果。

如果没有进一步稳定规则：

case_only。


3. “可能很多数据没导入，正在处理”

如果问题问：

“为什么没有消费记录？”

这只是一个尚未确认的可能原因：

insufficient_evidence。


4. source 中明确出现：

“基于客户请求推断”

应高度警惕 inferred_solution。


5. 对 how_to 问题：

如果 source 提供清晰、稳定的操作步骤，
可以 publish。


6. 不要为了提高 KB 数量而放宽标准。

宁愿少发布，
也不能把临时状态和猜测变成产品知识。


============================================================
输出
============================================================

只输出合法 JSON object。

{
  "verdict": "publish | publish_with_caution | case_only | insufficient_evidence | scope_mismatch | manual_review",
  "confidence": 0.0,
  "answer_supported": true,
  "reusable": true,
  "scope_aligned": true,
  "stable_enough": true,
  "evidence_quality": "high | medium | low",
  "reason": "简短原因",
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


def parse_keys(value):

    text = clean_text(value)

    if not text:
        return []

    return [
        x.strip()
        for x in text.split("|")
        if x.strip()
    ]


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

                obj = json.loads(line)

                cluster_id = obj.get(
                    "cluster_id"
                )

                if cluster_id:
                    processed.add(
                        cluster_id
                    )

            except Exception:
                continue

    return processed


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    kb_row,
    source_rows,
):

    blocks = []

    for idx, row in enumerate(
        source_rows,
        start=1,
    ):

        blocks.append(
            f"""
----------------------------------------
SOURCE {idx}

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

alignment:
{clean_text(row.get("alignment"))}

alignment_confidence:
{clean_text(row.get("alignment_confidence"))}
""".strip()
        )

    return f"""
请判断下面重新生成的 CRM 知识，
是否适合作为正式、长期可复用的 RAG 知识。

============================================================
GENERATED KB
============================================================

cluster_id:
{clean_text(kb_row.get("cluster_id"))}

canonical_question:
{clean_text(kb_row.get("canonical_question"))}

answer:
{clean_text(kb_row.get("answer"))}

solution_steps:
{clean_text(kb_row.get("solution_steps"))}

notes:
{clean_text(kb_row.get("notes"))}

crm_module:
{clean_text(kb_row.get("crm_module"))}

crm_feature:
{clean_text(kb_row.get("crm_feature"))}

problem_type:
{clean_text(kb_row.get("problem_type"))}

knowledge_confidence:
{clean_text(kb_row.get("knowledge_confidence"))}


============================================================
ACTUAL SOURCES
============================================================

{chr(10).join(blocks)}
"""


# ============================================================
# API
# ============================================================

def validate_entry(
    kb_row,
    source_rows,
):

    client = build_client()

    prompt = build_prompt(
        kb_row,
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
                client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
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
                PublishabilityResult
                .model_validate(
                    data
                )
            )

            # --------------------------------------------
            # 硬规则
            # --------------------------------------------

            if not result.answer_supported:
                result.verdict = (
                    "insufficient_evidence"
                )

            if not result.scope_aligned:
                result.verdict = (
                    "scope_mismatch"
                )

            if (
                result.verdict
                == "publish"
                and (
                    not result.reusable
                    or not result.stable_enough
                )
            ):
                result.verdict = (
                    "publish_with_caution"
                )

            return {
                "cluster_id":
                    clean_text(
                        kb_row.get(
                            "cluster_id"
                        )
                    ),

                "canonical_question":
                    clean_text(
                        kb_row.get(
                            "canonical_question"
                        )
                    ),

                "verdict":
                    result.verdict,

                "audit_confidence":
                    result.confidence,

                "answer_supported":
                    result.answer_supported,

                "reusable":
                    result.reusable,

                "scope_aligned":
                    result.scope_aligned,

                "stable_enough":
                    result.stable_enough,

                "evidence_quality":
                    result.evidence_quality,

                "risk_flags":
                    result.risk_flags,

                "reason":
                    result.reason,

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
# Excel
# ============================================================

def export_excel(
    kb_df,
):

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
                    json.loads(line)
                )
            except Exception:
                pass

    audit_df = pd.DataFrame(
        rows
    )

    if audit_df.empty:
        return

    audit_df[
        "risk_flags"
    ] = audit_df[
        "risk_flags"
    ].apply(
        lambda x:
            " | ".join(x)
            if isinstance(x, list)
            else clean_text(x)
    )

    merged_df = kb_df.merge(
        audit_df,
        on=[
            "cluster_id",
            "canonical_question",
        ],
        how="left",
    )

    approved_df = merged_df[
        merged_df[
            "verdict"
        ].isin(
            [
                "publish",
                "publish_with_caution",
            ]
        )
        &
        (
            merged_df[
                "answer_supported"
            ]
            == True
        )
        &
        (
            merged_df[
                "scope_aligned"
            ]
            == True
        )
        &
        (
            merged_df[
                "reusable"
            ]
            == True
        )
    ].copy()

    rejected_df = merged_df[
        ~merged_df[
            "cluster_id"
        ].isin(
            approved_df[
                "cluster_id"
            ]
        )
    ].copy()

    verdict_stats = (
        audit_df[
            "verdict"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "verdict"
        )
        .reset_index(
            name="count"
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "audited_count",
                "value":
                    len(audit_df),
            },
            {
                "metric":
                    "approved_count",
                "value":
                    len(approved_df),
            },
            {
                "metric":
                    "rejected_count",
                "value":
                    len(rejected_df),
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

        verdict_stats.to_excel(
            writer,
            sheet_name="verdict_stats",
            index=False,
        )

        merged_df.to_excel(
            writer,
            sheet_name="all_entries",
            index=False,
        )

        approved_df.to_excel(
            writer,
            sheet_name="approved",
            index=False,
        )

        rejected_df.to_excel(
            writer,
            sheet_name="rejected",
            index=False,
        )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 9.6 - Validate regenerated publishability"
    )
    print("=" * 70)

    kb_df = pd.read_excel(
        REGENERATED_FILE,
        sheet_name="regenerated",
    )

    member_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    alignment_df = pd.read_excel(
        ALIGNMENT_FILE,
        sheet_name="all_issues",
    )

    alignment_small = (
        alignment_df[
            [
                "issue_key",
                "alignment",
                "alignment_confidence",
            ]
        ]
        .drop_duplicates(
            subset=["issue_key"]
        )
    )

    members = member_df.merge(
        alignment_small,
        on="issue_key",
        how="left",
    )

    processed = load_processed()

    tasks = []

    for _, kb_row in kb_df.iterrows():

        cluster_id = clean_text(
            kb_row.get(
                "cluster_id"
            )
        )

        if cluster_id in processed:
            continue

        source_keys = set(
            parse_keys(
                kb_row.get(
                    "source_issue_keys"
                )
            )
        )

        source_rows = (
            members[
                members[
                    "issue_key"
                ].astype(str)
                .isin(
                    source_keys
                )
            ]
            .to_dict(
                orient="records"
            )
        )

        tasks.append(
            (
                kb_row,
                source_rows,
            )
        )

    print(
        f"Regenerated entry count: "
        f"{len(kb_df)}"
    )

    print(
        f"本次待审核: "
        f"{len(tasks)}"
    )

    total = len(tasks)
    completed = 0
    started_at = time.time()

    lock = threading.Lock()

    with OUTPUT_JSONL.open(
        "a",
        encoding="utf-8",
    ) as fout:

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_map = {
                executor.submit(
                    validate_entry,
                    kb_row,
                    sources,
                ):
                clean_text(
                    kb_row.get(
                        "cluster_id"
                    )
                )

                for kb_row, sources
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
                    result = future.result()

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
                    f"| ETA "
                    f"{eta/60:.1f} min"
                )

    export_excel(
        kb_df
    )

    print()
    print(
        f"Excel: {OUTPUT_XLSX}"
    )


if __name__ == "__main__":
    main()