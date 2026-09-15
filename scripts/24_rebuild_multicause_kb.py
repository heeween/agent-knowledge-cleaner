import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

VALIDATION_FILE = (
    OUTPUT_DIR
    / "multicause_item_validations_v2.xlsx"
)

ORIGINAL_FILE = (
    OUTPUT_DIR
    / "kb_entries_multicause.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entries_multicause_rebuilt.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_multicause_rebuilt.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class FinalCause(BaseModel):

    cause: str

    check_or_action: str | None = None

    source_issue_keys: list[str]


class RebuiltEntry(BaseModel):

    canonical_question: str

    causes: list[FinalCause]

    notes: list[str]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在整理已经经过严格 source grounding 审核的
CRM multiple_causes 知识。

注意：

输入中的原因已经完成真实性审核。

你的任务不是重新分析原始问题，
也不是重新生成新的原因。

你只允许做：

1. 合并语义重复的原因
2. 使用审核后的 supported_cause
3. 使用审核后的 supported_action
4. 合并对应 source_issue_keys
5. 做轻微语言规范化


============================================================
最高优先级禁止事项
============================================================

禁止新增任何输入中不存在的：

- 原因
- 检查方法
- 操作步骤
- 产品规则
- 菜单路径
- 时间
- 限制条件

如果输入只有 2 个独立原因，
最终最多只能保留 2 个独立原因。

不能为了“完整”添加第三个原因。


============================================================
输入 Cause
============================================================

输入中只会提供已经允许保留的原因：

- valid
- action_unsupported

unsupported / wrong_intent / not_a_cause
已经被删除，不允许恢复。


============================================================
supported_cause
============================================================

优先使用：

supported_cause

如果为空，才使用：

original_cause


============================================================
supported_action
============================================================

如果 verdict = valid：

优先使用 supported_action。

如果 supported_action 为空，
且 original_action 有明确支持，
可以保留 original_action。


如果 verdict = action_unsupported：

只能使用：

supported_action

如果 supported_action 为空：

check_or_action = null

绝对不能恢复 original_action。


============================================================
重复原因合并
============================================================

只有语义本质相同才合并。

例如：

A：
短信签名正在报备中，尚未完成。

B：
短信签名尚未成功，需要重新提交报备。

可以合并：

“短信签名尚未完成报备或审核。”


同时合并：

source_issue_keys。


------------------------------------------------------------

下面不能合并：

短信签名未报备

短信内容包含敏感词

短信余额不足

这是三个独立机制。


============================================================
action 合并
============================================================

如果两个重复原因分别有不同、
但都有明确来源支持的操作，

可以将它们简洁合并。

不能增加新步骤。


============================================================
canonical_question
============================================================

保持输入 canonical_question。

最多轻微润色，
不得扩大或缩小意图。


============================================================
notes
============================================================

仅保留必要的风险提示。

如果某个原因存在：

temporary
unresolved
future_plan

可以写简短 note。

但不要因为这些标签删除真实原因。

最终是否发布属于后续 publishability 审核。


============================================================
source_issue_keys
============================================================

每个最终 cause：

source_issue_keys 必须来自输入原因。

若合并两个 cause，
将两边 key 去重合并。

禁止生成新 key。


============================================================
输出
============================================================

只输出合法 JSON：

{
  "canonical_question": "问题",
  "causes": [
    {
      "cause": "原因",
      "check_or_action": "操作或 null",
      "source_issue_keys": []
    }
  ],
  "notes": []
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
    cluster_id,
    canonical_question,
    rows,
):

    blocks = []

    for idx, row in enumerate(
        rows,
        start=1,
    ):

        supported_cause = clean_text(
            row.get(
                "supported_cause"
            )
        )

        if not supported_cause:

            supported_cause = clean_text(
                row.get(
                    "original_cause"
                )
            )

        supported_action = clean_text(
            row.get(
                "supported_action"
            )
        )

        verdict = clean_text(
            row.get(
                "verdict"
            )
        )

        # valid 时，如果审核没有返回修正版 action，
        # 可以使用原 action
        if (
            verdict == "valid"
            and not supported_action
        ):

            supported_action = clean_text(
                row.get(
                    "original_action"
                )
            )

        # action_unsupported：
        # 绝不能恢复 original_action
        if (
            verdict
            == "action_unsupported"
        ):

            supported_action = clean_text(
                row.get(
                    "supported_action"
                )
            )

        blocks.append(
            f"""
----------------------------------------
CAUSE {idx}

verdict:
{verdict}

cause:
{supported_cause}

check_or_action:
{supported_action or "null"}

source_issue_keys:
{clean_text(row.get("source_issue_keys"))}

risk_flags:
{clean_text(row.get("risk_flags"))}
""".strip()
        )

    return f"""
请整理下面已经审核通过的 multiple-causes 知识。

cluster_id:
{cluster_id}

canonical_question:
{canonical_question}


============================================================
ALLOWED CAUSES
============================================================

{chr(10).join(blocks)}
"""


# ============================================================
# API
# ============================================================

def rebuild_cluster(
    cluster_id,
    canonical_question,
    rows,
):

    client = build_client()

    prompt = build_prompt(
        cluster_id,
        canonical_question,
        rows,
    )

    allowed_keys = set()

    for row in rows:

        allowed_keys.update(
            parse_keys(
                row.get(
                    "source_issue_keys"
                )
            )
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
                RebuiltEntry
                .model_validate(
                    data
                )
            )

            cleaned_causes = []

            for cause in result.causes:

                keys = []

                for key in (
                    cause.source_issue_keys
                ):

                    if (
                        key in allowed_keys
                        and key not in keys
                    ):
                        keys.append(
                            key
                        )

                # 无来源的不允许进入
                if not keys:
                    continue

                cleaned_causes.append(
                    {
                        "cause":
                            cause.cause,

                        "check_or_action":
                            cause.check_or_action,

                        "source_issue_keys":
                            keys,
                    }
                )

            return {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    result.canonical_question,

                "causes":
                    cleaned_causes,

                "notes":
                    result.notes,

                "cause_count_before":
                    len(rows),

                "cause_count_after":
                    len(
                        cleaned_causes
                    ),

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
# Export
# ============================================================

def export_excel(
    rejected_cluster_rows,
):

    objects = []

    if OUTPUT_JSONL.exists():

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

    entry_rows = []
    cause_rows = []

    for obj in objects:

        cluster_id = obj[
            "cluster_id"
        ]

        causes = obj.get(
            "causes",
            []
        )

        entry_rows.append(
            {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    obj.get(
                        "canonical_question"
                    ),

                "cause_count_before":
                    obj.get(
                        "cause_count_before"
                    ),

                "cause_count_after":
                    len(causes),

                "notes":
                    " | ".join(
                        obj.get(
                            "notes",
                            []
                        )
                    ),

                "request_seconds":
                    obj.get(
                        "request_seconds"
                    ),
            }
        )

        for idx, cause in enumerate(
            causes,
            start=1,
        ):

            cause_rows.append(
                {
                    "cluster_id":
                        cluster_id,

                    "canonical_question":
                        obj.get(
                            "canonical_question"
                        ),

                    "cause_index":
                        idx,

                    "cause":
                        cause.get(
                            "cause"
                        ),

                    "check_or_action":
                        cause.get(
                            "check_or_action"
                        ),

                    "source_issue_keys":
                        " | ".join(
                            cause.get(
                                "source_issue_keys",
                                []
                            )
                        ),
                }
            )

    entries_df = pd.DataFrame(
        entry_rows
    )

    causes_df = pd.DataFrame(
        cause_rows
    )

    rejected_df = pd.DataFrame(
        rejected_cluster_rows
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "rebuilt_cluster_count",
                "value":
                    len(entries_df),
            },
            {
                "metric":
                    "rejected_cluster_count",
                "value":
                    len(rejected_df),
            },
            {
                "metric":
                    "final_cause_count",
                "value":
                    len(causes_df),
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

        if not entries_df.empty:

            entries_df.to_excel(
                writer,
                sheet_name="entries",
                index=False,
            )

        if not causes_df.empty:

            causes_df.to_excel(
                writer,
                sheet_name="causes",
                index=False,
            )

        if not rejected_df.empty:

            rejected_df.to_excel(
                writer,
                sheet_name="rejected_clusters",
                index=False,
            )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 10.3 - Rebuild validated multiple-causes KB"
    )
    print("=" * 70)

    validation_df = pd.read_excel(
        VALIDATION_FILE,
        sheet_name="all_causes",
    )

    original_entries_df = (
        pd.read_excel(
            ORIGINAL_FILE,
            sheet_name="entries",
        )
    )

    # --------------------------------------------------------
    # 只允许：
    # valid
    # action_unsupported
    # --------------------------------------------------------

    usable_df = validation_df[
        validation_df[
            "verdict"
        ].isin(
            [
                "valid",
                "action_unsupported",
            ]
        )
        &
        (
            validation_df[
                "cause_supported"
            ]
            == True
        )
        &
        (
            validation_df[
                "answers_question"
            ]
            == True
        )
        &
        (
            validation_df[
                "is_real_cause"
            ]
            == True
        )
    ].copy()

    processed = load_processed()

    tasks = []

    rejected_cluster_rows = []

    all_cluster_ids = (
        original_entries_df[
            "cluster_id"
        ]
        .astype(str)
        .tolist()
    )

    for cluster_id in all_cluster_ids:

        entry_row = (
            original_entries_df[
                original_entries_df[
                    "cluster_id"
                ]
                .astype(str)
                == cluster_id
            ]
            .iloc[0]
        )

        canonical_question = clean_text(
            entry_row.get(
                "canonical_question"
            )
        )

        rows_df = usable_df[
            usable_df[
                "cluster_id"
            ]
            .astype(str)
            == cluster_id
        ].copy()

        # ----------------------------------------------------
        # 少于 2 个原因：
        # 已经不再是 multiple-causes
        # 不自动发布
        # ----------------------------------------------------

        if len(rows_df) < 2:

            rejected_cluster_rows.append(
                {
                    "cluster_id":
                        cluster_id,

                    "canonical_question":
                        canonical_question,

                    "usable_cause_count":
                        len(rows_df),

                    "status":
                        "INSUFFICIENT_MULTICAUSE",

                    "reason":
                        (
                            "经过 cause grounding "
                            "过滤后不足两个可靠独立原因。"
                        ),
                }
            )

            continue

        if cluster_id in processed:
            continue

        tasks.append(
            (
                cluster_id,
                canonical_question,
                rows_df.to_dict(
                    orient="records"
                ),
            )
        )

    print(
        f"Original cluster count: "
        f"{len(all_cluster_ids)}"
    )

    print(
        f"Can rebuild: "
        f"{len(tasks)}"
    )

    print(
        f"Insufficient multicause: "
        f"{len(rejected_cluster_rows)}"
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
                        rebuild_cluster,
                        cluster_id,
                        canonical_question,
                        rows,
                    ):
                    cluster_id

                    for (
                        cluster_id,
                        canonical_question,
                        rows,
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
                        f"| ETA "
                        f"{eta/60:.1f} min"
                    )

    export_excel(
        rejected_cluster_rows
    )

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