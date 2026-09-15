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

REBUILT_FILE = (
    OUTPUT_DIR
    / "kb_entries_multicause_rebuilt.xlsx"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "multicause_item_validations_v2.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "multicause_rebuild_validations.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "multicause_rebuild_validations.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class RebuildValidation(BaseModel):

    verdict: Literal[
        "pass",
        "cause_overreach",
        "action_overreach",
        "bad_merge",
        "source_key_mismatch",
        "needs_manual_review",
    ]

    confidence: float

    cause_supported: bool

    action_supported: bool

    merge_valid: bool

    source_keys_valid: bool

    corrected_cause: str | None = None

    corrected_action: str | None = None

    reason: str

    risk_flags: list[str]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在审核 CRM multiple_causes 知识的“重建结果”。

之前已经有一层审核，
确认哪些原始 Cause 可以使用。

现在模型把这些允许的 Cause：

- 做了语言规范化
- 合并重复原因
- 合并 source_issue_keys

本轮只判断：

“重建后的 Cause 有没有超出允许素材。”


============================================================
非常重要
============================================================

不要重新判断 CRM 产品知识是否正确。

不要根据常识补充信息。

只比较：

ALLOWED CAUSES

和：

REBUILT CAUSE


============================================================
verdict = pass
============================================================

满足：

1. rebuilt cause 完全由 allowed causes 支持
2. 没有增加新原因
3. 如果合并多个 allowed cause，它们语义确实是同一原因
4. rebuilt action 完全由 allowed action 支持
5. source keys 来自实际支持该最终 cause 的 allowed causes


============================================================
cause_overreach
============================================================

重建后的 cause：

- 新增了原素材没有的机制
- 新增了时间
- 新增了条件
- 新增了产品规则
- 把处理方案写成了原因的一部分
- 把“可能”改成了确定事实


例如：

允许：

cause:
短信签名尚未完成报备

重建：

短信签名尚未完成报备，
需等待次日重新提交报备尝试

如果“等待次日重新提交”属于处理方式，
不是原因本体：

cause_overreach

corrected_cause 应只保留：

短信签名尚未完成报备或审核


============================================================
action_overreach
============================================================

cause 本身没问题，
但 check_or_action：

- 新增操作
- 新增菜单
- 新增时间
- 新增处理规则
- 合并出了来源没有支持的操作

则：

action_overreach


============================================================
bad_merge
============================================================

重建时把本质不同的原因合成一个原因。

例如：

A：
没有连接麦克风

B：
没有切换耳机模式

如果来源表明二者是不同独立故障机制，
不能因为都属于“音频问题”就合成一个原因。

但如果来源本身描述：

“设备未正确连接或未切换耳机模式”

作为同一套配置问题，

则允许合并。

必须基于输入 ALLOWED CAUSES 判断。


============================================================
source_key_mismatch
============================================================

最终 cause 的 source_issue_keys：

- 包含没有支持这个最终原因的 key
- 缺失实际被合并的重要 key
- 使用了 ALLOWED CAUSES 中不存在的 key

则使用此 verdict。


============================================================
corrected_cause
============================================================

如果 reconstructed cause 有轻微越权，
但可以直接从 allowed causes 得到一个安全版本，

输出安全版本。

否则 null。


============================================================
corrected_action
============================================================

只保留 allowed actions 明确支持的内容。

没有安全 action：

null。


============================================================
特别规则：Cause 与 Action 分离
============================================================

cause 只应该描述：

“为什么发生问题”。

例如：

正确：

cause:
短信签名尚未完成运营商报备

action:
等待报备完成后再尝试发送


不推荐：

cause:
短信签名尚未完成运营商报备，
需要等待明天重新提交


后半段属于 action，
不能混入 cause。


============================================================
输出
============================================================

只输出合法 JSON：

{
  "verdict": "pass | cause_overreach | action_overreach | bad_merge | source_key_mismatch | needs_manual_review",
  "confidence": 0.0,
  "cause_supported": true,
  "action_supported": true,
  "merge_valid": true,
  "source_keys_valid": true,
  "corrected_cause": null,
  "corrected_action": null,
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

    result = set()

    if not OUTPUT_JSONL.exists():
        return result

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

                item_id = obj.get(
                    "item_id"
                )

                if item_id:
                    result.add(item_id)

            except Exception:
                pass

    return result


# ============================================================
# 生成审核 Prompt
# ============================================================

def build_prompt(
    rebuilt_row,
    allowed_rows,
):

    blocks = []

    for idx, row in enumerate(
        allowed_rows,
        start=1,
    ):

        verdict = clean_text(
            row.get("verdict")
        )

        cause = clean_text(
            row.get("supported_cause")
        )

        if not cause:
            cause = clean_text(
                row.get("original_cause")
            )

        action = clean_text(
            row.get("supported_action")
        )

        if (
            verdict == "valid"
            and not action
        ):
            action = clean_text(
                row.get("original_action")
            )

        if (
            verdict
            == "action_unsupported"
        ):
            action = clean_text(
                row.get("supported_action")
            )

        blocks.append(
            f"""
----------------------------------------
ALLOWED CAUSE {idx}

original_item_id:
{clean_text(row.get("item_id"))}

verdict:
{verdict}

cause:
{cause}

action:
{action or "null"}

source_issue_keys:
{clean_text(row.get("source_issue_keys"))}
""".strip()
        )

    return f"""
请审核下面的 rebuilt cause 是否严格来自允许素材。

============================================================
QUESTION
============================================================

cluster_id:
{clean_text(rebuilt_row.get("cluster_id"))}

canonical_question:
{clean_text(rebuilt_row.get("canonical_question"))}


============================================================
REBUILT CAUSE
============================================================

cause_index:
{clean_text(rebuilt_row.get("cause_index"))}

cause:
{clean_text(rebuilt_row.get("cause"))}

check_or_action:
{clean_text(rebuilt_row.get("check_or_action")) or "null"}

source_issue_keys:
{clean_text(rebuilt_row.get("source_issue_keys"))}


============================================================
ALLOWED CAUSES FROM PREVIOUS GATE
============================================================

{chr(10).join(blocks)}
"""


# ============================================================
# Validate
# ============================================================

def validate_item(
    rebuilt_row,
    allowed_rows,
):

    client = build_client()

    cluster_id = clean_text(
        rebuilt_row.get(
            "cluster_id"
        )
    )

    cause_index = int(
        rebuilt_row.get(
            "cause_index"
        )
    )

    item_id = (
        f"{cluster_id}"
        f"#REBUILT-{cause_index:02d}"
    )

    prompt = build_prompt(
        rebuilt_row,
        allowed_rows,
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
                RebuildValidation
                .model_validate(
                    data
                )
            )

            # --------------------------------------------
            # 硬规则
            # --------------------------------------------

            if not result.cause_supported:

                result.verdict = (
                    "cause_overreach"
                )

            elif not result.merge_valid:

                result.verdict = (
                    "bad_merge"
                )

            elif not result.action_supported:

                result.verdict = (
                    "action_overreach"
                )

            elif not result.source_keys_valid:

                result.verdict = (
                    "source_key_mismatch"
                )

            return {
                "item_id":
                    item_id,

                "cluster_id":
                    cluster_id,

                "canonical_question":
                    clean_text(
                        rebuilt_row.get(
                            "canonical_question"
                        )
                    ),

                "cause_index":
                    cause_index,

                "rebuilt_cause":
                    clean_text(
                        rebuilt_row.get(
                            "cause"
                        )
                    ),

                "rebuilt_action":
                    clean_text(
                        rebuilt_row.get(
                            "check_or_action"
                        )
                    ),

                "source_issue_keys":
                    clean_text(
                        rebuilt_row.get(
                            "source_issue_keys"
                        )
                    ),

                "verdict":
                    result.verdict,

                "audit_confidence":
                    result.confidence,

                "cause_supported":
                    result.cause_supported,

                "action_supported":
                    result.action_supported,

                "merge_valid":
                    result.merge_valid,

                "source_keys_valid":
                    result.source_keys_valid,

                "corrected_cause":
                    result.corrected_cause,

                "corrected_action":
                    result.corrected_action,

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
# Excel
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
                    json.loads(line)
                )
            except Exception:
                pass

    df = pd.DataFrame(rows)

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

    pass_df = df[
        df["verdict"]
        == "pass"
    ].copy()

    fix_df = df[
        df["verdict"]
        != "pass"
    ].copy()

    verdict_stats = (
        df["verdict"]
        .value_counts(
            dropna=False
        )
        .rename_axis("verdict")
        .reset_index(
            name="count"
        )
    )

    cluster_stats = (
        df.groupby(
            [
                "cluster_id",
                "canonical_question",
            ],
            dropna=False,
        )
        .agg(
            cause_count=(
                "item_id",
                "count",
            ),
            pass_count=(
                "verdict",
                lambda s:
                    int(
                        (
                            s == "pass"
                        ).sum()
                    ),
            ),
            fix_count=(
                "verdict",
                lambda s:
                    int(
                        (
                            s != "pass"
                        ).sum()
                    ),
            ),
        )
        .reset_index()
    )

    cluster_stats[
        "cluster_status"
    ] = cluster_stats.apply(
        lambda row:
            "CLEAN"
            if row["fix_count"] == 0
            else "NEEDS_FIX",
        axis=1,
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "rebuilt_cause_count",
                "value":
                    len(df),
            },
            {
                "metric":
                    "pass_count",
                "value":
                    len(pass_df),
            },
            {
                "metric":
                    "needs_fix_count",
                "value":
                    len(fix_df),
            },
            {
                "metric":
                    "cluster_count",
                "value":
                    len(cluster_stats),
            },
            {
                "metric":
                    "clean_cluster_count",
                "value":
                    int(
                        (
                            cluster_stats[
                                "cluster_status"
                            ]
                            == "CLEAN"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "needs_fix_cluster_count",
                "value":
                    int(
                        (
                            cluster_stats[
                                "cluster_status"
                            ]
                            == "NEEDS_FIX"
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

        verdict_stats.to_excel(
            writer,
            sheet_name="verdict_stats",
            index=False,
        )

        cluster_stats.to_excel(
            writer,
            sheet_name="cluster_stats",
            index=False,
        )

        df.to_excel(
            writer,
            sheet_name="all_causes",
            index=False,
        )

        pass_df.to_excel(
            writer,
            sheet_name="pass",
            index=False,
        )

        fix_df.to_excel(
            writer,
            sheet_name="needs_fix",
            index=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 10.4 - Validate rebuilt multicause"
    )
    print("=" * 70)

    rebuilt_df = pd.read_excel(
        REBUILT_FILE,
        sheet_name="causes",
    )

    validation_df = pd.read_excel(
        VALIDATION_FILE,
        sheet_name="all_causes",
    )

    # 只允许前一层通过的素材
    allowed_df = validation_df[
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

    for _, rebuilt_row in (
        rebuilt_df.iterrows()
    ):

        cluster_id = clean_text(
            rebuilt_row.get(
                "cluster_id"
            )
        )

        cause_index = int(
            rebuilt_row.get(
                "cause_index"
            )
        )

        item_id = (
            f"{cluster_id}"
            f"#REBUILT-{cause_index:02d}"
        )

        if item_id in processed:
            continue

        rebuilt_keys = set(
            parse_keys(
                rebuilt_row.get(
                    "source_issue_keys"
                )
            )
        )

        # 只提供 source key 有交集的 allowed causes，
        # 防止模型把整个 cluster 其他 cause 混进来
        candidate_df = allowed_df[
            allowed_df[
                "cluster_id"
            ].astype(str)
            == cluster_id
        ].copy()

        allowed_rows = []

        for _, row in (
            candidate_df.iterrows()
        ):

            allowed_keys = set(
                parse_keys(
                    row.get(
                        "source_issue_keys"
                    )
                )
            )

            if (
                rebuilt_keys
                & allowed_keys
            ):
                allowed_rows.append(
                    row.to_dict()
                )

        tasks.append(
            (
                rebuilt_row,
                allowed_rows,
            )
        )

    print(
        f"Rebuilt cause count: "
        f"{len(rebuilt_df)}"
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
                        validate_item,
                        rebuilt_row,
                        allowed_rows,
                    ):
                    clean_text(
                        rebuilt_row.get(
                            "cluster_id"
                        )
                    )

                    for (
                        rebuilt_row,
                        allowed_rows,
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

                    if (
                        completed % 10 == 0
                        or completed == total
                    ):

                        print(
                            f"[{completed}/{total}] "
                            f"| ETA "
                            f"{eta/60:.1f} min"
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