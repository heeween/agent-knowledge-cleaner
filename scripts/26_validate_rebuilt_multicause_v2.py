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
    / "multicause_rebuild_validations_v2.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "multicause_rebuild_validations_v2.xlsx"
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
你正在审核 CRM multiple_causes 知识的重建结果。

之前已经有一层审核，
确认哪些原始 Cause 可以使用。

现在模型可能：

- 规范化了文字
- 合并了一些 Cause
- 合并了 source_issue_keys

本轮最重要的任务是判断：

“被合并的 Cause 是否真的属于同一个故障机制。”


============================================================
核心原则
============================================================

multiple_causes 知识的价值，
就是保留多个独立原因。

因此：

宁可保留两个相似但独立的 Cause，
也不能为了减少条数，
把不同故障机制合成一个 Cause。


只有：

“语义等价”

或：

“同一个原因的不同表达/状态”

才允许合并。


============================================================
允许合并
============================================================

例如：

A：
短信签名正在运营商报备中

B：
短信签名尚未完成运营商报备

本质机制相同：

短信签名未完成报备。

→ merge_valid = true


例如：

A：
客户手机号为空号

B：
接收号码为空号

本质完全一致。

→ merge_valid = true


============================================================
禁止合并：同一类别不等于同一原因
============================================================

例如：

A：
电脑没有正确连接麦克风/音频输出

B：
没有切换到耳机模式

虽然都属于“音频配置”，
但发生机制不同。

应保留两个 Cause。

→ merge_valid = false
→ verdict = bad_merge


============================================================
禁止合并：不同数据链路原因
============================================================

例如：

A：
工单没有录入

B：
跨店数据没有同步

虽然最终都导致：

“保养记录没有显示”

但这是两个独立原因。

→ merge_valid = false
→ bad_merge


============================================================
禁止合并：不同系统环节
============================================================

例如：

A：
浏览器不兼容

B：
运营商后台故障

绝不能因为都造成“电话没有声音”
而合并。


============================================================
禁止合并：用“或”连接不同原因
============================================================

如果 rebuilt cause 的结构是：

“原因 A 或 原因 B 导致……”

需要高度警惕。

如果 A 和 B 可以分别独立发生，
就是两个 Cause。

应该：

merge_valid = false


注意：

“或”本身不是绝对错误。

例如：

“短信签名正在报备或尚未完成报备”

两者本质都是：

“签名尚未完成报备”

这种情况可以合并。


============================================================
如何判断是不是同一机制
============================================================

判断：

如果一个原因成立，
另一个原因仍然可以完全不成立，
并且问题仍然会发生，

通常说明它们是两个独立原因。


例如：

没有录工单，
即使数据同步完全正常，
记录仍然不会显示。

数据没有同步，
即使工单已经正常录入，
记录也可能不会显示。

因此这是两个原因。


============================================================
verdict = pass
============================================================

满足：

1. rebuilt cause 由允许素材支持
2. 没新增原因
3. 没新增规则
4. 如果发生合并，只合并了语义等价原因
5. action 有允许素材支持
6. source keys 正确


============================================================
cause_overreach
============================================================

Cause 新增了：

- 处理动作
- 时间
- 条件
- 产品规则
- 新故障机制

例如：

允许：

短信签名尚未完成报备

重建：

短信签名尚未完成报备，
需等待次日重新提交

后半段是 Action。

→ cause_overreach


============================================================
action_overreach
============================================================

Cause 正确，
但 Action 新增了允许素材不存在的：

- 操作
- 菜单
- 时间
- 条件
- 流程

→ action_overreach


============================================================
bad_merge
============================================================

只要一个 rebuilt Cause
合并了两个或以上可以独立成立的故障机制：

→ bad_merge

即使：

- 都属于同一模块
- 都属于同一配置问题
- 都有相同处理结果
- 都导致相同最终现象

也不能合并。


============================================================
source_key_mismatch
============================================================

Source keys：

- 引用了不支持当前 Cause 的来源
- 出现允许素材外的 key
- 因错误合并把多个独立 Cause 的 key 混在一起

则：

source_key_mismatch

如果核心问题首先是错误合并，
优先 verdict = bad_merge。


============================================================
corrected_cause
============================================================

只有可以通过“删除越权文字”
得到一个安全单一 Cause 时填写。

例如：

原：

短信签名尚未完成报备，需要次日重新提交

corrected：

短信签名尚未完成报备


如果问题属于 bad_merge：

不要尝试把两个原因重新写成一个。

corrected_cause = null

因为下一步应该恢复为多个独立 Cause。


============================================================
corrected_action
============================================================

只保留允许素材明确支持的 Action。

没有则 null。


============================================================
重要：不要使用“同一类”作为合并理由
============================================================

以下理由都是错误的：

“二者都属于音频配置问题”

“二者都属于数据同步问题”

“二者都是账号配置”

“二者都导致同一现象”

这些只能说明它们相关，
不能证明它们是同一个 Cause。


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