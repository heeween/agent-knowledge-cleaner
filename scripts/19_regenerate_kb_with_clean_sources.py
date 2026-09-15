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

SOURCE_QUALITY_FILE = (
    OUTPUT_DIR
    / "kb_source_quality.xlsx"
)

ALIGNMENT_FILE = (
    OUTPUT_DIR
    / "issue_qa_alignments.xlsx"
)

CLUSTER_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entries_regenerated.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_regenerated.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class KnowledgeEntry(BaseModel):

    canonical_question: str

    answer: str

    solution_steps: list[str]

    notes: list[str]

    crm_module: str

    crm_feature: str

    problem_type: str

    knowledge_confidence: float

    source_issue_keys: list[str]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在重新生成一条 CRM RAG 知识。

之前版本因为引用了：
- 答非所问的 source
- 没有实际答案的 source
- 不可靠 source

而被拦截。

现在输入给你的 source 已经过 QA Alignment 检查。

你的任务：

只能使用下面提供的“可用 source”，
重新生成知识答案。


============================================================
最高优先级规则
============================================================

1. 严格回答 canonical_question。

2. 只能使用输入 source 中明确存在的信息。

禁止：

- 根据之前知识猜测
- 根据 CRM 常识补充
- 发明产品规则
- 发明菜单路径
- 发明处理步骤
- 发明原因
- 发明限制条件

3. 不要恢复已经被过滤掉的坏 source 内容。

你不知道被过滤的 source 说了什么，
也不需要猜。


============================================================
如果证据不足
============================================================

如果现有 source 只能回答问题的一部分：

只回答能确认的部分。

不要为了让答案完整而补全。


例如：

问题：
为什么 CRM 登录失败？如何处理？

source 只确认：

“提供手机号后，由后台创建账号，
账号和密码均为手机号。”

那么只能表达：

“若当前尚未创建对应账号，可提供手机号，
由后台创建账号；该案例中新建账号密码与手机号一致。”

不能进一步发明：

- 找回密码菜单
- 管理员重置路径
- 登录失败的所有原因


============================================================
故障排查类问题
============================================================

如果来源只支持某一个可能原因：

不要写成：

“原因包括……”

应该写：

“已有案例显示，可能与 XXX 有关。”

只有 source 明确证明是确定规则时，
才可以使用确定表达。


============================================================
temporary / unresolved
============================================================

如果 source 的：

temporal_status = temporary

或：

resolution = unresolved / partial

不得包装成稳定产品规则。

应该：

- 限定为案例
- 降低 confidence
- 必要时写 notes


============================================================
partially_aligned source
============================================================

输入中可能存在：

alignment = partially_aligned

这种 source 只能使用与 canonical_question
明确相关的那部分。

不能将 source 中其他内容一起融合。


============================================================
canonical_question
============================================================

原则上保持输入 canonical_question。

可以做很轻微的语言整理，
但不得改变问题范围。


============================================================
answer
============================================================

要求：

- 直接回答
- 中文自然
- 不使用客服聊天口吻
- 不写“这边”“我看一下”
- 不写无依据内容
- 不为了完整而扩写


============================================================
solution_steps
============================================================

只有来源存在明确操作步骤才填写。

没有：

[]


============================================================
notes
============================================================

适合记录：

- 当前只是已知案例
- temporary
- unresolved
- 特定前提条件
- 尚需技术确认

没有则 []。


============================================================
source_issue_keys
============================================================

只能填写实际用于答案的 issue_key。

必须来自本次输入 source。

禁止输出不存在的 issue_key。


============================================================
knowledge_confidence
============================================================

0 ~ 1。

单一来源：
不要无理由给极高置信度。

来源 temporary / partial / unresolved：
适当降低。

多个稳定来源互相支持：
可以提高。


============================================================
输出
============================================================

只输出合法 JSON object。

格式：

{
  "canonical_question": "问题",
  "answer": "答案",
  "solution_steps": [],
  "notes": [],
  "crm_module": "模块",
  "crm_feature": "功能",
  "problem_type": "类型",
  "knowledge_confidence": 0.0,
  "source_issue_keys": []
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


def bool_value(value):

    if isinstance(value, bool):
        return value

    text = clean_text(
        value
    ).lower()

    return text in {
        "true",
        "1",
        "yes",
    }


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
    cluster_row,
    members,
):

    blocks = []

    for idx, row in enumerate(
        members,
        start=1,
    ):

        blocks.append(
            f"""
----------------------------------------
SOURCE {idx}

issue_key:
{clean_text(row.get("issue_key"))}

alignment:
{clean_text(row.get("alignment"))}

alignment_confidence:
{clean_text(row.get("alignment_confidence"))}

question:
{clean_text(row.get("question"))}

question_normalized:
{clean_text(row.get("question_normalized"))}

description:
{clean_text(row.get("description"))}

answer:
{clean_text(row.get("answer"))}

solution:
{clean_text(row.get("solution"))}

crm_module:
{clean_text(row.get("crm_module"))}

crm_feature:
{clean_text(row.get("crm_feature"))}

problem_type:
{clean_text(row.get("problem_type"))}

resolution:
{clean_text(row.get("resolution"))}

knowledge_value:
{clean_text(row.get("knowledge_value"))}

temporal_status:
{clean_text(row.get("temporal_status"))}
""".strip()
        )

    return f"""
请重新生成该 CRM 知识条目。

cluster_id:
{clean_text(cluster_row.get("cluster_id"))}

canonical_question:
{clean_text(cluster_row.get("canonical_question"))}

原 answer_relation:
{clean_text(cluster_row.get("answer_relation"))}

本次只允许使用以下经过过滤的 source：

{chr(10).join(blocks)}
"""


# ============================================================
# API
# ============================================================

def regenerate_entry(
    cluster_row,
    members,
):

    client = build_client()

    cluster_id = clean_text(
        cluster_row.get(
            "cluster_id"
        )
    )

    prompt = build_prompt(
        cluster_row,
        members,
    )

    valid_keys = {
        clean_text(
            row.get(
                "issue_key"
            )
        )
        for row in members
    }

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

            content = (
                response
                .choices[0]
                .message
                .content
            )

            data = json.loads(
                content
            )

            result = (
                KnowledgeEntry
                .model_validate(
                    data
                )
            )

            # ------------------------------------------------
            # 防止乱造 issue key
            # ------------------------------------------------

            result.source_issue_keys = [
                key
                for key
                in result.source_issue_keys
                if key in valid_keys
            ]

            return {
                "cluster_id":
                    cluster_id,

                "regeneration_status":
                    "generated",

                "canonical_question":
                    result.canonical_question,

                "answer":
                    result.answer,

                "solution_steps":
                    result.solution_steps,

                "notes":
                    result.notes,

                "crm_module":
                    result.crm_module,

                "crm_feature":
                    result.crm_feature,

                "problem_type":
                    result.problem_type,

                "knowledge_confidence":
                    result.knowledge_confidence,

                "source_issue_keys":
                    result.source_issue_keys,

                "usable_source_count":
                    len(members),

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
    no_source_rows,
):

    generated_rows = []

    if OUTPUT_JSONL.exists():

        with OUTPUT_JSONL.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = (
                    line.strip()
                )

                if not line:
                    continue

                try:
                    generated_rows.append(
                        json.loads(
                            line
                        )
                    )
                except Exception:
                    pass

    generated_df = pd.DataFrame(
        generated_rows
    )

    no_source_df = pd.DataFrame(
        no_source_rows
    )

    # --------------------------------------------------------
    # 展平 list
    # --------------------------------------------------------

    if not generated_df.empty:

        for col in [
            "solution_steps",
            "notes",
            "source_issue_keys",
        ]:

            if col in generated_df.columns:

                generated_df[col] = (
                    generated_df[col]
                    .apply(
                        lambda x:
                            " | ".join(
                                str(v)
                                for v in x
                            )
                            if isinstance(
                                x,
                                list,
                            )
                            else clean_text(x)
                    )
                )

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "regenerated_count",
                "value":
                    len(generated_df),
            },
            {
                "metric":
                    "no_valid_source_count",
                "value":
                    len(no_source_df),
            },
            {
                "metric":
                    "total_original_regenerate_count",
                "value":
                    (
                        len(generated_df)
                        + len(no_source_df)
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

        if not generated_df.empty:

            generated_df.to_excel(
                writer,
                sheet_name="regenerated",
                index=False,
            )

        if not no_source_df.empty:

            no_source_df.to_excel(
                writer,
                sheet_name="no_valid_source",
                index=False,
            )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 9.5 - Regenerate KB with clean sources"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 读取需要重新生成的 16 条
    # --------------------------------------------------------

    regenerate_df = pd.read_excel(
        SOURCE_QUALITY_FILE,
        sheet_name="regenerate",
    )

    alignment_df = pd.read_excel(
        ALIGNMENT_FILE,
        sheet_name="all_issues",
    )

    member_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    print(
        f"Original regenerate count: "
        f"{len(regenerate_df)}"
    )

    processed = (
        load_processed()
    )

    print(
        f"Already processed: "
        f"{len(processed)}"
    )

    # --------------------------------------------------------
    # alignment lookup
    # --------------------------------------------------------

    align_cols = [
        "issue_key",
        "alignment",
        "alignment_confidence",
        "usable_for_kb",
    ]

    alignment_small = (
        alignment_df[
            align_cols
        ]
        .drop_duplicates(
            subset=[
                "issue_key"
            ]
        )
    )

    # --------------------------------------------------------
    # 把 alignment 合并回原 Cluster member
    # --------------------------------------------------------

    member_with_alignment = (
        member_df.merge(
            alignment_small,
            on="issue_key",
            how="left",
        )
    )

    tasks = []

    no_source_rows = []

    for _, cluster_row in (
        regenerate_df.iterrows()
    ):

        cluster_id = clean_text(
            cluster_row.get(
                "cluster_id"
            )
        )

        # ----------------------------------------------------
        # 取这个 Cluster 所有成员
        # ----------------------------------------------------

        cluster_members = (
            member_with_alignment[
                member_with_alignment[
                    "cluster_id"
                ]
                .astype(str)
                == cluster_id
            ]
            .copy()
        )

        # ----------------------------------------------------
        # 只允许 usable_for_kb=True
        # ----------------------------------------------------

        usable_members = (
            cluster_members[
                cluster_members[
                    "usable_for_kb"
                ]
                .apply(
                    bool_value
                )
            ]
            .copy()
        )

        # ----------------------------------------------------
        # 再做一层硬过滤：
        # misaligned/no_answer/uncertain 永远不能用
        # ----------------------------------------------------

        usable_members = (
            usable_members[
                ~usable_members[
                    "alignment"
                ]
                .isin(
                    [
                        "misaligned",
                        "no_answer",
                        "uncertain",
                    ]
                )
            ]
        )

        # ----------------------------------------------------
        # 没有可靠 source
        # ----------------------------------------------------

        if usable_members.empty:

            no_source_rows.append(
                {
                    "cluster_id":
                        cluster_id,

                    "canonical_question":
                        clean_text(
                            cluster_row.get(
                                "canonical_question"
                            )
                        ),

                    "original_source_quality":
                        clean_text(
                            cluster_row.get(
                                "source_quality"
                            )
                        ),

                    "original_verdict":
                        clean_text(
                            cluster_row.get(
                                "verdict"
                            )
                        ),

                    "status":
                        "NO_VALID_SOURCE",

                    "reason":
                        (
                            "过滤 misaligned / "
                            "no_answer / uncertain "
                            "来源后，没有可用于生成知识的答案来源。"
                        ),

                    "original_source_issue_keys":
                        clean_text(
                            cluster_row.get(
                                "source_issue_keys"
                            )
                        ),

                    "bad_source_issue_keys":
                        clean_text(
                            cluster_row.get(
                                "bad_source_issue_keys"
                            )
                        ),
                }
            )

            continue

        if cluster_id in processed:
            continue

        members = (
            usable_members
            .to_dict(
                orient="records"
            )
        )

        tasks.append(
            (
                cluster_row,
                members,
            )
        )

    # --------------------------------------------------------
    # 打印分流结果
    # --------------------------------------------------------

    print(
        f"Can regenerate: "
        f"{len(tasks)}"
    )

    print(
        f"No valid source: "
        f"{len(no_source_rows)}"
    )

    # --------------------------------------------------------
    # 调用 LLM
    # --------------------------------------------------------

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
                        regenerate_entry,
                        cluster_row,
                        members,
                    ):
                    clean_text(
                        cluster_row.get(
                            "cluster_id"
                        )
                    )

                    for (
                        cluster_row,
                        members,
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

                        print()
                        print(
                            f"[FAILED] "
                            f"{cluster_id}"
                        )
                        print(
                            type(exc).__name__,
                            exc,
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
                        f"[{completed}/"
                        f"{total}] "
                        f"| ETA "
                        f"{eta/60:.1f} min"
                    )

    export_excel(
        no_source_rows
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